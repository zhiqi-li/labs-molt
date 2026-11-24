# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime FLA Triton autotune coarsening to prevent per-length compilation spikes.

In ``flash-linear-attention`` (FLA 0.4.x), ``causal_conv1d_fwd`` / ``causal_conv1d_bwd``
pass ``NB = triton.cdiv(B*T, 1024)`` into ``@triton.autotune(key=['D', 'W', 'NB'])``,
and ``l2norm_fwd`` / ``l2norm_bwd`` pass ``NB = triton.cdiv(T, 2048 * 32)`` where
``T = B * S_local * H`` (so ``NB = cdiv(S_local, 2048)`` for H=32).

``NB`` is never referenced inside the kernel math — it is purely an autotune bucket key.
On variable-length RL trajectories up to 256K tokens, those fine-grained buckets trigger
25-32 separate Triton autotune waves (45-80 s each, ~27 min per cold restart). Coarsening
``NB`` to power-of-2 buckets of 16,384 local tokens collapses ~32 waves to 1-3 while
producing mathematically identical outputs.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False


def _coarse_nb(tokens: int, bucket_tokens: int) -> int:
    import triton

    return triton.next_power_of_2(max(1, triton.cdiv(tokens, bucket_tokens)))


def _patch_causal_conv1d(bucket_tokens: int) -> bool:
    try:
        from einops import rearrange
        import torch
        import triton
        import fla.modules.conv.triton.ops as conv_ops
        from fla.ops.utils import prepare_chunk_indices
    except Exception:
        return False

    if getattr(conv_ops, "_molt_coarse_nb_patched", False):
        return True

    orig_fwd = conv_ops.causal_conv1d_fwd
    orig_bwd = conv_ops.causal_conv1d_bwd

    def _patched_fwd(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        residual: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        activation: str | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        BT: int = 64,
    ):
        shape = x.shape
        if x.shape[-1] != weight.shape[0]:
            x = rearrange(x, "b t ... -> b t (...)")
        B, T, D = x.shape[0], x.shape[1], weight.shape[0]
        W = weight.shape[1]
        stride_x_n, stride_x_t, stride_x_d = x.stride()
        BW = triton.next_power_of_2(W)
        if cu_seqlens is not None and chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
        NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
        NB = _coarse_nb(B * T, bucket_tokens)

        y = torch.empty_like(x, memory_format=torch.contiguous_format)

        def grid(meta):
            return (triton.cdiv(D, meta["BD"]), NT, B)

        conv_ops.causal_conv1d_fwd_kernel[grid](
            x=x,
            y=y,
            weight=weight,
            bias=bias,
            residual=residual,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            chunk_indices=chunk_indices,
            B=B,
            T=T,
            D=D,
            W=W,
            BT=BT,
            BW=BW,
            NB=NB,
            stride_x_n=stride_x_n,
            stride_x_t=stride_x_t,
            stride_x_d=stride_x_d,
            ACTIVATION=activation,
        )
        final_state = None
        if output_final_state:
            final_state = conv_ops.causal_conv1d_update_states(
                x=x,
                state_len=W,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
            )
        return y.view(shape), final_state

    def _patched_bwd(
        x: torch.Tensor,
        dy: torch.Tensor,
        dht: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        activation: str | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        BT: int = 64,
    ):
        shape = x.shape
        if x.shape[-1] != weight.shape[0]:
            x = rearrange(x, "b t ... -> b t (...)")
        B, T, D = x.shape
        W = weight.shape[1] if weight is not None else None
        stride_x_n, stride_x_t, stride_x_d = x.stride()
        BW = triton.next_power_of_2(W)
        if cu_seqlens is not None and chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
        NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
        NB = _coarse_nb(B * T, bucket_tokens)

        y = None
        if activation is not None:
            y, _ = _patched_fwd(
                x=x,
                weight=weight,
                bias=bias,
                residual=None,
                initial_state=initial_state,
                activation=None,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                output_final_state=False,
                chunk_indices=chunk_indices,
                BT=BT,
            )
        dx = torch.empty_like(x)
        dw = weight.new_empty(B * NT, *weight.shape, dtype=torch.float) if weight is not None else None
        db = bias.new_empty(B * NT, *bias.shape, dtype=torch.float) if bias is not None else None
        dr = dy if residual is not None else None
        stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()

        def grid(meta):
            return (triton.cdiv(D, meta["BD"]), NT, B)

        conv_ops.causal_conv1d_bwd_kernel[grid](
            x=x,
            y=y,
            weight=weight,
            initial_state=initial_state,
            dht=dht,
            dy=dy,
            dx=dx,
            dw=dw,
            db=db,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            B=B,
            T=T,
            D=D,
            W=W,
            BT=BT,
            BW=BW,
            NB=NB,
            stride_x_n=stride_x_n,
            stride_x_t=stride_x_t,
            stride_x_d=stride_x_d,
            stride_dx_n=stride_dx_n,
            stride_dx_t=stride_dx_t,
            stride_dx_d=stride_dx_d,
            ACTIVATION=activation,
        )
        if weight is not None:
            dw = dw.sum(0).to(weight)
        if bias is not None:
            db = db.sum(0).to(bias)

        dh0 = None
        if initial_state is not None:
            dh0 = conv_ops.compute_dh0_triton(
                dy=dy,
                y=y,
                weight=weight,
                initial_state=initial_state,
                activation=activation,
                cu_seqlens=cu_seqlens,
            )
        return dx.view(shape), dw, db, dr, dh0

    from fla.utils import input_guard

    conv_ops.causal_conv1d_fwd = input_guard(no_guard_contiguous=["x"])(_patched_fwd)
    conv_ops.causal_conv1d_bwd = _patched_bwd
    conv_ops._molt_orig_causal_conv1d_fwd = orig_fwd
    conv_ops._molt_orig_causal_conv1d_bwd = orig_bwd
    conv_ops._molt_coarse_nb_patched = True
    return True


def _patch_l2norm(bucket_tokens: int) -> bool:
    try:
        import torch
        import triton
        import fla.modules.l2norm as l2mod
        import fla.ops.gated_delta_rule.chunk as gdr_chunk
    except Exception:
        return False

    if getattr(l2mod, "_molt_coarse_nb_patched", False):
        return True

    # l2norm flattens [B, S_local, H, D] -> [B*S_local*H, D]. Use 32 * bucket_tokens so
    # an H=32 head count shares the same local-sequence bucket cadence as causal_conv1d.
    l2_bucket = bucket_tokens * 32

    def _patched_l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
        x_shape_og = x.shape
        x = x.view(-1, x.shape[-1])
        y = torch.empty_like(x) if output_dtype is None else torch.empty_like(x, dtype=output_dtype)
        assert y.stride(-1) == 1
        T, D = x.shape[0], x.shape[-1]
        max_fused = 65536 // x.element_size()
        BD = min(max_fused, triton.next_power_of_2(D))
        if D > BD:
            raise RuntimeError("This layer doesn't support feature dim >= 64KB.")
        rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
        if D <= 512:
            NB = _coarse_nb(T, l2_bucket)

            def grid(meta):
                return (triton.cdiv(T, meta["BT"]),)

            l2mod.l2norm_fwd_kernel[grid](x=x, y=y, rstd=rstd, eps=eps, T=T, D=D, BD=BD, NB=NB)
        else:
            l2mod.l2norm_fwd_kernel1[(T,)](x=x, y=y, rstd=rstd, eps=eps, D=D, BD=BD)
        return y.view(x_shape_og), rstd.view(x_shape_og[:-1])

    def _patched_l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6):
        y_shape_og = y.shape
        y = y.view(-1, dy.shape[-1])
        dy = dy.view(-1, dy.shape[-1])
        assert dy.shape == y.shape
        dx = torch.empty_like(y)
        T, D = y.shape[0], y.shape[-1]
        max_fused = 65536 // y.element_size()
        BD = min(max_fused, triton.next_power_of_2(D))
        if D > BD:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        if D <= 512:
            NB = _coarse_nb(T, l2_bucket)

            def grid(meta):
                return (triton.cdiv(T, meta["BT"]),)

            l2mod.l2norm_bwd_kernel[grid](y=y, rstd=rstd, dy=dy, dx=dx, eps=eps, T=T, D=D, BD=BD, NB=NB)
        else:
            l2mod.l2norm_bwd_kernel1[(T,)](y=y, rstd=rstd, dy=dy, dx=dx, eps=eps, D=D, BD=BD)
        return dx.view(y_shape_og)

    l2mod.l2norm_fwd = _patched_l2norm_fwd
    l2mod.l2norm_bwd = _patched_l2norm_bwd
    # chunk.py imported l2norm_fwd / l2norm_bwd by value at module load time.
    gdr_chunk.l2norm_fwd = _patched_l2norm_fwd
    gdr_chunk.l2norm_bwd = _patched_l2norm_bwd
    l2mod._molt_coarse_nb_patched = True
    return True


def install_fla_coarse_nb_bucketing() -> bool:
    """Install coarse power-of-2 NB bucketing for FLA causal_conv1d and l2norm kernels."""
    global _INSTALLED
    if _INSTALLED or os.environ.get("MOLT_FLA_COARSE_NB", "1") != "1":
        return _INSTALLED
    bucket_tokens = int(os.environ.get("MOLT_FLA_NB_BUCKET_TOKENS", "1048576"))
    conv_ok = _patch_causal_conv1d(bucket_tokens)
    l2_ok = _patch_l2norm(bucket_tokens)
    _INSTALLED = conv_ok or l2_ok
    if _INSTALLED:
        logger.info(
            "[FLA] Installed coarse NB autotune bucketing (bucket_tokens=%d, conv=%s, l2norm=%s)",
            bucket_tokens,
            conv_ok,
            l2_ok,
        )
    return _INSTALLED
