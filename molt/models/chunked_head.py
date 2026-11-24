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

"""Chunked LM-head log-prob computation that avoids materializing [S_local, V] logits.

For large-vocabulary models (e.g. Qwen3.5 with V=248,320) on long sequences (S=180K-256K),
an unchunked ``nn.Linear`` LM head materializes a full ``[B, S_local, V]`` bf16 logits tensor
(~19 GiB at CP=4, ~10-14 GiB at CP=8) and autograd allocates a matching ``grad_logits`` tensor
during backward (~38 GiB combined at CP=4). This module intercepts ``lm_head`` inside
``Actor.forward`` (via a ContextVar, leaving all external callers unchanged) and computes
exact token log-probs and gradients one row-chunk at a time, optionally restricted to
``action_mask`` positions.
"""

from contextvars import ContextVar
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

_DEFAULT_CHUNK_ROWS = 1024  # 1024 * 248320 * 4 bytes ~= 1.0 GiB fp32 working buffer

_chunked_head_consumer: ContextVar[Optional[nn.Module]] = ContextVar("molt_chunked_lm_head", default=None)


def _chunk_rows() -> int:
    raw = os.environ.get("MOLT_LM_HEAD_CHUNK_ROWS")
    if not raw:
        return _DEFAULT_CHUNK_ROWS
    value = int(raw)
    if value <= 0:
        raise ValueError(f"MOLT_LM_HEAD_CHUNK_ROWS must be positive, got {value}")
    return value


def _compute_chunk_logits(
    hidden_chunk: torch.Tensor,
    weight: torch.Tensor,
    temperature: float,
    cast_logits_to_input_dtype: bool,
) -> torch.Tensor:
    if cast_logits_to_input_dtype:
        # Match stock nn.Linear(bf16, bf16) -> .float() rounding bit-for-bit so
        # train-vs-vLLM log-probs stay identical when vLLM uses its default bf16 head.
        logits = F.linear(hidden_chunk, weight).float()
    elif hidden_chunk.is_cuda and hidden_chunk.dtype != torch.float32:
        logits = torch.mm(hidden_chunk, weight.t(), out_dtype=torch.float32)
    else:
        logits = hidden_chunk.float() @ weight.float().t()
    if temperature != 1.0:
        logits.div_(temperature)
    return logits


def _scatter_grad_logits(grad_logits: torch.Tensor, ids_chunk: torch.Tensor, g_chunk: torch.Tensor) -> None:
    """Scatter per-target log-prob gradients into ``grad_logits``.

    When score centering passes ``[sampled_id, top_1, ..., top_k]`` (shape ``[R, 1+k]``),
    column 0 can repeat one of the distinct top-k ids in columns ``1..k``. Splitting into
    two vectorized scatters keeps accumulation order deterministic without a k-step Python loop.
    """
    grad_logits.scatter_add_(1, ids_chunk[:, :1], g_chunk[:, :1])
    if ids_chunk.shape[1] > 1:
        grad_logits.scatter_add_(1, ids_chunk[:, 1:], g_chunk[:, 1:])


class ChunkedHeadLogProbs(torch.autograd.Function):
    """Chunked ``lm_head`` + log-softmax + gather with optional active-row filtering."""

    @staticmethod
    def forward(
        ctx,
        flat_hidden: torch.Tensor,
        weight: torch.Tensor,
        flat_ids: torch.Tensor,
        active_idx: Optional[torch.Tensor],
        temperature: float,
        chunk_rows: int,
        cast_logits_to_input_dtype: bool,
    ) -> torch.Tensor:
        total_rows = flat_hidden.shape[0]
        num_targets = flat_ids.shape[1]
        log_probs = flat_hidden.new_zeros((total_rows, num_targets), dtype=torch.float32)

        if active_idx is not None:
            num_active = active_idx.numel()
            if num_active > 0:
                sel_hidden = flat_hidden.index_select(0, active_idx)
                sel_ids = flat_ids.index_select(0, active_idx)
                sel_log_probs = flat_hidden.new_empty((num_active, num_targets), dtype=torch.float32)
                sel_lse = flat_hidden.new_empty((num_active,), dtype=torch.float32)
                for start in range(0, num_active, chunk_rows):
                    rows = slice(start, min(start + chunk_rows, num_active))
                    logits = _compute_chunk_logits(
                        sel_hidden[rows], weight, temperature, cast_logits_to_input_dtype
                    )
                    lse = logits.logsumexp(-1)
                    sel_lse[rows] = lse
                    sel_log_probs[rows] = logits.gather(1, sel_ids[rows]) - lse[:, None]
                log_probs.index_copy_(0, active_idx, sel_log_probs)
            else:
                sel_lse = flat_hidden.new_empty((0,), dtype=torch.float32)
            ctx.has_active_idx = True
            ctx.save_for_backward(flat_hidden, weight, flat_ids, active_idx, sel_lse)
        else:
            all_lse = flat_hidden.new_empty((total_rows,), dtype=torch.float32)
            for start in range(0, total_rows, chunk_rows):
                rows = slice(start, min(start + chunk_rows, total_rows))
                logits = _compute_chunk_logits(
                    flat_hidden[rows], weight, temperature, cast_logits_to_input_dtype
                )
                lse = logits.logsumexp(-1)
                all_lse[rows] = lse
                log_probs[rows] = logits.gather(1, flat_ids[rows]) - lse[:, None]
            ctx.has_active_idx = False
            ctx.save_for_backward(
                flat_hidden, weight, flat_ids, torch.empty(0, dtype=torch.long, device=flat_hidden.device), all_lse
            )

        ctx.temperature = temperature
        ctx.chunk_rows = chunk_rows
        ctx.cast_logits_to_input_dtype = cast_logits_to_input_dtype
        return log_probs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_log_probs: torch.Tensor):
        flat_hidden, weight, flat_ids, active_idx, lse = ctx.saved_tensors
        temperature = ctx.temperature
        chunk_rows = ctx.chunk_rows
        cast_logits_to_input_dtype = ctx.cast_logits_to_input_dtype

        grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)

        if ctx.has_active_idx:
            grad_hidden = torch.zeros_like(flat_hidden)
            num_active = active_idx.numel()
            if num_active > 0:
                sel_hidden = flat_hidden.index_select(0, active_idx)
                sel_ids = flat_ids.index_select(0, active_idx)
                sel_grad = grad_log_probs.index_select(0, active_idx).float()
                grad_sel_hidden = torch.empty_like(sel_hidden)
                for start in range(0, num_active, chunk_rows):
                    rows = slice(start, min(start + chunk_rows, num_active))
                    h_chunk = sel_hidden[rows]
                    g_chunk = sel_grad[rows]
                    logits = _compute_chunk_logits(h_chunk, weight, temperature, cast_logits_to_input_dtype)
                    grad_logits = logits.sub_(lse[rows, None]).exp_().mul_(-g_chunk.sum(-1, keepdim=True))
                    _scatter_grad_logits(grad_logits, sel_ids[rows], g_chunk)
                    if temperature != 1.0:
                        grad_logits.div_(temperature)
                    grad_logits_in = grad_logits.to(h_chunk.dtype)
                    grad_sel_hidden[rows] = grad_logits_in @ weight
                    if grad_logits_in.is_cuda and grad_logits_in.dtype != torch.float32:
                        torch.addmm(
                            grad_weight,
                            grad_logits_in.t(),
                            h_chunk,
                            out_dtype=torch.float32,
                            out=grad_weight,
                        )
                    else:
                        grad_weight += grad_logits_in.t().float() @ h_chunk.float()
                grad_hidden.index_copy_(0, active_idx, grad_sel_hidden)
            return grad_hidden, grad_weight.to(weight.dtype), None, None, None, None, None

        total_rows = flat_hidden.shape[0]
        grad_hidden = torch.empty_like(flat_hidden)
        grad_lp_fp32 = grad_log_probs.float()
        for start in range(0, total_rows, chunk_rows):
            rows = slice(start, min(start + chunk_rows, total_rows))
            h_chunk = flat_hidden[rows]
            g_chunk = grad_lp_fp32[rows]
            logits = _compute_chunk_logits(h_chunk, weight, temperature, cast_logits_to_input_dtype)
            grad_logits = logits.sub_(lse[rows, None]).exp_().mul_(-g_chunk.sum(-1, keepdim=True))
            _scatter_grad_logits(grad_logits, flat_ids[rows], g_chunk)
            if temperature != 1.0:
                grad_logits.div_(temperature)
            grad_logits_in = grad_logits.to(h_chunk.dtype)
            grad_hidden[rows] = grad_logits_in @ weight
            if grad_logits_in.is_cuda and grad_logits_in.dtype != torch.float32:
                torch.addmm(
                    grad_weight,
                    grad_logits_in.t(),
                    h_chunk,
                    out_dtype=torch.float32,
                    out=grad_weight,
                )
            else:
                grad_weight += grad_logits_in.t().float() @ h_chunk.float()
        return grad_hidden, grad_weight.to(weight.dtype), None, None, None, None, None


def can_use_chunked_head(model: nn.Module, *, return_entropy: bool) -> bool:
    """True when ``model.lm_head`` can safely bypass full-vocab logits materialization."""
    if return_entropy or os.environ.get("MOLT_CHUNKED_LM_HEAD", "1") != "1":
        return False
    head = getattr(model, "lm_head", None)
    if head is None or type(head) is not nn.Linear or head.bias is not None:
        return False
    if getattr(head, "fp32", False):
        # Defer to external fp32-head patchers (e.g. ember_bridge_rl.frameworks.molt.precision).
        return False
    # Check TP sharding: TP shards lm_head along vocab dim (-1), which uses vocab-parallel path.
    # Under FSDP2 without TP, weight may be a DTensor before root forward unshards it, so check
    # placements rather than isinstance(DTensor).
    weight = head.weight
    if isinstance(weight, DTensor):
        mesh_names = getattr(weight.device_mesh, "mesh_dim_names", None) or ()
        if "tp" in mesh_names and weight.device_mesh["tp"].size() > 1:
            return False
    return True


def ensure_chunked_head_hook(head: nn.Linear) -> None:
    """Wrap ``head.forward`` once so it yields hidden states only while ``_chunked_head_consumer`` names it."""
    if getattr(head, "_molt_chunked_hooked", False):
        return
    orig_forward = head.forward

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if _chunked_head_consumer.get() is head:
            return hidden_states
        return orig_forward(hidden_states)

    head.forward = forward
    head._molt_chunked_hooked = True


def chunked_head_log_probs(
    hidden_states: torch.Tensor,
    head: nn.Linear,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    active_mask: Optional[torch.Tensor] = None,
    cast_logits_to_input_dtype: bool = True,
) -> torch.Tensor:
    """Compute selected token log-probs directly from final ``hidden_states`` and ``head.weight``."""
    weight = head.weight
    if isinstance(weight, DTensor):
        raise RuntimeError("Chunked LM head expects unsharded lm_head.weight after root FSDP forward")
    if hidden_states.shape[-1] != head.in_features:
        raise RuntimeError(
            f"Chunked LM head expected [..., {head.in_features}] hidden states, got {tuple(hidden_states.shape)}"
        )
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).to(weight.dtype)
    flat_ids = labels.reshape(flat_hidden.shape[0], -1)
    active_idx = None
    if active_mask is not None:
        active_idx = active_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    result = ChunkedHeadLogProbs.apply(
        flat_hidden,
        weight,
        flat_ids,
        active_idx,
        float(temperature),
        _chunk_rows(),
        cast_logits_to_input_dtype,
    )
    several = labels.numel() != flat_hidden.shape[0]
    return result.view(*hidden_states.shape[:-1], -1) if several else result.view(hidden_states.shape[:-1])
