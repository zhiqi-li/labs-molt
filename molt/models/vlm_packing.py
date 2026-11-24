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

"""VLM packed-sequence preparation for AutoModel block-diagonal context parallelism.

Packs ``B`` right-padded multimodal trajectories into a single ``[1, S_packed]`` stream
with 1-based per-document IDs (``_packed_seq_ids``), per-document next-token rolls, and
independently computed per-document mRoPE ``position_ids``. AutoModel's Qwen3.5 CP hook
(``prepare_model_inputs_for_cp``) consumes ``_packed_seq_ids`` to activate contiguous
block-diagonal varlen CP for both softmax attention and FLA GatedDeltaNet.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F

_CP_VARLEN_CONFIGURED = False


def ensure_blockdiag_cp_varlen_configured() -> None:
    """Configure AutoModel's block-diagonal CP attention backend and KV exchange once."""
    global _CP_VARLEN_CONFIGURED
    if _CP_VARLEN_CONFIGURED:
        return
    try:
        from nemo_automodel.components.distributed.blockdiag_cp import configure_cp_varlen

        attn_backend = os.environ.get("MOLT_BLOCKDIAG_ATTN_BACKEND", "flash")
        kv_exchange = os.environ.get("MOLT_BLOCKDIAG_KV_EXCHANGE", "allgather")
        configure_cp_varlen(attn_backend=attn_backend, kv_exchange=kv_exchange)
        _CP_VARLEN_CONFIGURED = True
    except ImportError:
        pass


def _spatial_merge_area(model: Any) -> int:
    config = getattr(model, "config", None)
    vision_cfg = getattr(config, "vision_config", None)
    merge_size = getattr(vision_cfg, "spatial_merge_size", None)
    if merge_size is None:
        visual = getattr(getattr(model, "model", None), "visual", None)
        merge_size = getattr(visual, "spatial_merge_size", 2)
    merge_size = int(merge_size or 2)
    return merge_size * merge_size


def _slice_grid_rows_for_doc(
    grid_thw: Optional[torch.Tensor],
    cursor: int,
    expected_tokens: int,
    merge_area: int,
    modality: str,
) -> tuple[Optional[torch.Tensor], int]:
    """Advance ``cursor`` across ``grid_thw`` rows whose merged token counts sum to ``expected_tokens``."""
    if expected_tokens == 0:
        return None, cursor
    if grid_thw is None:
        raise ValueError(f"VLM packing: document has {expected_tokens} {modality} tokens but no grid_thw")
    total_rows = grid_thw.shape[0]
    acc = 0
    end = cursor
    per_row = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2] // merge_area).tolist()
    while end < total_rows and acc < expected_tokens:
        acc += int(per_row[end])
        end += 1
    if acc != expected_tokens:
        raise ValueError(
            f"VLM packing: {modality} placeholder token count ({expected_tokens}) does not align with "
            f"grid_thw row boundaries starting at row {cursor} (accumulated {acc})."
        )
    return grid_thw[cursor:end], end


def _pad_mrope_positions(position_ids: torch.Tensor, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return position_ids
    last = position_ids[..., -1:].contiguous()
    inc = torch.arange(1, pad_len + 1, device=position_ids.device, dtype=position_ids.dtype)
    view_shape = [1] * (position_ids.ndim - 1) + [pad_len]
    return torch.cat([position_ids, last + inc.view(*view_shape)], dim=-1)


def pack_vlm_batch_for_cp(
    model: Any,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_inputs: dict[str, Any],
    *,
    image_token_id: Optional[int],
    video_token_id: Optional[int],
    pad_token_id: int,
    align_multiple: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack a ``[B, S]`` VLM batch into a ``[1, S_aligned]`` block-diagonal CP stream.

    Returns:
        ``(packed_ids, rolled_packed, packed_pos_ids, packed_seq_ids, indices, packed_len)``
        where ``indices`` maps flat ``[B*S]`` positions of real tokens into ``[0..packed_len)``.
    """
    ensure_blockdiag_cp_varlen_configured()

    batch, seqlen = sequences.shape
    device = sequences.device
    mask = attention_mask.bool()
    doc_lens = mask.sum(dim=-1).tolist()
    indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()

    inner_model = getattr(model, "model", model)
    get_rope_index = getattr(inner_model, "get_rope_index", None)
    if get_rope_index is None:
        raise RuntimeError(f"VLM packing requires get_rope_index on {type(inner_model).__name__}")

    image_grid_thw = mm_inputs.get("image_grid_thw")
    video_grid_thw = mm_inputs.get("video_grid_thw")
    merge_area = _spatial_merge_area(model)
    img_cursor = 0
    vid_cursor = 0

    packed_id_chunks: list[torch.Tensor] = []
    rolled_chunks: list[torch.Tensor] = []
    doc_id_chunks: list[torch.Tensor] = []
    pos_chunks: list[torch.Tensor] = []

    import inspect

    rope_params = inspect.signature(get_rope_index).parameters
    accepts_mm_type = "mm_token_type_ids" in rope_params

    for b_idx, doc_len in enumerate(doc_lens):
        doc_len = int(doc_len)
        if doc_len <= 0:
            continue
        doc_mask = mask[b_idx]
        doc_ids = sequences[b_idx][doc_mask]  # [doc_len]
        packed_id_chunks.append(doc_ids)
        rolled_chunks.append(torch.roll(doc_ids, shifts=-1, dims=0))
        doc_id_chunks.append(torch.full((doc_len,), b_idx + 1, dtype=torch.long, device=device))

        n_img = int((doc_ids == image_token_id).sum().item()) if image_token_id is not None else 0
        n_vid = int((doc_ids == video_token_id).sum().item()) if video_token_id is not None else 0
        doc_img_grid, img_cursor = _slice_grid_rows_for_doc(image_grid_thw, img_cursor, n_img, merge_area, "image")
        doc_vid_grid, vid_cursor = _slice_grid_rows_for_doc(video_grid_thw, vid_cursor, n_vid, merge_area, "video")

        rope_kwargs: dict[str, Any] = {
            "image_grid_thw": doc_img_grid,
            "video_grid_thw": doc_vid_grid,
            "attention_mask": torch.ones((1, doc_len), dtype=torch.long, device=device),
        }
        if accepts_mm_type:
            mm_types = torch.zeros((1, doc_len), dtype=torch.long, device=device)
            if image_token_id is not None and n_img > 0:
                mm_types = mm_types.masked_fill(doc_ids.unsqueeze(0) == image_token_id, 1)
            if video_token_id is not None and n_vid > 0:
                mm_types = mm_types.masked_fill(doc_ids.unsqueeze(0) == video_token_id, 2)
            rope_kwargs["mm_token_type_ids"] = mm_types

        doc_pos, _ = get_rope_index(doc_ids.unsqueeze(0), **rope_kwargs)
        pos_chunks.append(doc_pos)

    if image_grid_thw is not None and img_cursor != image_grid_thw.shape[0]:
        raise ValueError(
            f"VLM packing: consumed {img_cursor} of {image_grid_thw.shape[0]} image_grid_thw rows across {batch} docs."
        )
    if video_grid_thw is not None and vid_cursor != video_grid_thw.shape[0]:
        raise ValueError(
            f"VLM packing: consumed {vid_cursor} of {video_grid_thw.shape[0]} video_grid_thw rows across {batch} docs."
        )

    packed_ids = torch.cat(packed_id_chunks, dim=0).unsqueeze(0)  # [1, S_packed]
    rolled_packed = torch.cat(rolled_chunks, dim=0).unsqueeze(0)  # [1, S_packed]
    packed_seq_ids = torch.cat(doc_id_chunks, dim=0).unsqueeze(0)  # [1, S_packed]
    pos_seq_dim = 2 if pos_chunks[0].ndim == 3 else 1
    packed_pos_ids = torch.cat(pos_chunks, dim=pos_seq_dim)  # [3, 1, S_packed] or [1, S_packed]
    packed_len = int(packed_ids.shape[1])

    pad = (-packed_len) % align_multiple if align_multiple > 1 else 0
    if pad:
        packed_ids = F.pad(packed_ids, (0, pad), value=pad_token_id)
        rolled_packed = F.pad(rolled_packed, (0, pad), value=0)
        packed_seq_ids = F.pad(packed_seq_ids, (0, pad), value=0)
        packed_pos_ids = _pad_mrope_positions(packed_pos_ids, pad)

    return packed_ids, rolled_packed, packed_pos_ids, packed_seq_ids, indices, packed_len
