# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Molt trainer speed optimizations (chunked LM head, bounded FFD packing, FLA NB, HF export)."""

from __future__ import annotations

import torch
import torch.nn as nn

from molt.models.chunked_head import ChunkedHeadLogProbs, chunked_head_log_probs
from molt.models.utils import log_probs_from_logits
from molt.trainer.algorithm.replay_buffer import _bounded_ffd_bins, _expand_bins_to_count
from molt.trainer.fsdp.fla_opt import _coarse_nb
from molt.trainer.fsdp.hf_export import _prune_hf_dirs


def test_chunked_head_matches_dense_linear_forward_and_backward_with_score_centering():
    torch.manual_seed(0)
    B, S, H, V, K = 2, 19, 32, 128, 5
    hidden_ref = torch.randn(B, S, H, dtype=torch.bfloat16, requires_grad=True)
    hidden_opt = hidden_ref.detach().clone().requires_grad_(True)

    head_ref = nn.Linear(H, V, bias=False, dtype=torch.bfloat16)
    head_opt = nn.Linear(H, V, bias=False, dtype=torch.bfloat16)
    head_opt.weight.data.copy_(head_ref.weight.data)

    # Construct targets where col 0 frequently repeats one of the distinct top-k cols 1..K.
    top_ids = torch.stack([torch.randperm(V)[:K] for _ in range(B * S)]).view(B, S, K)
    sampled = top_ids[:, :, 0].clone()
    sampled[0, 0] = (top_ids[0, 0, 0] + 17) % V  # some outside top-k
    targets = torch.cat([sampled.unsqueeze(-1), top_ids], dim=-1)

    active_mask = torch.zeros(B, S, dtype=torch.bool)
    active_mask[0, 4:15] = True
    active_mask[1, 10:18] = True

    # Reference path: dense nn.Linear -> log_probs_from_logits
    logits_ref = head_ref(hidden_ref)
    lp_ref = log_probs_from_logits(logits_ref, targets, temperature=1.0)

    # Optimized path: chunked active-only with tiny chunk_rows=3 to exercise chunk loops
    flat_h = hidden_opt.reshape(-1, H)
    flat_ids = targets.reshape(-1, K + 1)
    active_idx = active_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    lp_opt = ChunkedHeadLogProbs.apply(flat_h, head_opt.weight, flat_ids, active_idx, 1.0, 3, True).view(B, S, K + 1)

    torch.testing.assert_close(lp_opt[active_mask], lp_ref[active_mask], atol=1e-5, rtol=1e-5)

    # Backward through identical masked loss
    weights = torch.randn(B, S, K + 1) * active_mask.unsqueeze(-1).float()
    loss_ref = (lp_ref * weights).sum()
    loss_opt = (lp_opt * weights).sum()
    loss_ref.backward()
    loss_opt.backward()

    torch.testing.assert_close(hidden_opt.grad, hidden_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(head_opt.weight.grad, head_ref.weight.grad, atol=2e-2, rtol=2e-2)


def test_chunked_head_zero_active_tokens_still_produces_zero_weight_grad():
    hidden = torch.randn(1, 8, 16, dtype=torch.bfloat16, requires_grad=True)
    head = nn.Linear(16, 64, bias=False, dtype=torch.bfloat16)
    targets = torch.randint(0, 64, (1, 8))
    active_mask = torch.zeros(1, 8, dtype=torch.bool)

    lp = chunked_head_log_probs(hidden, head, targets, active_mask=active_mask)
    assert (lp == 0).all()
    (lp.sum()).backward()
    assert hidden.grad is not None and (hidden.grad == 0).all()
    assert head.weight.grad is not None and (head.weight.grad == 0).all()


def test_bounded_ffd_packing_and_dp_expansion():
    # Mix of oversized solo sequences (>128K) and packable short sequences (30K-60K).
    lengths = [249000, 170000, 150000, 60000, 55000, 48000, 42000, 35000, 32000]
    indices = list(range(len(lengths)))
    cap = 128000
    bins = _bounded_ffd_bins(indices, lengths, cap)

    # Every oversized sequence stays solo; every multi-item bin obeys cap.
    flat = [i for b in bins for i in b]
    assert sorted(flat) == indices
    for b in bins:
        if len(b) > 1:
            assert sum(lengths[i] for i in b) <= cap
        else:
            assert lengths[b[0]] > 0

    # Expanding bins to match another DP rank's larger bin count preserves cap and descending order.
    expanded = _expand_bins_to_count(bins, lengths, len(bins) + 2)
    assert len(expanded) == len(bins) + 2
    assert sorted(i for b in expanded for i in b) == indices
    sums = [sum(lengths[i] for i in b) for b in expanded]
    assert sums == sorted(sums, reverse=True)
    for b in expanded:
        if len(b) > 1:
            assert sum(lengths[i] for i in b) <= cap


def test_coarse_nb_single_bucket_across_262k():
    assert _coarse_nb(1000, 1048576) == 1
    assert _coarse_nb(65536, 1048576) == 1
    assert _coarse_nb(262144, 1048576) == 1


def test_prune_hf_dirs_ignores_hidden_staging(tmp_path):
    for name in (".staging_global_step30", "global_step28", "global_step29", "global_step30"):
        (tmp_path / name).mkdir()
    _prune_hf_dirs(str(tmp_path), "global_step30", max_num=2, max_mem=float("inf"))
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert ".staging_global_step30" in remaining
    assert "global_step30" in remaining
    assert len([n for n in remaining if not n.startswith(".")]) == 2
