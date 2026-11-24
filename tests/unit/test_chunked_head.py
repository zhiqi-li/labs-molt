# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for chunked active-only LM head log-prob computation."""

from __future__ import annotations

import torch
import torch.nn as nn

from molt.models.chunked_head import ChunkedHeadLogProbs, chunked_head_log_probs
from molt.models.utils import log_probs_from_logits


def test_chunked_head_matches_dense_linear_forward_and_backward_with_score_centering():
    torch.manual_seed(0)
    B, S, H, V, K = 2, 19, 32, 128, 5
    hidden_ref = torch.randn(B, S, H, dtype=torch.bfloat16, requires_grad=True)
    hidden_opt = hidden_ref.detach().clone().requires_grad_(True)

    head_ref = nn.Linear(H, V, bias=False, dtype=torch.bfloat16)
    head_opt = nn.Linear(H, V, bias=False, dtype=torch.bfloat16)
    head_opt.weight.data.copy_(head_ref.weight.data)

    top_ids = torch.stack([torch.randperm(V)[:K] for _ in range(B * S)]).view(B, S, K)
    sampled = top_ids[:, :, 0].clone()
    sampled[0, 0] = (top_ids[0, 0, 0] + 17) % V
    targets = torch.cat([sampled.unsqueeze(-1), top_ids], dim=-1)

    active_mask = torch.zeros(B, S, dtype=torch.bool)
    active_mask[0, 4:15] = True
    active_mask[1, 10:18] = True

    logits_ref = head_ref(hidden_ref)
    lp_ref = log_probs_from_logits(logits_ref, targets, temperature=1.0)

    flat_h = hidden_opt.reshape(-1, H)
    flat_ids = targets.reshape(-1, K + 1)
    active_idx = active_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    lp_opt = ChunkedHeadLogProbs.apply(flat_h, head_opt.weight, flat_ids, active_idx, 1.0, 3, True).view(B, S, K + 1)

    torch.testing.assert_close(lp_opt[active_mask], lp_ref[active_mask], atol=1e-5, rtol=1e-5)

    weights = torch.randn(B, S, K + 1) * active_mask.unsqueeze(-1).float()
    (lp_ref * weights).sum().backward()
    (lp_opt * weights).sum().backward()

    torch.testing.assert_close(hidden_opt.grad, hidden_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(head_opt.weight.grad, head_ref.weight.grad, atol=2e-2, rtol=2e-2)


def test_chunked_head_zero_active_tokens_still_produces_zero_weight_grad():
    hidden = torch.randn(1, 8, 16, dtype=torch.bfloat16, requires_grad=True)
    head = nn.Linear(16, 64, bias=False, dtype=torch.bfloat16)
    targets = torch.randint(0, 64, (1, 8))
    active_mask = torch.zeros(1, 8, dtype=torch.bool)

    lp = chunked_head_log_probs(hidden, head, targets, active_mask=active_mask)
    assert (lp == 0).all()
    lp.sum().backward()
    assert hidden.grad is not None and (hidden.grad == 0).all()
    assert head.weight.grad is not None and (head.weight.grad == 0).all()
