# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from molt.models.utils import log_probs_from_logits


def test_bfloat16_log_probs_use_bounded_chunks(monkeypatch):
    torch.manual_seed(0)
    logits = torch.randn(130, 17, dtype=torch.bfloat16)
    labels = torch.randint(0, logits.shape[-1], (logits.shape[0],))
    expected = torch.log_softmax(logits.float(), dim=-1).gather(1, labels[:, None]).squeeze(1)

    calls = []
    original_logsumexp = torch.logsumexp

    def recording_logsumexp(input, *args, **kwargs):
        calls.append(input.shape[0])
        return original_logsumexp(input, *args, **kwargs)

    monkeypatch.setattr(torch, "logsumexp", recording_logsumexp)

    actual = log_probs_from_logits(logits, labels)

    torch.testing.assert_close(actual, expected)
    assert calls == [64, 64, 2]
