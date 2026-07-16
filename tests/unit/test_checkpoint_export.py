# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from molt.trainer.fsdp.checkpoint import CheckpointManager


@pytest.mark.parametrize("synchronize_after_promotion, expected_barriers", [(True, 2), (False, 1)])
def test_save_model_can_skip_post_promotion_barrier(monkeypatch, synchronize_after_promotion, expected_barriers):
    strategy = SimpleNamespace(_unwrap_model=lambda model: model)
    manager = CheckpointManager(strategy)
    checkpointer = SimpleNamespace(save_model=lambda **kwargs: None)
    monkeypatch.setattr(manager, "_build_checkpointer", lambda *args, **kwargs: checkpointer)
    monkeypatch.setattr(manager, "_promote_hf_export", lambda output_dir: None)
    monkeypatch.setattr("molt.trainer.fsdp.checkpoint.dist.is_initialized", lambda: True)
    barriers = []
    monkeypatch.setattr("molt.trainer.fsdp.checkpoint.dist.barrier", lambda: barriers.append(True))

    manager.save_model(
        object(),
        tokenizer=None,
        output_dir="unused",
        synchronize_after_promotion=synchronize_after_promotion,
    )

    assert len(barriers) == expected_barriers
