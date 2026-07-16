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

from types import SimpleNamespace

import pytest
import torch

from molt.trainer.algorithm.experience import (
    Experience,
    balance_experiences,
    get_model_parallel_size,
    make_experience_batch,
    replay_buffer_drop_last,
    split_experience_batch,
)


def _args(
    cp=1,
    tp=1,
    ep=1,
    actor_gpus=1,
    aux_loss_coef=0.0,
    force_on_policy=False,
    dynamic_batch_enable=False,
):
    return SimpleNamespace(
        actor=SimpleNamespace(
            num_nodes=1,
            num_gpus_per_node=actor_gpus,
            aux_loss_coef=aux_loss_coef,
        ),
        fsdp=SimpleNamespace(cp_size=cp, tp_size=tp, ep_size=ep),
        train=SimpleNamespace(
            force_on_policy=force_on_policy,
            dynamic_batch_enable=dynamic_batch_enable,
        ),
    )


def test_model_parallel_size_excludes_ep():
    assert get_model_parallel_size(_args(cp=2, tp=3, ep=4)) == 6


def test_balance_experiences_uses_fsdp_data_parallel_size():
    exp = Experience(
        sequences=torch.arange(8).view(4, 2),
        attention_mask=torch.ones(4, 2, dtype=torch.long),
        total_length=torch.tensor([8, 7, 6, 5]),
    )

    balanced = balance_experiences([exp], _args(ep=2, actor_gpus=4))

    assert len(balanced) == 4
    assert [len(item.sequences) for item in balanced] == [1, 1, 1, 1]


def test_balance_experiences_equalizes_per_rank_counts():
    # 10 samples across 4 DP ranks: pad 2 zero-action rows so every rank receives
    # the SAME count without dropping real samples.
    exp = Experience(
        sequences=torch.arange(20).view(10, 2),
        attention_mask=torch.ones(10, 2, dtype=torch.long),
        action_mask=torch.ones(10, 1, dtype=torch.bool),
        total_length=torch.arange(10, 0, -1),
    )

    balanced = balance_experiences([exp], _args(actor_gpus=4))

    assert len(balanced) == 4
    counts = [len(item.sequences) for item in balanced]
    assert counts == [3, 3, 3, 3]
    assert sum(counts) == 12
    assert sum(int(item.action_mask.any(dim=-1).sum()) for item in balanced) == 10


def test_balance_experiences_pads_213_to_224_without_losing_real_segments():
    exp = Experience(
        sequences=torch.arange(426).view(213, 2),
        attention_mask=torch.ones(213, 2, dtype=torch.long),
        action_mask=torch.ones(213, 1, dtype=torch.bool),
        advantages=torch.arange(213, dtype=torch.float).view(213, 1),
        response_length=torch.ones(213),
        total_length=torch.arange(213, 0, -1),
        info={"reward": torch.arange(213, dtype=torch.float)},
    )

    balanced = balance_experiences([exp], _args(actor_gpus=32))

    assert len(balanced) == 32
    assert [len(rank.sequences) for rank in balanced] == [7] * 32
    flattened = [item for rank in balanced for item in split_experience_batch(rank)]
    real = [item for item in flattened if bool(item.action_mask.any())]
    padding = [item for item in flattened if not bool(item.action_mask.any())]
    assert len(real) == 213
    assert len(padding) == 11
    assert sorted(int(item.sequences[0]) for item in real) == list(range(0, 426, 2))
    assert all(not bool(item.action_mask.any()) for item in padding)
    assert all(float(item.advantages.abs().sum()) == 0.0 for item in padding)


@pytest.mark.parametrize("aux_loss_coef", [0.01, -0.01, float("nan"), float("inf")])
def test_balance_experiences_fails_closed_for_padding_with_moe_aux_loss(aux_loss_coef):
    exp = Experience(
        sequences=torch.arange(20).view(10, 2),
        attention_mask=torch.ones(10, 2, dtype=torch.long),
        action_mask=torch.ones(10, 1, dtype=torch.bool),
        total_length=torch.arange(10, 0, -1),
    )

    with pytest.raises(ValueError, match="aux_loss_coef=0"):
        balance_experiences([exp], _args(actor_gpus=4, aux_loss_coef=aux_loss_coef))


def test_force_on_policy_static_replay_keeps_partial_microbatch_tail():
    samples = list(range(7))
    force_static = _args(force_on_policy=True, dynamic_batch_enable=False)
    async_static = _args(force_on_policy=False, dynamic_batch_enable=False)
    force_dynamic = _args(force_on_policy=True, dynamic_batch_enable=True)

    force_batches = list(
        torch.utils.data.DataLoader(
            samples,
            batch_size=4,
            drop_last=replay_buffer_drop_last(force_static),
        )
    )
    async_batches = list(
        torch.utils.data.DataLoader(
            samples,
            batch_size=4,
            drop_last=replay_buffer_drop_last(async_static),
        )
    )

    assert [len(batch) for batch in force_batches] == [4, 3]
    assert [len(batch) for batch in async_batches] == [4]
    assert replay_buffer_drop_last(force_dynamic)


def test_make_experience_batch_intersects_sparse_info_keys():
    first = Experience(
        sequences=torch.tensor([1, 2]),
        attention_mask=torch.ones(2, dtype=torch.long),
        total_length=torch.tensor(2),
        info={"reward": torch.tensor(1.0), "task_angle": torch.tensor(1.0)},
    )
    second = Experience(
        sequences=torch.tensor([3, 4]),
        attention_mask=torch.ones(2, dtype=torch.long),
        total_length=torch.tensor(2),
        info={"reward": torch.tensor(0.0), "task_deformable": torch.tensor(0.0)},
    )

    batch = make_experience_batch([first, second])

    assert set(batch.info) == {"reward"}
    torch.testing.assert_close(batch.info["reward"], torch.tensor([1.0, 0.0]))


def test_balance_experiences_uses_global_info_intersection_across_ranks():
    # Rank-local intersections could differ (one rank gets only angle rows and
    # another only deformable rows), which would desynchronize the metric-dict
    # all_reduce.  The global intersection must be applied before partitioning.
    angle = Experience(
        sequences=torch.arange(8).view(2, 4),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        total_length=torch.tensor([8, 7]),
        info={
            "reward": torch.tensor([1.0, 0.0]),
            "always_present": torch.tensor([2.0, 2.0]),
            "task_angle": torch.tensor([0.0, 1.0]),
        },
    )
    deformable = Experience(
        sequences=torch.arange(8, 16).view(2, 4),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        total_length=torch.tensor([6, 5]),
        info={
            "reward": torch.tensor([1.0, 0.0]),
            "always_present": torch.tensor([2.0, 2.0]),
            "task_deformable": torch.tensor([2.0, 3.0]),
        },
    )

    balanced = balance_experiences([angle, deformable], _args(actor_gpus=2))

    assert len(balanced) == 2
    assert all(set(rank.info) == {"reward", "always_present"} for rank in balanced)
