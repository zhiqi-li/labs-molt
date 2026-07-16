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

import torch

from molt.trainer.algorithm.experience import (
    Experience,
    balance_experiences,
    get_model_parallel_size,
    make_experience_batch,
)


def _args(cp=1, tp=1, ep=1, actor_gpus=1):
    return SimpleNamespace(
        actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=actor_gpus),
        fsdp=SimpleNamespace(cp_size=cp, tp_size=tp, ep_size=ep),
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
    # 10 samples across 4 DP ranks: the 2-sample remainder is dropped so every
    # rank receives the SAME count. Unequal counts would desync num_steps across
    # ranks and deadlock the world all_reduce in setup_dynamic_batch.
    exp = Experience(
        sequences=torch.arange(20).view(10, 2),
        attention_mask=torch.ones(10, 2, dtype=torch.long),
        total_length=torch.arange(10, 0, -1),
    )

    balanced = balance_experiences([exp], _args(actor_gpus=4))

    assert len(balanced) == 4
    counts = [len(item.sequences) for item in balanced]
    assert counts == [2, 2, 2, 2]  # equal counts, trailing remainder dropped
    assert sum(counts) == 8


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
