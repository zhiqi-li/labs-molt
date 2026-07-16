# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from molt.trainer.workers.actor_group import BaseModelActor, RayActorGroup
from molt.utils.seqlen_balancing import get_forward_balanced_partitions


def test_forward_partitions_balance_divisible_batches():
    costs = [100, 90, 80, 10, 9, 8]

    partitions = get_forward_balanced_partitions(costs, 3)

    assert [len(partition) for partition in partitions] == [2, 2, 2]
    assert sorted(index for partition in partitions for index in partition) == list(range(len(costs)))
    totals = [sum(costs[index] for index in partition) for partition in partitions]
    naive_totals = [sum(costs[offset : offset + 2]) for offset in range(0, len(costs), 2)]
    assert max(totals) - min(totals) < max(naive_totals) - min(naive_totals)
    assert max(totals) - min(totals) == 18


@pytest.mark.parametrize(
    "costs, actors, expected_counts",
    [([100, 90, 1, 80, 70], 3, [2, 2, 1]), ([9, 1], 4, [1, 1, 0, 0])],
)
def test_forward_partitions_put_shortest_item_at_padded_tail(costs, actors, expected_counts):
    partitions = get_forward_balanced_partitions(costs, actors)

    assert [len(partition) for partition in partitions] == expected_counts
    flattened = [index for partition in partitions for index in partition]
    assert sorted(flattened) == list(range(len(costs)))
    assert flattened[-1] == min(range(len(costs)), key=lambda index: (costs[index], index))


def test_execute_batch_runs_dummy_forwards_but_trims_results():
    actor = object.__new__(BaseModelActor)
    actor.strategy = SimpleNamespace(is_rank_0=lambda: False)
    calls = []
    actor.forward = lambda value: calls.append(value) or value * 10

    results = actor.execute_batch("forward", {"value": [1, 2, 2]}, start_idx=0, end_idx=3, valid_result_count=2)

    assert calls == [1, 2, 2]
    assert results == [10, 20]


class _RemoteMethod:
    def __init__(self, calls):
        self.calls = calls

    def remote(self, *args):
        self.calls.append(args)
        return args


class _FakeActor:
    def __init__(self):
        self.calls = []
        self.execute_batch = _RemoteMethod(self.calls)


def test_ray_actor_group_pads_read_only_forward_and_marks_valid_results(monkeypatch):
    group = object.__new__(RayActorGroup)
    actors = [_FakeActor(), _FakeActor()]
    group._actor_handlers = actors
    group.duplicate_actors = 1
    monkeypatch.setattr("molt.trainer.workers.actor_group.ray.put", lambda value: value)

    group.async_run_method_batch("forward", pad_to_divisible=True, value=[1, 2, 3])

    first = actors[0].calls[0]
    second = actors[1].calls[0]
    assert first[1] == {"value": [1, 2]}
    assert first[-1] == 2
    assert second[1] == {"value": [3, 3]}
    assert second[-1] == 1


def test_ray_actor_group_rejects_padding_for_mutating_methods():
    group = object.__new__(RayActorGroup)

    with pytest.raises(ValueError, match="restricted to the read-only forward"):
        group.async_run_method_batch("append", pad_to_divisible=True, value=[1])
