# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from molt.utils.distributed_util import initialize_default_process_group


@pytest.mark.parametrize(
    "cpu_offload, local_rank, expected_backend, expected_device, expected_barrier",
    [
        (False, 3, "nccl", torch.device("cuda", 3), {"device_ids": [3]}),
        (True, 1, "cuda:nccl,cpu:gloo", torch.device("cuda", 1), {"device_ids": [1]}),
        (False, -1, "nccl", None, {}),
    ],
)
def test_initialize_default_process_group_eagerly_materializes_world_communicator(
    monkeypatch,
    cpu_offload,
    local_rank,
    expected_backend,
    expected_device,
    expected_barrier,
):
    init_calls = []
    barrier_calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "barrier", lambda **kwargs: barrier_calls.append(kwargs))
    timeout = timedelta(minutes=5)

    world_size = initialize_default_process_group(timeout=timeout, cpu_offload=cpu_offload, local_rank=local_rank)

    assert world_size == 4
    assert init_calls == [
        {
            "backend": expected_backend,
            "timeout": timeout,
            **({"device_id": expected_device} if expected_device is not None else {}),
        }
    ]
    assert barrier_calls == [expected_barrier]


def test_initialize_default_process_group_reuses_initialized_single_rank(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda **kwargs: pytest.fail("must not initialize an existing process group"),
    )
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "barrier", lambda **kwargs: pytest.fail("single rank needs no barrier"))

    assert initialize_default_process_group(timeout=timedelta(minutes=5), cpu_offload=False, local_rank=0) == 1
