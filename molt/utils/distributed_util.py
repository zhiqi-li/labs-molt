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
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.


def initialize_default_process_group(*, timeout, cpu_offload, local_rank):
    """Create and synchronize the default process group before mesh groups.

    NCCL process groups are lazy by default. If the world communicator is
    first materialized only after a framework has created several subgroups,
    ranks that progress at different speeds can bootstrap different
    communicators at the same time. Supplying ``device_id`` asks PyTorch to
    initialize the world communicator eagerly; the barrier is both a health
    check and a hard ordering point before any mesh subgroup is created.
    """

    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        backend = "cuda:nccl,cpu:gloo" if cpu_offload else "nccl"
        init_kwargs = {"backend": backend, "timeout": timeout}
        if local_rank >= 0:
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**init_kwargs)

    world_size = dist.get_world_size()
    if world_size > 1:
        if local_rank >= 0:
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
    return world_size


def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and CUDA operations.
    This function ensures that:
    1. All distributed processes reach this point (barrier)
    2. All CUDA operations are completed (synchronize)
    """
    import torch

    torch.distributed.barrier()
    torch.cuda.synchronize()


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl
