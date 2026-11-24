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

from typing import List, Optional

import torch
from torch import distributed as dist

from molt.trainer.algorithm.experience import (
    Experience,
    get_model_parallel_size,
    make_experience_batch,
    remove_padding_in_sequences,
    split_experience_batch,
)
from molt.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions


def _experience_media_keys(item: Experience) -> tuple[str, ...]:
    """Return sorted multimodal input keys for DP collective alignment (works with or without #23's media_keys)."""
    keys = getattr(item, "media_keys", None)
    if keys is not None:
        return tuple(keys)
    return tuple(
        sorted(
            {
                key
                for entry in (item.mm_train_inputs or [])
                for inputs in (entry if isinstance(entry, list) else [entry])
                for key in (inputs or ())
            }
        )
    )


def _bounded_ffd_bins(
    indices: List[int],
    sample_lengths: List[int],
    max_tokens: int,
) -> List[List[int]]:
    """First-Fit Decreasing bin packing with a strict token cap on multi-sample bins.

    Any single sequence longer than ``max_tokens`` stays in a solo bin of size 1 so
    packing never raises peak sequence length above the single-sample maximum.
    """
    ordered = sorted(indices, key=lambda idx: sample_lengths[idx], reverse=True)
    bins: List[List[int]] = []
    sums: List[int] = []
    for idx in ordered:
        length = sample_lengths[idx]
        placed = False
        for b_idx in range(len(bins)):
            if sums[b_idx] + length <= max_tokens:
                bins[b_idx].append(idx)
                sums[b_idx] += length
                placed = True
                break
        if not placed:
            bins.append([idx])
            sums.append(length)
    return bins


def _expand_bins_to_count(
    bins: List[List[int]],
    sample_lengths: List[int],
    target_count: int,
) -> List[List[int]]:
    """Split multi-sample bins until ``len(bins) == target_count`` while preserving the token cap."""
    bins = [list(b) for b in bins]
    sums = [sum(sample_lengths[idx] for idx in b) for b in bins]
    while len(bins) < target_count:
        splittable = [i for i, b in enumerate(bins) if len(b) > 1]
        if not splittable:
            raise RuntimeError(
                f"Cannot expand {len(bins)} packed microbatches to {target_count} across DP ranks "
                f"(only {sum(len(b) for b in bins)} samples in media group)."
            )
        pick = max(splittable, key=lambda i: (sums[i], len(bins[i])))
        moved = bins[pick].pop()
        moved_len = sample_lengths[moved]
        sums[pick] -= moved_len
        bins.append([moved])
        sums.append(moved_len)
    order = sorted(range(len(bins)), key=lambda i: sums[i], reverse=True)
    return [bins[i] for i in order]


class NaiveReplayBuffer:
    """Stores rollout experiences and yields training microbatches via the Dataset protocol.

    Consumed by a DataLoader (``__len__`` / ``__getitem__`` / ``collate_fn``), not by random sampling.

    Args:
        sample_batch_size (int): DataLoader batch size (microbatch size); forced to 1 in dynamic-batch mode.
        limit (int, optional): Cap on stored samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Offload stored experience to CPU until fetched for a microbatch. Defaults to True.
    """

    def __init__(
        self,
        sample_batch_size: int,
        limit: int = 0,
        cpu_offload: bool = True,
        dynamic_batch: bool = False,
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.items: List[Experience] = []
        self.dynamic_batch = dynamic_batch
        self.dynamic_indices: List[List[int]] = []
        self.dynamic_optimizer_step: List[int] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    def has_mixed_media(self) -> bool:
        """True when buffered samples do not all share the same media_keys."""
        return len({item.media_keys for item in self.items}) > 1

    def __len__(self) -> int:
        if self.dynamic_batch:
            return len(self.dynamic_indices)
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        if self.dynamic_batch:
            indices = self.dynamic_indices[idx]
            return [self.items[i] for i in indices]
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        if self.dynamic_batch:
            batch = batch[0]
        return make_experience_batch(batch)

    def setup_dynamic_batch(self, strategy):
        args = strategy.args
        sample_lengths = [int(sample.total_length.item()) for sample in self.items]

        world_size = dist.get_world_size()
        dp_size = world_size // get_model_parallel_size(args)
        if args.train.force_on_policy:
            # On-policy: the whole buffer is ONE optimizer step. A fixed
            # train.batch_size would split the rollout into several steps (every
            # step after the first off-policy) and drop the trailing samples that
            # don't fill a batch. balance_experiences already equalized the
            # per-rank sample count, so every rank takes exactly one step in
            # lockstep; the token-budget split below still bounds microbatch size.
            local_train_batch_size = len(sample_lengths)
            num_steps = 1 if sample_lengths else 0
        else:
            local_train_batch_size = args.train.batch_size // dp_size
            # Async generation can deliver a short buffer at episode boundaries, and
            # per-media DP alignment can trim a single-step rollout below one full batch.
            # Run one shorter step rather than silently dropping the entire batch to 0 steps.
            if 0 < len(sample_lengths) < local_train_batch_size:
                strategy.print(
                    f"[ReplayBuffer] shortening train batch per rank from {local_train_batch_size} "
                    f"to {len(sample_lengths)} for trimmed buffer."
                )
                local_train_batch_size = len(sample_lengths)
                num_steps = 1
            else:
                num_steps = len(sample_lengths) // local_train_batch_size
            # balance_experiences only equalizes the per-rank count to a multiple
            # of dp_size, not of local_train_batch_size, so a remainder here is
            # dropped from this update. Surface it instead of dropping silently.
            dropped = len(sample_lengths) - num_steps * local_train_batch_size
            if dropped:
                strategy.print(
                    f"[ReplayBuffer] dropping {dropped} trailing sample(s) per rank that don't fill a "
                    f"{local_train_batch_size}-sample train batch (buffer={len(sample_lengths)})."
                )

        # Group each optimizer step's samples by media_keys (text first, media last) so
        # every DP rank executes the same media type on each microbatch and deferred
        # gradient sync still includes the vision tower on the last microbatch.
        # Non-empty media groups use bounded FFD packing (never pairing sequences beyond
        # max_tokens while keeping oversized solo sequences untouched); text-only groups
        # keep Karmarkar-Karp balanced partitioning.
        max_tokens = int(args.train.max_tokens_per_gpu) * int(args.fsdp.cp_size) * int(args.fsdp.tp_size)
        step_groups: List[List[tuple[tuple[str, ...], List[int]]]] = []
        local_bins_per_group: List[Optional[List[List[int]]]] = []
        num_microbatches = []
        for i in range(num_steps):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            by_media: dict[tuple[str, ...], List[int]] = {}
            for idx in range(start, end):
                by_media.setdefault(_experience_media_keys(self.items[idx]), []).append(idx)
            groups = [(media, by_media[media]) for media in sorted(by_media)]
            step_groups.append(groups)
            for media, group in groups:
                if media:
                    bins = _bounded_ffd_bins(group, sample_lengths, max_tokens)
                    local_bins_per_group.append(bins)
                    num_microbatches.append(len(bins))
                else:
                    local_bins_per_group.append(None)
                    num_microbatches.append(
                        get_minimum_num_micro_batch_size(
                            [sample_lengths[idx] for idx in group],
                            args.train.max_tokens_per_gpu,
                            args.fsdp.cp_size,
                            args.fsdp.tp_size,
                        )
                    )

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        num_microbatches = strategy.all_reduce(num_microbatches, op="max")
        num_microbatches = num_microbatches.tolist()

        # Partition each (step, media) group independently into the synced microbatch count.
        micro_batch_indices = []
        data_partitions = []
        mb_iter = iter(zip(num_microbatches, local_bins_per_group))
        for groups in step_groups:
            step_partitions = []
            for _media, group in groups:
                num_mbs, local_bins = next(mb_iter)
                if local_bins is not None:
                    step_partitions.extend(_expand_bins_to_count(local_bins, sample_lengths, num_mbs))
                else:
                    partitions = get_seqlen_balanced_partitions(
                        [sample_lengths[idx] for idx in group], num_mbs, equal_size=False
                    )
                    step_partitions.extend([[group[k] for k in partition] for partition in partitions])
            micro_batch_indices.extend(step_partitions)
            data_partitions.append(step_partitions)
        self.dynamic_indices = micro_batch_indices
        self.sample_batch_size = 1

        # Mark each step's last microbatch as the optimizer-step boundary. The
        # global token-mean (one denominator per window) handles per-microbatch
        # token-count weighting, so no per-microbatch loss scale is needed.
        optimizer_steps = []
        for partitions in data_partitions:
            optimizer_steps.extend([0] * (len(partitions) - 1) + [1])
        self.dynamic_optimizer_step = optimizer_steps
