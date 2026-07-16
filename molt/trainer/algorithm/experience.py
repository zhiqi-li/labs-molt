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

import itertools
import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, List, Union

import torch

from molt.utils.logging_utils import init_logger
from molt.utils.seqlen_balancing import get_seqlen_balanced_partitions
from molt.utils.utils import zero_pad_sequences

logger = init_logger(__name__)


def tensor_field(role: str, **kwargs):
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["tensor_role"] = role
    return field(metadata=metadata, **kwargs)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def get_model_parallel_size(args) -> int:
    """Members of one DP group — the ranks that share a data shard, i.e. ``cp * tp``.

    EP shards experts on a separate MoE mesh but each EP rank still owns its full data
    shard, so EP must NOT enter the divisor that splits a batch across DP groups.
    """
    fsdp = args.fsdp
    return int(fsdp.cp_size) * int(fsdp.tp_size)


@dataclass
class Experience:
    """A batch of RL experience for policy optimization.

    Fields are grouped by RL semantics:
    - Trajectory: token-level state-action sequences and masks (B, T)
    - Policy: next-token step tensors under different policies (B, T-1)
    - Optimization: per-step returns and advantages (B, T-1)
    - Outcome: per-episode rewards and generation metadata (B,)
    - Metadata: non-tensor fields for logging and data tracking

    Policy/target tensors keep the dense next-token axis instead of compressing
    to action-only positions. In multi-turn rollouts, observation/tool feedback
    remains present on that axis and is excluded by action_mask=False.
    """

    # Trajectory: state-action sequences
    sequences: torch.Tensor = tensor_field("step", default=None)  # (B, T) token ids [prompt + response]
    attention_mask: torch.LongTensor = tensor_field("step", default=None)  # (B, T)
    action_mask: torch.BoolTensor = tensor_field("step", default=None)  # (B, T-1) generated-token steps

    # Policy: log probs under current, reference, and rollout policies
    action_log_probs: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) log pi_theta(a|s)
    base_action_log_probs: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) log pi_ref(a|s)
    rollout_log_probs: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) log pi_old(a|s)
    # R3 rollout routing replay: the rollout router's top-k expert ids per token, one row
    # per MoE layer. Stored seq-LAST as (B, num_moe_layers, topk, T) so it rides the same
    # right-pad/concat/stack machinery as the (B, T) step tensors; the actor forward
    # permutes it back to token-major and replays it. None when R3 off.
    routed_experts: torch.Tensor = tensor_field("step", default=None)

    # Policy-gradient targets
    returns: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) G_t (PPO: value-regression target)
    advantages: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) A(s,a)
    values: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) critic V(s) at collection (PPO old_values)
    kl: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) D_KL(pi_theta || pi_ref)

    # Episode outcomes (per-sample scalars)
    rewards: torch.Tensor = tensor_field("episode", default=None)  # (B,) R, used for advantage calculation
    scores: torch.Tensor = tensor_field("episode", default=None)  # (B,) binary score for dynamic sampling
    response_length: torch.Tensor = tensor_field("episode", default=None)  # (B,) number of generated action tokens
    truncated: torch.Tensor = tensor_field("episode", default=None)  # (B,) whether generation was truncated
    total_length: torch.Tensor = tensor_field("episode", default=None)  # (B,) prompt + response length

    # Per-sample row id within the rollout batch (set to [i] per sample). After
    # concat_experiences, len(index) = number of samples in this Experience —
    # the advantage/merge logic relies on this count, so it is NOT pure metadata.
    index: list[int] = None

    # Metadata (not part of RL computation)
    prompts: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    images: list = field(default_factory=list)  # per-sample image paths/URLs for VLM (None entries for text-only)
    mm_train_inputs: list = field(default_factory=list)  # per-sample processor outputs (pixel_values dicts) for VLM
    info: dict = field(default_factory=dict)  # per-sample metrics for logging
    # GRPO grouping identity. `group_ids` (= prompt id) is shared by all N rollouts of one
    # prompt — the trainer averages their rewards to form the baseline. `rollout_ids` is
    # unique per trajectory; multi-turn agents emit several step-samples sharing one
    # rollout_id so the trainer dedups to one reward per rollout before grouping by group_id.
    group_ids: list[str] = field(default_factory=list)
    rollout_ids: list[str] = field(default_factory=list)

    @classmethod
    def is_step_tensor_field(cls, name: str) -> bool:
        field_info = cls.__dataclass_fields__.get(name)
        return field_info is not None and field_info.metadata.get("tensor_role") == "step"

    @classmethod
    def is_episode_tensor_field(cls, name: str) -> bool:
        field_info = cls.__dataclass_fields__.get(name)
        return field_info is not None and field_info.metadata.get("tensor_role") == "episode"

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for name, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, name, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, name, to(value, device))

        return self

    @staticmethod
    def _merge_item(items: List, pad_value: int = 0) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return list(itertools.chain.from_iterable(items))
        elif isinstance(items[0], dict):
            result = {}
            # Collect all values for each key
            for d in items:
                for key, value in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            # Merge all values for each key at once
            return {key: Experience._merge_item(values, pad_value) for key, values in result.items()}
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for name in field_names:
            values = [getattr(e, name) for e in experiences_list]
            # sequences pad with pad_token_id; routed_experts with the R3 -1 sentinel
            # ("keep live routing" — 0 is a valid expert id); everything else with 0.
            pad_value = pad_token_id if name == "sequences" else (-1 if name == "routed_experts" else 0)
            result[name] = Experience._merge_item(values, pad_value)

        return Experience(**result)


# Batch manipulation utilities


def split_experience_batch(experience: Experience) -> List[Experience]:
    """Split a batched Experience into individual single-sample Experiences."""
    batch_size = len(experience.sequences)
    experience.index = None

    items = []
    for i in range(batch_size):
        kwargs = {}
        for f in fields(Experience):
            value = getattr(experience, f.name)
            if value is None:
                kwargs[f.name] = None
            elif isinstance(value, torch.Tensor):
                if len(value) != batch_size:
                    raise ValueError(f"Size of {f.name} ({len(value)}) does not match batch_size ({batch_size})")
                kwargs[f.name] = value[i]
            elif isinstance(value, dict):
                d = {}
                for k, v in value.items():
                    if isinstance(v, (torch.Tensor, list)):
                        if len(v) != batch_size:
                            raise ValueError(
                                f"Size of {f.name}[{k}] ({len(v)}) does not match batch_size ({batch_size})"
                            )
                        d[k] = v[i]
                    else:
                        raise TypeError(f"Unsupported type for {f.name}[{k}]: {type(v)}")
                kwargs[f.name] = d
            elif isinstance(value, list):
                kwargs[f.name] = [value[i]] if len(value) == batch_size else value
        items.append(Experience(**kwargs))

    return items


def make_experience_batch(items: List[Experience]) -> Experience:
    """Combine individual single-sample Experiences into a batched Experience."""
    if not items:
        raise ValueError("Empty items list")

    kwargs = {}
    for f in fields(Experience):
        first = getattr(items[0], f.name)
        if first is None:
            kwargs[f.name] = None
        elif isinstance(first, torch.Tensor):
            tensors = [getattr(item, f.name) for item in items]
            if Experience.is_step_tensor_field(f.name):
                # routed_experts pads with the R3 -1 sentinel (keep live routing); 0 is a
                # valid expert id and would force pad tokens to expert 0. Others pad with 0.
                pad_value = -1 if f.name == "routed_experts" else 0
                kwargs[f.name] = zero_pad_sequences(tensors, "right", stack=True, value=pad_value)
            elif Experience.is_episode_tensor_field(f.name) or first.dim() == 0:
                kwargs[f.name] = torch.stack(tensors)
            else:
                raise ValueError(f"Unsupported tensor field batching rule for {f.name}")
        elif isinstance(first, dict):
            kwargs[f.name] = {}
            mappings = [getattr(item, f.name) for item in items]
            if not all(isinstance(mapping, dict) for mapping in mappings):
                raise TypeError(f"Inconsistent types in {f.name}")
            # Per-environment logging dictionaries can contain sparse metrics
            # (for example only the current ESI-Bench task's task-specific
            # counters).  A batch has one value per sample, so only keys present
            # for every item can be represented without turning "not applicable"
            # into a misleading numeric zero.
            common_keys = set.intersection(*(set(mapping) for mapping in mappings))
            for key in sorted(common_keys):
                vals = [getattr(item, f.name)[key] for item in items]
                if not vals:
                    continue
                first_type = type(vals[0])
                if not all(isinstance(v, first_type) for v in vals):
                    raise TypeError(f"Inconsistent types in {f.name}[{key}]")
                if all(isinstance(v, torch.Tensor) for v in vals):
                    kwargs[f.name][key] = torch.stack(vals)
                elif all(isinstance(v, (int, float)) for v in vals):
                    kwargs[f.name][key] = torch.tensor(vals)
                else:
                    kwargs[f.name][key] = vals
        elif isinstance(first, list):
            kwargs[f.name] = list(itertools.chain.from_iterable(getattr(item, f.name) for item in items))

    return Experience(**kwargs)


def remove_padding_in_sequences(items: List[Experience]) -> List[Experience]:
    """Remove right padding from per-step fields of single-sample Experiences."""
    for item in items:
        right_pad = item.attention_mask.flip(0).argmax()
        right_pad = None if right_pad == 0 else -right_pad

        for f in fields(Experience):
            value = getattr(item, f.name)
            if isinstance(value, torch.Tensor) and Experience.is_step_tensor_field(f.name):
                # Slice the LAST (sequence) dim: 1D step tensors are [T], but
                # routed_experts is [num_moe_layers, topk, T] (seq last).
                setattr(item, f.name, value[..., :right_pad])

    return items


def _make_zero_action_dummy(template: Experience) -> Experience:
    """Shallow-clone one valid sample into a collective-safe padding sample.

    The token/context and multimodal payload stay valid for the model forward,
    while every optimization/outcome tensor is zeroed.  In particular the
    all-false action mask makes the dummy contribute neither policy/value
    gradients nor action tokens.  ``replace`` deliberately keeps large
    multimodal objects shared rather than copying image tensors for DP padding.
    """
    if not isinstance(template.action_mask, torch.Tensor):
        raise ValueError("DP padding requires an action_mask so dummy gradients can be masked safely")

    preserve = {"sequences", "attention_mask", "routed_experts", "total_length"}
    updates = {}
    for f in fields(Experience):
        value = getattr(template, f.name)
        if isinstance(value, torch.Tensor) and f.name not in preserve:
            updates[f.name] = torch.zeros_like(value)

    # Keep the same keys/types so every rank reduces an identical metric dict.
    # Sample-weighted logging removes zero-action rows before taking means.
    updates["info"] = {
        key: torch.zeros_like(value) if isinstance(value, torch.Tensor) else value
        for key, value in template.info.items()
    }
    dummy = replace(template, **updates)
    if bool(dummy.action_mask.any()):  # defensive: the contract above is safety-critical
        raise AssertionError("DP padding dummy unexpectedly contains trainable action tokens")
    return dummy


def replay_buffer_drop_last(args) -> bool:
    """Whether replay DataLoaders may discard a final partial microbatch.

    Strict on-policy static training accumulates the entire balanced shard into
    one optimizer step, so its final microbatch is part of that policy batch and
    must never be dropped. Async and dynamic modes retain their old behavior.
    """
    train = args.train
    return not (bool(train.force_on_policy) and not bool(train.dynamic_batch_enable))


def balance_experiences(experiences, args):
    """Balance samples across DP ranks by total sequence length, equal-count.

    Every DP rank must receive the SAME number of samples: unequal counts yield different
    ``num_steps`` per rank → mismatched collective shapes at the world all_reduces → NCCL
    hang. Use equal-size length balancing and pad to the next DP multiple with
    zero-action dummy Experiences, preserving every real flattened segment.
    """
    items_all = []
    for item in experiences:
        items_all.extend(split_experience_batch(item))

    actor_world_size = args.actor.num_nodes * args.actor.num_gpus_per_node
    effective_num = actor_world_size // get_model_parallel_size(args)
    if effective_num <= 0:
        raise ValueError(f"Invalid effective actor count: {effective_num}")
    if not items_all:
        raise ValueError("Cannot balance an empty experience list")

    # Equal counts per rank ⇒ identical num_steps on every rank. Pad rather than
    # dropping real samples; at most effective_num - 1 dummies are needed.
    remainder = len(items_all) % effective_num
    if remainder:
        padding = effective_num - remainder
        aux_loss_coef = float(getattr(args.actor, "aux_loss_coef", 0.0) or 0.0)
        if not math.isfinite(aux_loss_coef) or abs(aux_loss_coef) > 1e-8:
            raise ValueError(
                "DP balance padding with zero-action dummies requires a finite actor.aux_loss_coef=0: "
                "MoE auxiliary loss is not action-masked, so padded samples could change gradients."
            )
        shortest = min(
            items_all,
            key=lambda item: int(
                item.total_length.item() if isinstance(item.total_length, torch.Tensor) else item.total_length
            ),
        )
        logger.warning(
            f"[balance_experiences] padding {padding} zero-action dummy sample(s) so {len(items_all)} "
            f"real samples divide evenly across {effective_num} DP ranks."
        )
        items_all.extend(_make_zero_action_dummy(shortest) for _ in range(padding))

    # Every DP rank enters the same all_reduce over its metric dictionary in
    # PolicyModelActor._record_status, so all partitions must carry the exact
    # same keys.  Keep the global intersection before partitioning.  Sparse
    # task-only metrics remain available in the trajectory log/eval report, but
    # are intentionally omitted from this mixed-task optimization batch rather
    # than being imputed as zero (which would corrupt their means).
    for f in fields(Experience):
        mappings = [getattr(item, f.name) for item in items_all]
        if not mappings or not all(isinstance(mapping, dict) for mapping in mappings):
            continue
        common_keys = set.intersection(*(set(mapping) for mapping in mappings))
        all_keys = set.union(*(set(mapping) for mapping in mappings))
        sparse_keys = sorted(all_keys - common_keys)
        if sparse_keys:
            logger.warning(
                "[balance_experiences] dropping sparse per-sample keys from %s: %s",
                f.name,
                sparse_keys,
            )
            for item, mapping in zip(items_all, mappings):
                setattr(item, f.name, {key: mapping[key] for key in common_keys})

    lengths = [
        int(item.total_length.item() if isinstance(item.total_length, torch.Tensor) else item.total_length)
        for item in items_all
    ]
    # equal_size=True keeps each rank's sample count identical while still
    # minimizing the per-rank total-token spread (Karmarkar–Karp).
    partitions = get_seqlen_balanced_partitions(lengths, effective_num, equal_size=True)
    # Sort each rank's items by length (descending) so the k-th microbatch is similarly
    # sized on every rank. The k-th microbatch runs in lockstep at cross-node
    # reduce-scatters (expert-grad over ep_shard, plus per-microbatch FSDP), so a size
    # mismatch makes short ranks wait on a straggler long enough to trip the 600s NCCL
    # watchdog → SIGABRT. KK balances each rank's total tokens/count but not within-rank
    # order, so pairing was random. The dataloader preserves this order (no shuffle when
    # model-parallel size > 1).
    partitions = [sorted(partition, key=lambda idx: lengths[idx], reverse=True) for partition in partitions]
    return [make_experience_batch([items_all[idx] for idx in partition]) for partition in partitions]
