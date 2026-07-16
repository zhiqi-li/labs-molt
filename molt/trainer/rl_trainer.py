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

import asyncio
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

import ray
import torch
from ray.util.queue import Queue
from tqdm import tqdm

from molt.agents.base import _first_scalar
from molt.datasets import PromptDataset
from molt.datasets.utils import blending_datasets
from molt.trainer.algorithm.experience import balance_experiences
from molt.trainer.algorithm.kl_controller import AdaptiveKLController, FixedKLController
from molt.trainer.eval_schedule import async_eval_due, buffer_until_eval_payload, checkpoint_aligned_eval_due
from molt.trainer.fsdp import FsdpStrategy
from molt.trainer.resume_state import resolve_rollout_resume_position
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker
from molt.trainer.rollout.samples_generator import SamplesGenerator
from molt.trainer.terminal_checkpoint import handle_terminal_payload, make_terminal_payload
from molt.trainer.vllm.vllm_engine import batch_vllm_engine_call
from molt.trainer.workers.actor_group import RayActorGroup
from molt.utils.distributed_sampler import DistributedSampler
from molt.utils.logging_utils import TensorboardLogger, WandbLogger, init_logger
from molt.utils.utils import get_tokenizer
from molt.utils.vlm_utils import rebuild_mm_train_inputs

logger = init_logger(__name__)


def prepare_datasets(strategy, tokenizer):
    args = strategy.args
    eval_only = bool(getattr(args.eval, "eval_only", False))

    # BOTH runner types consume the SAME chat-format dataset (--data.apply_chat_template);
    # Runner.PRERENDER_PROMPT only decides WHERE the template is applied. The step runner needs the
    # dataset to pre-render; the chat runner hands the raw messages to the chat server, which renders
    # them exactly once with the model's own template (a dataset pre-render would double-template and
    # drop the image on structured-content VLMs) — so for chat agents the flag is required, never
    # applied dataset-side.
    from molt.agents.base import load_agent_runner

    prerender = True
    if getattr(args.train, "agent_path", None):
        prerender = getattr(load_agent_runner(args.train.agent_path), "PRERENDER_PROMPT", True)
    if not prerender and not args.data.apply_chat_template:
        raise ValueError(
            "Chat agents consume the same chat-format dataset as step agents: pass "
            "--data.apply_chat_template. The dataset hands the messages through raw; the chat "
            "server renders them once with the model's own template."
        )

    if eval_only:
        # Standalone checkpoint evaluation must not require a dummy training
        # dataset. The training iterator is never touched in eval-only mode.
        prompts_dataset = None
        prompts_dataloader = None
    else:
        if not args.data.prompt_dataset:
            raise ValueError("--data.prompt_dataset is required unless --eval.eval_only is set")
        train_data = blending_datasets(
            args.data.prompt_dataset,
            args.data.prompt_probs,
            strategy,
            args.train.seed,
            max_count=args.data.max_samples,
            dataset_split=args.data.prompt_split,
        )

        # Create train dataset
        train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))
        prompts_dataset = PromptDataset(train_data, tokenizer, strategy, prerender=prerender)
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=not getattr(args.data, "disable_prompt_shuffle", False),
            collate_fn=prompts_dataset.collate_fn,
            num_workers=args.data.dataloader_num_workers,
        )

    # Create eval dataset if eval data exists
    if getattr(args.eval, "dataset", None):
        eval_data = blending_datasets(
            args.eval.dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=args.eval.split,
        )
        eval_data = eval_data.select(range(min(args.data.max_samples, len(eval_data))))
        eval_dataset = PromptDataset(eval_data, tokenizer, strategy, prerender=prerender)
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn,
            num_workers=args.data.dataloader_num_workers,
        )
    else:
        eval_dataloader = None

    if eval_only:
        # Model initialization still expects a positive scheduler horizon even
        # though no optimizer step will run.
        max_steps = 1
    elif args.train.force_on_policy:
        # On-policy: one optimizer step per rollout batch (per epoch), regardless
        # of how many samples multi-turn flatten produces. The generator consumes
        # rollout.batch_size prompt-groups per round. The generator emits a final
        # short batch, so scheduler horizon uses ceil-div (including datasets
        # smaller than one rollout batch) rather than silently omitting it.
        rollout_batch_size = int(args.rollout.batch_size)
        if rollout_batch_size <= 0:
            raise ValueError(f"rollout.batch_size must be positive, got {rollout_batch_size}")
        rollout_batches = (len(prompts_dataset) + rollout_batch_size - 1) // rollout_batch_size
        max_steps = rollout_batches * args.train.num_episodes * args.train.max_epochs
    else:
        max_steps = (
            len(prompts_dataset)
            * args.rollout.n_samples_per_prompt
            // args.train.batch_size
            * args.train.num_episodes
            * args.train.max_epochs
        )
    return prompts_dataloader, eval_dataloader, max_steps


def compute_eval_metrics(eval_dataloader, samples_list, n_samples_per_prompt):
    """Compute pass@k eval metrics from generated samples.

    Robust to dropped rollouts: samples are grouped by their rollout ``group_id``
    (falling back to the prompt string when group ids are absent) rather than a
    rigid ``reshape(-1, n_samples_per_prompt)``. A rollout dropped during
    generation (empty / zero-action / VLM-truncated — see
    ``SamplesGenerator._process_response_into_experience``) only shrinks that
    prompt's group instead of crashing the reshape or misaligning samples across
    prompt boundaries.
    """
    if not samples_list:
        return {}

    prompt_to_datasource = {}
    expected_group_count = 0
    for datasources, prompts, labels, _images, _tools in eval_dataloader:
        expected_group_count += len(prompts)
        for prompt, datasource in zip(prompts, datasources):
            if isinstance(prompt, list):
                # Chat rows pass through as messages; key on the last user turn's text —
                # the same scalar ChatAgentRunner stores as Trajectory.prompt.
                texts = [
                    m.get("content") for m in prompt if m.get("role") == "user" and isinstance(m.get("content"), str)
                ]
                prompt = texts[-1] if texts else str(prompt)
            prompt_to_datasource[prompt] = datasource

    # Each Experience here is a single rollout sample (B=1). Group the per-sample
    # scalars by the rollout group_id (one uuid per prompt group, i.e.
    # per eval prompt instance), NOT by the prompt STRING: two distinct eval rows
    # that render to the same string (same question across blended eval sets, dup
    # rows, or short templated prompts) would otherwise merge into one pass@k
    # group and distort pass1/count/eval_num_samples. Fall back to the prompt
    # string only when group ids are absent. Datasource is still resolved via the
    # prompt string (a benign attribution choice for genuine duplicates).
    grouped: Dict[str, Dict[str, list]] = {}
    group_order = []
    group_prompt: Dict[str, str] = {}
    # Chat rollouts can emit multiple trainable segments for one terminal episode.
    # Keep raw info for fail-closed invariants, but log terminal metrics once per rollout.
    global_info: Dict[str, list[float]] = {}
    raw_info: Dict[str, list[float]] = {}
    rollout_info: Dict[str, Dict[object, float]] = {}
    for s in samples_list:
        prompt = s.prompts[0]
        key = s.group_ids[0] if getattr(s, "group_ids", None) else prompt
        if key not in grouped:
            grouped[key] = {
                "rewards": {},
                "lengths": {},
                "truncated": {},
                "rollout_ids": set(),
            }
            group_order.append(key)
            group_prompt[key] = prompt
        rollout_ids = getattr(s, "rollout_ids", None)
        rollout_id = rollout_ids[0] if rollout_ids else id(s)
        grouped[key]["rollout_ids"].add(rollout_id)

        # stitch_session emits one Experience per trainable segment and copies
        # the terminal reward to each. Core eval statistics must operate on
        # trajectories, not weight long multi-turn rollouts by segment count.
        reward = _first_scalar(s.rewards)
        if reward is not None:
            previous = grouped[key]["rewards"].get(rollout_id)
            if previous is not None and reward != previous:
                raise ValueError(f"Inconsistent rewards for eval rollout {rollout_id!r}: {previous!r} != {reward!r}")
            grouped[key]["rewards"].setdefault(rollout_id, reward)
        length = _first_scalar(s.response_length)
        if length is not None:
            grouped[key]["lengths"][rollout_id] = grouped[key]["lengths"].get(rollout_id, 0.0) + length
        sample_info = getattr(s, "info", None) or {}
        # Chat agents expose the generation/context cutoff explicitly because
        # Experience.truncated also includes a terminal environment horizon.
        # Preserve the legacy fallback for agents that do not emit the new
        # signal, while keeping eval_truncated_rate generation-specific when it
        # is available. The terminal info["truncated"] aggregate remains a
        # separate eval_truncated metric.
        generation_truncated = sample_info.get("generation_truncated")
        truncated = _first_scalar(
            s.truncated if generation_truncated is None else generation_truncated
        )
        if truncated is not None:
            grouped[key]["truncated"][rollout_id] = max(grouped[key]["truncated"].get(rollout_id, 0.0), truncated)
        for metric_name, metric_value in sample_info.items():
            scalar = _first_scalar(metric_value)
            if isinstance(scalar, (int, float, bool)):
                scalar = float(scalar)
                raw_info.setdefault(metric_name, []).append(scalar)
                by_rollout = rollout_info.setdefault(metric_name, {})
                if metric_name in {"response_clip_ratio", "generation_truncated"} or metric_name.endswith(
                    "_backend_error"
                ):
                    # These flags may be segment-local. A single affected segment
                    # makes the terminal rollout affected.
                    by_rollout[rollout_id] = max(by_rollout.get(rollout_id, 0.0), scalar)
                else:
                    # stitch_session copies terminal reward/info to every segment
                    # of a rollout. Count the first copy only so the metric is not
                    # biased by variable turn/segment counts.
                    by_rollout.setdefault(rollout_id, scalar)

    global_info = {
        metric_name: list(values_by_rollout.values()) for metric_name, values_by_rollout in rollout_info.items()
    }

    metrics = {}
    for key in group_order:
        g = grouped[key]
        prompt = group_prompt[key]
        rewards = list(g["rewards"].values())
        if not rewards:
            continue
        ds = prompt_to_datasource.get(prompt, "unknown")
        if ds not in metrics:
            metrics[ds] = {
                f"pass{n_samples_per_prompt}": 0.0,
                "pass1": 0.0,
                "count": 0,
                "lengths": [],
                "truncated": [],
            }
        if n_samples_per_prompt > 1:
            metrics[ds][f"pass{n_samples_per_prompt}"] += max(rewards)
        metrics[ds]["pass1"] += sum(rewards) / len(rewards)
        metrics[ds]["count"] += 1
        metrics[ds]["lengths"].extend(g["lengths"].values())
        metrics[ds]["truncated"].extend(g["truncated"].values())

    logs = {}
    total_lengths = []
    total_truncated = []
    for ds, m in metrics.items():
        logs[f"eval_{ds}_pass{n_samples_per_prompt}"] = m[f"pass{n_samples_per_prompt}"] / m["count"]
        logs[f"eval_{ds}_pass1"] = m["pass1"] / m["count"]
        if m["lengths"]:
            logs[f"eval_{ds}_response_length_mean"] = sum(m["lengths"]) / len(m["lengths"])
            total_lengths.extend(m["lengths"])
        if m["truncated"]:
            logs[f"eval_{ds}_truncated_rate"] = sum(m["truncated"]) / len(m["truncated"])
            total_truncated.extend(m["truncated"])

    if total_lengths:
        logs["eval_response_length_mean"] = sum(total_lengths) / len(total_lengths)
    if total_truncated:
        logs["eval_truncated_rate"] = sum(total_truncated) / len(total_truncated)
    # Agent-specific metrics (e.g. gomoku_win/raw_reward/legal_rate) live in
    # Experience.info. Expose global eval means so best-checkpoint selection can
    # use the true task metric instead of the shaped training reward/pass1.
    for metric_name, values in sorted(global_info.items()):
        if values:
            logs[f"eval_{metric_name}"] = sum(values) / len(values)

    # Async evaluation can overlap a policy broadcast when partial rollouts are
    # enabled.  A mean policy version hides that race: even one game generated
    # across two versions makes the aggregate unsuitable for best-checkpoint
    # selection.  Emit an explicit invariant that callers can enforce.
    policy_starts = raw_info.get("policy_version_start", [])
    policy_ends = raw_info.get("policy_version_end", [])
    policy_frozen = raw_info.get("policy_frozen", [])
    logs["eval_policy_consistent"] = 0.0
    if policy_starts or policy_ends:
        policy_versions = policy_starts + policy_ends
        logs["eval_policy_version_min"] = min(policy_versions)
        logs["eval_policy_version_max"] = max(policy_versions)
        logs["eval_policy_version_span"] = max(policy_versions) - min(policy_versions)
        pairwise_consistent = (
            len(policy_starts) == len(samples_list)
            and len(policy_ends) == len(samples_list)
            and len(policy_frozen) == len(samples_list)
            and all(value == 1.0 for value in policy_frozen)
            and all(start == end for start, end in zip(policy_starts, policy_ends))
        )
        logs["eval_policy_consistent"] = float(pairwise_consistent and logs["eval_policy_version_span"] == 0)

    actual_group_count = len(grouped)
    complete_group_count = sum(len(grouped[key]["rollout_ids"]) == n_samples_per_prompt for key in group_order)
    actual_rollout_count = sum(len(grouped[key]["rollout_ids"]) for key in group_order)
    logs["eval_group_count"] = float(actual_group_count)
    logs["eval_expected_group_count"] = float(expected_group_count)
    logs["eval_num_rollouts"] = float(actual_rollout_count)
    logs["eval_expected_num_rollouts"] = float(expected_group_count * n_samples_per_prompt)
    logs["eval_groups_complete"] = float(
        actual_group_count == expected_group_count and complete_group_count == expected_group_count
    )
    backend_error_rollouts = set()
    for key in ("nanobot_backend_error", "esibench_backend_error"):
        backend_error_rollouts.update(
            rollout_id for rollout_id, value in rollout_info.get(key, {}).items() if bool(value)
        )
    logs["eval_backend_error_count"] = float(len(backend_error_rollouts))
    logs["eval_num_samples"] = float(len(samples_list))

    return logs


def eval_matches_current_policy(eval_metrics, current_policy_versions) -> bool:
    """Whether the actor/vLLM policy now resident is exactly the evaluated one."""
    if not current_policy_versions or eval_metrics.get("eval_policy_consistent") != 1.0:
        return False
    eval_min = eval_metrics.get("eval_policy_version_min")
    eval_max = eval_metrics.get("eval_policy_version_max")
    if eval_min is None or eval_max is None or eval_min != eval_max:
        return False
    return all(float(version) == float(eval_min) for version in current_policy_versions)


def eval_is_checkpoint_safe(eval_metrics) -> bool:
    """Fail-closed gate for selecting a best checkpoint from asynchronous eval."""
    return bool(
        eval_metrics
        and eval_metrics.get("eval_policy_consistent") == 1.0
        and eval_metrics.get("eval_checkpoint_policy_matches") == 1.0
        and eval_metrics.get("eval_groups_complete") == 1.0
        and eval_metrics.get("eval_backend_error_count") == 0.0
    )


def validate_force_on_policy_runtime_config(train_args) -> None:
    """Fail closed unless the sampler/trainer topology is strictly synchronous."""
    if not bool(getattr(train_args, "force_on_policy", False)):
        return
    queue_size = int(getattr(train_args, "async_queue_size", 1))
    if queue_size != 1:
        raise ValueError("--train.force_on_policy requires --train.async_queue_size 1")
    if bool(getattr(train_args, "partial_rollout_enable", False)):
        raise ValueError("--train.force_on_policy requires --train.partial_rollout_enable disabled")


def _exact_nonnegative_version(value, label: str) -> int:
    scalar = _first_scalar(value)
    if scalar is None or isinstance(scalar, (bool, str, bytes)):
        raise RuntimeError(f"{label} must be a finite nonnegative integer, got {scalar!r}")
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be a finite nonnegative integer, got {scalar!r}") from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise RuntimeError(f"{label} must be a finite nonnegative integer, got {scalar!r}")
    return int(numeric)


def _provenance_rows(value, batch_size: int, label: str) -> list:
    if isinstance(value, torch.Tensor):
        rows = value.detach().flatten().tolist()
    elif isinstance(value, (list, tuple)):
        rows = [_first_scalar(item) for item in value]
    else:
        rows = [value]
    if len(rows) != batch_size:
        raise RuntimeError(f"{label} has {len(rows)} row(s), expected Experience batch size {batch_size}")
    return rows


def _experience_batch_size(sample, info) -> int:
    sequences = getattr(sample, "sequences", None)
    if isinstance(sequences, torch.Tensor) and sequences.dim() > 0:
        return int(sequences.shape[0])
    prompts = getattr(sample, "prompts", None)
    if isinstance(prompts, list) and prompts:
        return len(prompts)
    for key in ("policy_version_start", "policy_version_end", "policy_frozen"):
        value = info.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.numel())
        if isinstance(value, (list, tuple)):
            return len(value)
    return 1


def _backend_error_flag(value, label: str) -> bool:
    scalar = _first_scalar(value)
    if isinstance(scalar, bool):
        return scalar
    if isinstance(scalar, str):
        normalized = scalar.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0", ""}:
            return False
    if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
        numeric = float(scalar)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(numeric)
    raise RuntimeError(f"{label} must be a boolean backend-error flag, got {scalar!r}")


def validate_force_on_policy_batch(rollout_samples, current_policy_versions) -> int:
    """Prove that every segment came from the single currently loaded policy.

    The scheduler ``global_step`` is only a logical update counter. vLLM's
    physical version advances once per packed weight transfer, so provenance
    must be checked against the engines themselves rather than ``global_step``.
    """
    versions = [
        _exact_nonnegative_version(version, f"vLLM engine {index} version")
        for index, version in enumerate(current_policy_versions)
    ]
    if not versions or len(set(versions)) != 1:
        raise RuntimeError(f"force_on_policy requires one synchronized vLLM version, got {versions}")
    expected_version = versions[0]
    if not rollout_samples:
        raise RuntimeError("force_on_policy received an empty rollout batch")

    for index, sample in enumerate(rollout_samples):
        info = getattr(sample, "info", None) or {}
        if any(key not in info for key in ("policy_version_start", "policy_version_end", "policy_frozen")):
            raise RuntimeError(f"force_on_policy segment {index} is missing policy provenance")
        batch_size = _experience_batch_size(sample, info)
        for key in ("nanobot_backend_error", "esibench_backend_error"):
            if key not in info:
                continue
            backend_rows = _provenance_rows(info[key], batch_size, key)
            for row, backend_value in enumerate(backend_rows):
                if _backend_error_flag(backend_value, f"segment {index} row {row} {key}"):
                    raise RuntimeError(
                        f"force_on_policy segment {index} row {row} reported {key}; "
                        "refusing to train on a backend failure"
                    )
        starts = _provenance_rows(info["policy_version_start"], batch_size, "policy_version_start")
        ends = _provenance_rows(info["policy_version_end"], batch_size, "policy_version_end")
        frozen_rows = _provenance_rows(info["policy_frozen"], batch_size, "policy_frozen")
        for row, (start_value, end_value, frozen_value) in enumerate(zip(starts, ends, frozen_rows)):
            start = _exact_nonnegative_version(start_value, f"segment {index} row {row} start version")
            end = _exact_nonnegative_version(end_value, f"segment {index} row {row} end version")
            if isinstance(frozen_value, bool):
                frozen = float(frozen_value)
            elif isinstance(frozen_value, (str, bytes)):
                frozen = float("nan")
            else:
                try:
                    frozen = float(frozen_value)
                except (TypeError, ValueError, OverflowError):
                    frozen = float("nan")
            if not math.isfinite(frozen) or frozen != 1.0 or start != end or start != expected_version:
                raise RuntimeError(
                    "force_on_policy provenance mismatch for segment "
                    f"{index} row {row}: start={start_value}, end={end_value}, frozen={frozen_value}, "
                    f"current_vllm_version={expected_version}"
                )
    return expected_version


class BaseRLTrainer:
    """Training-side base class for non-critic policy RL."""

    def __init__(
        self,
        strategy: FsdpStrategy,
        actor_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines,
        tokenizer,
        critic_model_group: RayActorGroup = None,
    ) -> None:
        self.strategy = strategy
        self.args = strategy.args

        self.actor_model_group = actor_model_group
        self.reference_model_group = reference_model_group
        self.critic_model_group = critic_model_group
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer

        # Critic warmup: freeze the actor's policy update for the first N optimizer
        # steps so the value model can fit the initial rollouts before its early,
        # high-variance advantages start moving the policy. The critic still trains
        # (and the actor buffer is still drained) while frozen. Only with a critic;
        # 0 disables.
        self.freezing_actor_steps = self.args.actor.freezing_steps if critic_model_group is not None else 0

        if self.args.algo.kl.target:
            self.kl_ctl = AdaptiveKLController(
                self.args.algo.kl.init_coef, self.args.algo.kl.target, self.args.algo.kl.horizon
            )
        else:
            self.kl_ctl = FixedKLController(self.args.algo.kl.init_coef)

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            tokenizer,
            critic_model_group=self.critic_model_group,
        )

        # Tracking backends
        self.wandb_logger = WandbLogger(self.args) if self.args.logger.wandb.key else None
        self.tensorboard_logger = TensorboardLogger(self.args) if self.args.logger.tensorboard_dir else None

        # Best eval metric tracking
        self.best_eval_metric_value = float("-inf")
        self.best_eval_metric_key = getattr(self.args.ckpt, "best_metric_key", "") or ""
        self._latest_eval_metric_value = None

    def restore_best_metric_tracker(self, checkpoint_states) -> None:
        if not checkpoint_states:
            return

        checkpoint_metric_key = checkpoint_states.get("best_eval_metric_key")
        checkpoint_metric_value = checkpoint_states.get("best_eval_metric_value")

        if checkpoint_metric_key:
            self.best_eval_metric_key = checkpoint_metric_key
        if checkpoint_metric_value is not None:
            self.best_eval_metric_value = checkpoint_metric_value
            self._latest_eval_metric_value = checkpoint_metric_value

    def fit(self, global_step: int = 0) -> None:
        raise NotImplementedError("fit method is not implemented")

    def train_step(self, rollout_samples, global_step: int) -> Tuple[Dict, int]:
        # Turn raw rollouts into policy-gradient trajectories with rewards.
        t0 = time.time()
        mm_reprocess_time = 0.0
        compact_raw_bytes = 0
        compact_tensor_bytes = 0
        compact_samples = [
            sample for sample in rollout_samples if getattr(sample, "mm_train_input_specs", None)
        ]
        if compact_samples:
            requested_workers = int(os.environ.get("MOLT_VLM_REPROCESS_WORKERS", "8"))
            if requested_workers <= 0:
                raise ValueError(f"MOLT_VLM_REPROCESS_WORKERS must be positive, got {requested_workers}")

            for sample in compact_samples:
                if sample.mm_train_inputs:
                    raise RuntimeError("Compact VLM sample unexpectedly retained mm_train_inputs")
                if len(sample.images) != 1 or len(sample.mm_train_input_specs) != 1:
                    raise RuntimeError(
                        "Compact VLM sample must carry one per-sample image list and one transport spec"
                    )
                for image in sample.images[0]:
                    compact_raw_bytes += int(image.width) * int(image.height) * len(image.getbands())
                for expected in sample.mm_train_input_specs[0].values():
                    compact_tensor_bytes += math.prod(expected["shape"]) * expected["values"].element_size()

            def rebuild_one(sample):
                return rebuild_mm_train_inputs(
                    self.tokenizer, sample.images[0], sample.mm_train_input_specs[0]
                )

            rebuild_t0 = time.time()
            worker_count = min(requested_workers, len(compact_samples))
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                rebuilt_inputs = list(pool.map(rebuild_one, compact_samples))
            mm_reprocess_time = time.time() - rebuild_t0
            for sample, rebuilt in zip(compact_samples, rebuilt_inputs):
                sample.mm_train_inputs = [rebuilt]
                sample.images = []
                sample.mm_train_input_specs = []
            logger.info(
                "Rebuilt compact VLM inputs for %s samples in %.2fs with %s workers",
                len(compact_samples),
                mm_reprocess_time,
                worker_count,
            )
        experiences = self.experience_maker.build_experiences(rollout_samples)
        make_experience_time = time.time() - t0

        # Peek at the first decoded sample for quick sanity check.
        sample0 = [
            self.tokenizer.decode(experiences[0].sequences[0], skip_special_tokens=True),
            experiences[0].info["reward"][0].item(),
        ]
        logger.info(f"Sample: {sample0}")
        if os.environ.get("MOLT_DEBUG_ROLLOUT") == "1":
            debug_rows = []
            for exp_idx, exp in enumerate(experiences):
                batch = exp.sequences.shape[0]
                rewards = exp.info.get("reward")
                returns = exp.info.get("return")
                group_stds = exp.info.get("group_reward_std")
                for row in range(batch):
                    index = exp.index[row] if isinstance(exp.index, list) and row < len(exp.index) else row
                    debug_rows.append(
                        {
                            "exp": exp_idx,
                            "row": row,
                            "index": int(index),
                            "reward": rewards[row].item() if isinstance(rewards, torch.Tensor) else None,
                            "return": returns[row].item() if isinstance(returns, torch.Tensor) else None,
                            "group_reward_std": (
                                group_stds[row].item() if isinstance(group_stds, torch.Tensor) else None
                            ),
                            "text": self.tokenizer.decode(exp.sequences[row], skip_special_tokens=True),
                        }
                    )
            logger.info(f"RolloutDebug: {debug_rows}")

        # Compute ground-truth rollout stats BEFORE dynamic batch splitting.
        all_rewards = torch.cat([exp.info["reward"] for exp in experiences if "reward" in exp.info])
        all_response_lengths = torch.cat(
            [exp.response_length for exp in experiences if exp.response_length is not None]
        )
        all_truncated = torch.cat([exp.truncated for exp in experiences if exp.truncated is not None])
        rollout_stats = {
            "rollout/reward_mean": all_rewards.float().mean().item(),
            "rollout/reward_std": all_rewards.float().std().item() if len(all_rewards) > 1 else 0.0,
            "rollout/response_length_mean": all_response_lengths.float().mean().item(),
            "rollout/truncated_rate": all_truncated.float().mean().item(),
            "rollout/num_samples": float(len(all_rewards)),
        }

        # Balance experiences so every DP rank gets an equal sample count (both
        # dynamic and static paths). Unequal counts -> different per-rank
        # training-step counts -> mismatched collective shapes (per-microbatch
        # global_token_count all-reduce and FSDP reduce-scatter) -> NCCL hang.
        # Multi-turn agents (variable step-samples per rollout) and rollout counts
        # not divisible by the DP degree both break this invariant without re-balance.
        pre_balance_samples = sum(len(exp.sequences) for exp in experiences)
        experiences = balance_experiences(experiences, self.args)
        post_balance_samples = sum(len(exp.sequences) for exp in experiences)
        rollout_stats["rollout/dp_balance_padding_samples"] = float(post_balance_samples - pre_balance_samples)
        # Backward-compatible proof for dashboards/alerts: padding preserves
        # every real segment, so this legacy counter must now stay exactly zero.
        rollout_stats["rollout/dp_balance_dropped_samples"] = 0.0

        # Push experiences to actor shards (and the critic, which trains on the same
        # batch with values + returns) before optimization.
        refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
        if self.critic_model_group is not None:
            refs += self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
        ray.get(refs)

        # Perform policy optimization for the actor and gather metrics. During
        # critic warmup the actor is frozen — no policy update — while the value
        # model trains on the same rollouts.
        actor_frozen = global_step < self.freezing_actor_steps
        t0 = time.time()
        status = self.policy_train(train_actor=not actor_frozen)
        status["actor_frozen"] = float(actor_frozen)
        policy_train_time = time.time() - t0

        # Sync weights to vLLM (skipped while the actor is frozen: its weights are
        # unchanged, so the live vLLM copy is already current).
        t0 = time.time()
        # Reset per-step so a frozen/skipped broadcast reports 0 (not a stale
        # value); the TrainingActor.broadcast_to_vllm override sets these when
        # it runs (lock-wait vs actual transfer).
        self._broadcast_lock_wait_s = 0.0
        self._broadcast_transfer_s = 0.0
        if self.vllm_engines is not None and not actor_frozen:
            self.broadcast_to_vllm()
        broadcast_time = time.time() - t0

        # Refresh KL controller with the latest measurement (no-op for FixedKLController).
        if "kl" in status:
            self.kl_ctl.update(status["kl"], self.args.rollout.batch_size * self.args.rollout.n_samples_per_prompt)

        # Per-phase timing breakdown. timing/broadcast is the TOTAL; it splits into
        # broadcast_lock_wait (trainer blocked on the vllm_lock held by the overlapping
        # rollout generation — overlaps wall-clock, NOT a transfer cost) and
        # broadcast_transfer (the actual NCCL weight sync). Keep all three so the
        # total stays comparable while the lock-wait is no longer mistaken for transfer.
        status["timing/make_experience"] = make_experience_time
        status["timing/mm_reprocess"] = mm_reprocess_time
        if compact_samples:
            status["rollout/mm_transport_raw_gib"] = compact_raw_bytes / (1024**3)
            status["rollout/mm_preprocessed_gib"] = compact_tensor_bytes / (1024**3)
            status["rollout/mm_preprocessed_to_raw_ratio"] = compact_tensor_bytes / max(compact_raw_bytes, 1)
        status["timing/policy_train"] = policy_train_time
        status["timing/broadcast"] = broadcast_time
        status["timing/broadcast_lock_wait"] = self._broadcast_lock_wait_s
        status["timing/broadcast_transfer"] = self._broadcast_transfer_s

        # Merge rollout stats (ground-truth, pre-dynamic-batch)
        status.update(rollout_stats)

        status["generated_samples"] = sample0
        return status, global_step + 1

    def policy_train(self, train_actor: bool = True) -> Dict:
        """Run one actor optimization step (then the critic) and return merged status.

        Sequential, not concurrent: colocated actor and critic share GPUs, so running
        both trainings at once would double resident memory. Disaggregated setups pay
        a small no-overlap cost here. ``train_actor=False`` (critic warmup) skips the
        policy update but still drains the actor replay buffer.
        """
        refs = self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value, train=train_actor)
        status: dict = {}
        for result in ray.get(refs):
            status.update(result)
        # Fail loudly when the vLLM-IS filter dropped every sequence: the policy
        # gradient is exactly zero and the run silently optimizes nothing. The
        # usual cause is rollout-vs-train forward mismatch, for MoE models most
        # often unreplayed expert routing — enable --train.routing_replay.
        if status.get("is_filter_ratio", 0.0) >= 0.999:
            logger.warning(
                f"is_filter_ratio={status['is_filter_ratio']:.3f}: the vLLM importance-sampling filter dropped "
                f"(nearly) every sequence — zero policy gradient this step (vllm_kl={status.get('vllm_kl')}). "
                "Rollout and training forwards disagree; for MoE models enable --train.routing_replay."
            )
        if self.critic_model_group is not None:
            # Colocated actor and critic are separate processes sharing the same GPUs.
            # Release the actor's cached GPU blocks back to the driver before the critic
            # trains so the critic's activation / cuDNN-attn-workspace allocations have
            # headroom (the per-fit empty_cache does this too; this makes the release
            # deterministic at the actor->critic boundary).
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))
            critic_refs = self.critic_model_group.async_run_method(method_name="fit")
            for result in ray.get(critic_refs):
                status.update(result)
            ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        return status

    def broadcast_to_vllm(self) -> None:
        """Broadcast actor weights to vLLM engines."""
        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        # NOTE: We keep vLLM in weights-only state after weight sync.
        # KV cache will be woken up before generation in SamplesGenerator.

    def save_best_checkpoint(self, eval_metrics, global_step, client_states=None):
        """Save checkpoint if eval metric is the best so far.

        When best_metric_key is 'none' or no eval_*_pass1 metric is present,
        this is a no-op — regular save_steps checkpoints still save the most recent.
        """
        if not eval_metrics or self.best_eval_metric_key == "none":
            return

        if not eval_is_checkpoint_safe(eval_metrics):
            logger.warning(
                "Skipping best-checkpoint selection: policy_consistent=%s, "
                "checkpoint_policy_matches=%s, groups_complete=%s, backend_errors=%s "
                "(eval versions min=%s max=%s).",
                eval_metrics.get("eval_policy_consistent"),
                eval_metrics.get("eval_checkpoint_policy_matches"),
                eval_metrics.get("eval_groups_complete"),
                eval_metrics.get("eval_backend_error_count"),
                eval_metrics.get("eval_policy_version_min"),
                eval_metrics.get("eval_policy_version_max"),
            )
            return

        if self.best_eval_metric_key:
            metric_key = self.best_eval_metric_key if self.best_eval_metric_key in eval_metrics else None
        else:
            # Auto-detect: prefer eval_*_pass1 metric.
            metric_key = next((k for k in sorted(eval_metrics) if k.endswith("_pass1")), None)
            if metric_key is not None:
                self.best_eval_metric_key = metric_key
        if metric_key is None:
            return

        current_value = eval_metrics[metric_key]
        self._latest_eval_metric_value = current_value
        prev_best = self.best_eval_metric_value

        if current_value > self.best_eval_metric_value:
            self.best_eval_metric_value = current_value
            logger.info(
                f"New best eval metric: {metric_key}={current_value:.4f} at step {global_step} "
                f"(previous best: {prev_best if prev_best > float('-inf') else 'N/A'})"
            )

            client_states = client_states or {}
            client_states["best_eval_metric_key"] = metric_key
            client_states["best_eval_metric_value"] = current_value
            client_states["checkpoint_metric_key"] = metric_key

            tag = f"best_global_step{global_step}"
            refs = self.actor_model_group.async_run_method(
                method_name="save_checkpoint",
                tag=tag,
                client_states=client_states,
                metric_value=current_value,
                metric_key=metric_key,
            )
            if self.critic_model_group is not None:
                refs += self.critic_model_group.async_run_method(
                    method_name="save_checkpoint", tag=tag, metric_value=current_value, metric_key=metric_key
                )
            ray.get(refs)
            logger.info(f"Saved best checkpoint: {tag} ({metric_key}={current_value:.4f})")

    def save_logs_and_checkpoints(self, global_step: int, logs_dict=None, client_states=None) -> bool:
        logs_dict = logs_dict or {}
        if global_step % self.args.logger.logging_steps == 0:
            if self.wandb_logger:
                self.wandb_logger.log_train(global_step, logs_dict)
            if self.tensorboard_logger:
                self.tensorboard_logger.log_train(global_step, logs_dict)

        # save ckpt
        client_states = client_states or {}
        if global_step % self.args.ckpt.save_steps == 0:
            tag = f"global_step{global_step}"
            # Persist best-metric tracker on every rolling save so chain
            # successors can `restore_best_metric_tracker` from the latest
            # rolling DCP — not just from the `best_*` ckpt (which gets
            # rotated out by max_num cleanup).
            client_states["best_eval_metric_key"] = self.best_eval_metric_key
            client_states["best_eval_metric_value"] = self.best_eval_metric_value
            metric_value = self._latest_eval_metric_value
            metric_key = client_states.get("checkpoint_metric_key") or self.best_eval_metric_key or None
            refs = self.actor_model_group.async_run_method(
                method_name="save_checkpoint",
                tag=tag,
                client_states=client_states,
                metric_value=metric_value,
                metric_key=metric_key,
            )
            if self.critic_model_group is not None:
                refs += self.critic_model_group.async_run_method(
                    method_name="save_checkpoint", tag=tag, metric_value=metric_value, metric_key=metric_key
                )
            ray.get(refs)
            return True
        return False

    def save_terminal_checkpoint(self, global_step: int, client_states: Dict) -> None:
        """Persist an exhausted dataloader state even when save cadence is not due.

        The generator sends this state only after observing ``StopIteration``.
        Queue FIFO ordering guarantees all preceding rollout batches have been
        optimized when this method runs, so ``global_step`` names the completed
        endpoint rather than a speculative next step.
        """
        loader_state = client_states.get("data_loader_state_dict", {})
        if not bool(loader_state.get("_iterator_finished", False)):
            raise RuntimeError("refusing to save a terminal checkpoint before iterator exhaustion")
        tag = f"global_step{global_step}"
        metric_value = self._latest_eval_metric_value
        metric_key = client_states.get("checkpoint_metric_key") or self.best_eval_metric_key or None
        refs = self.actor_model_group.async_run_method(
            method_name="save_checkpoint",
            tag=tag,
            client_states=client_states,
            metric_value=metric_value,
            metric_key=metric_key,
        )
        if self.critic_model_group is not None:
            refs += self.critic_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, metric_value=metric_value, metric_key=metric_key
            )
        ray.get(refs)
        logger.info("Saved terminal checkpoint: %s", tag)

    def load_checkpoint_states_or_default(self) -> Dict:
        ckpt_path = os.path.join(self.args.ckpt.path, "_actor")
        if self.args.ckpt.load_enable and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            # Log scalars only; never f-string the whole dict (a legacy checkpoint
            # can carry a multi-MB sub-dict that would OOM the driver on resume).
            logger.info(
                "checkpoint_states: %s",
                {k: v for k, v in checkpoint_states.items() if not isinstance(v, (dict, list))},
            )
            return checkpoint_states
        return {
            "episode": 0,
            "global_step": 0,
            "total_consumed_prompts": 0,
            "data_loader_state_dict": {},
        }


@ray.remote(num_cpus=0)
class VLLMLock:
    """Cross-actor mutex for vLLM critical sections."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def acquire(self):
        await self._lock.acquire()

    async def release(self):
        self._lock.release()


@ray.remote
class GenerateSamplesActor:
    def __init__(
        self,
        pretrain,
        strategy,
        *,
        vllm_lock,
        rollout_queue,
        rollout_slots,
        router_url=None,
        version_source=None,
        **generate_kwargs,
    ):
        # No vllm_engines here: generation runs through the vllm-router via the runner
        # actors below; only the TrainingActor touches the engines (pause/refit/resume).
        self.args = strategy.args

        tokenizer = get_tokenizer(pretrain, None, "left", use_fast=not strategy.args.data.disable_fast_tokenizer)
        self.prompts_dataloader, self.eval_dataloader, self.max_steps = prepare_datasets(strategy, tokenizer)
        self.generate_kwargs = generate_kwargs

        # Rollout runs on a list of runner actors -> the shared vllm-router (generation),
        # grading in-process. Weight sync goes straight to the engines (bypasses the router).
        from molt.trainer.rollout.router import AgentRunnerActor

        num_runners = max(1, getattr(strategy.args.rollout, "num_runners", 2))
        agent_runners = [
            AgentRunnerActor.remote(
                strategy.args.train.agent_path,
                router_url,
                model_path=pretrain,
                version_source=version_source,
            )
            for _ in range(num_runners)
        ]
        ray.get([r.ready.remote() for r in agent_runners])
        self.samples_generator = SamplesGenerator(
            strategy=strategy,
            prompts_dataloader=self.prompts_dataloader,
            eval_dataloader=self.eval_dataloader,
            tokenizer=tokenizer,
            agent_runners=agent_runners,
        )

        self.vllm_lock = vllm_lock
        self._partial_rollout = getattr(strategy.args.train, "partial_rollout_enable", False)
        self.rollout_queue = rollout_queue
        self.rollout_slots = rollout_slots
        self._last_eval_step = -1
        # Eval fires once global_step crosses this threshold, then it advances by
        # eval_steps. Using `>=` (catch-up) instead of `% eval_steps == 0` is
        # robust to the async slot's global_step jumping past an exact multiple,
        # which silently skipped whole eval points (e.g. eval@20 never ran).
        self._next_eval_step = strategy.args.eval.steps
        # Optional baseline eval at global_step 0 (pre-RL model). Fresh runs only:
        # resume starts at a saved global_step > 0, so this never adds a redundant
        # eval on resume.
        self._eval_at_start = getattr(strategy.args.eval, "eval_at_start", False)
        self._eval_only = getattr(strategy.args.eval, "eval_only", False)

    def get_max_steps(self):
        return self.max_steps

    def load_dataloader_state_dict(self, state_dict, rollout_generator_state_dict=None):
        self.prompts_dataloader.load_state_dict(state_dict)
        self.samples_generator.load_state_dict(rollout_generator_state_dict)

    def fit(self, episode: int, total_consumed_prompts: int) -> None:
        eval_steps = self.args.eval.steps
        if self._eval_only:
            if self.eval_dataloader is None:
                raise ValueError("eval-only mode requires an evaluation dataset")
            global_step = self.rollout_slots.get(block=True)
            logger.info("Starting frozen-policy eval-only generation...")
            eval_kwargs = {
                **self.generate_kwargs,
                "temperature": self.args.eval.temperature,
                "n_samples_per_prompt": self.args.eval.n_samples_per_prompt,
                "rollout_kind": "eval",
                "policy_version": global_step,
                "policy_frozen": True,
            }
            if not self._partial_rollout:
                ray.get(self.vllm_lock.acquire.remote())
            try:
                samples_list = self.samples_generator.generate_eval_samples(**eval_kwargs)
            finally:
                if not self._partial_rollout:
                    ray.get(self.vllm_lock.release.remote())
            eval_metrics = compute_eval_metrics(
                self.eval_dataloader, samples_list, self.args.eval.n_samples_per_prompt
            )
            logger.info(f"Frozen-policy eval-only completed: {eval_metrics}")
            self.rollout_queue.put(("eval", global_step, eval_metrics), block=True)
            self.rollout_queue.put("done", block=True)
            return
        # eval_at_start must fire only on a genuinely FRESH run. The
        # GenerateSamplesActor transiently reads global_step 0 at startup even on
        # resume (the restored step propagates through rollout_slots a beat later),
        # so gating on global_step alone would also eval@0 on resume. The
        # consumed-prompt counter is 0 only on a fresh start (>0 once a checkpoint
        # is loaded), so it's the reliable fresh-run signal.
        fresh_start = total_consumed_prompts == 0
        # On resume, _next_eval_step was re-init to eval_steps (< the restored
        # global_step), so the catch-up `global_step >= _next_eval_step` would fire
        # one off-cadence eval at the resumed step every resubmit. Sync the threshold
        # past the resume point on the first real step so eval lands only on the
        # normal eval_steps multiples a continuous run would hit (no per-resume eval).
        # Fresh runs are already in sync (step-0 baseline + _next_eval_step=eval_steps).
        eval_synced = fresh_start
        terminal_client_states = None
        for ep in range(episode, self.args.train.num_episodes):
            # Reshuffle prompts each episode (seed+epoch); without this every
            # episode replays the identical order. The generator rebuilds its
            # iterator at the episode boundary, so the new epoch takes effect on
            # the next iter() — and on resume `ep` matches the saved episode.
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(ep)
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{ep + 1}/{self.args.train.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                # Backpressure: slot token carries trainer's latest global_step for eval
                # timing. Time the block — in this async split, the generator stuck here
                # means the vLLM side is sitting IDLE waiting for the trainer to free a
                # slot (training slower than generation). This is the true vLLM-idle signal,
                # which the wall-clock generation_time alone cannot show.
                _slot_wait_t0 = time.time()
                global_step = self.rollout_slots.get(block=True)
                vllm_idle_wait = time.time() - _slot_wait_t0

                # Resume sync (once): advance the eval threshold to the next eval_steps
                # multiple after the restored step, so a resubmit doesn't trigger an
                # off-cadence eval at the resumed global_step.
                if not eval_synced and global_step > 0 and eval_steps > 0 and eval_steps != float("inf"):
                    self._next_eval_step = (global_step // eval_steps + 1) * eval_steps
                    eval_synced = True

                should_eval = self.eval_dataloader is not None and async_eval_due(
                    global_step=global_step,
                    last_eval_step=self._last_eval_step,
                    next_eval_step=self._next_eval_step,
                    eval_steps=eval_steps,
                    eval_at_start=self._eval_at_start,
                    fresh_start=fresh_start,
                )
                if should_eval:
                    self._last_eval_step = global_step
                    self._next_eval_step = (global_step // eval_steps + 1) * eval_steps
                    logger.info(f"Starting async evaluation at step {global_step}...")
                    # Independent eval sampling: each knob left unset (None) falls back to the
                    # rollout value in generate_kwargs; override only what's set for eval.
                    eval_n = self.args.eval.n_samples_per_prompt
                    if eval_n is None:
                        eval_n = self.args.rollout.n_samples_per_prompt
                    eval_kwargs = {**self.generate_kwargs, "n_samples_per_prompt": eval_n}
                    for key in ("temperature", "top_p", "max_new_tokens"):
                        override = getattr(self.args.eval, key)
                        if override is not None:
                            eval_kwargs[key] = override
                    eval_kwargs.update(
                        rollout_kind="eval",
                        policy_version=global_step,
                        policy_frozen=not self._partial_rollout,
                    )
                    # Under partial rollout the rollout path (below) deliberately
                    # skips vllm_lock so the trainer's broadcast_to_vllm refit can
                    # interleave via pause/resume. Eval must follow the same
                    # contract: holding the lock across the whole eval generation
                    # (~1hr at 32K) blocks the refit's acquire and wedges training
                    # (train_step never returns → no global_step advance).
                    if not self._partial_rollout:
                        ray.get(self.vllm_lock.acquire.remote())
                    try:
                        samples_list = self.samples_generator.generate_eval_samples(**eval_kwargs)
                    finally:
                        if not self._partial_rollout:
                            ray.get(self.vllm_lock.release.remote())
                    eval_metrics = compute_eval_metrics(self.eval_dataloader, samples_list, eval_n)
                    logger.info(f"Async evaluation completed: {eval_metrics}")
                    self.rollout_queue.put(("eval", global_step, eval_metrics), block=True)
                    continue

                if self.args.train.rollout_replay_dir:
                    # Debug: replay dumped rollout batches (train-only) to iterate on the
                    # training/refit path without regenerating. When the dumps run out, report
                    # exhaustion so the normal end-of-data path (below) stops the run cleanly.
                    replay_path = os.path.join(self.args.train.rollout_replay_dir, f"rollout_step{global_step}.pt")
                    rollout_metrics, prompts_consumed, generation_time = {}, 0, 0.0
                    is_exhausted = not os.path.exists(replay_path)
                    rollout_samples = None if is_exhausted else torch.load(replay_path, weights_only=False)
                    logger.info(
                        f"[rollout_replay] exhausted at {replay_path}"
                        if is_exhausted
                        else f"[rollout_replay] loaded {len(rollout_samples)} samples from {replay_path}"
                    )
                else:
                    if not self._partial_rollout:
                        ray.get(self.vllm_lock.acquire.remote())
                    try:
                        t0 = time.time()
                        train_kwargs = {
                            **self.generate_kwargs,
                            "rollout_kind": "train",
                            "policy_version": global_step,
                            # Strict on-policy generation drains exactly at the batch
                            # boundary while this lock prevents a physical refit.
                            # Async mode retains the streaming slow-tail behavior.
                            "policy_frozen": bool(self.args.train.force_on_policy),
                        }
                        rollout_samples, rollout_metrics, prompts_consumed, is_exhausted = (
                            self.samples_generator.generate_samples(**train_kwargs)
                        )
                        generation_time = time.time() - t0
                        total_consumed_prompts += prompts_consumed
                    finally:
                        if not self._partial_rollout:
                            ray.get(self.vllm_lock.release.remote())
                    if self.args.train.rollout_dump_dir and rollout_samples:
                        os.makedirs(self.args.train.rollout_dump_dir, exist_ok=True)
                        dump_path = os.path.join(self.args.train.rollout_dump_dir, f"rollout_step{global_step}.pt")
                        torch.save(rollout_samples, dump_path)
                        logger.info(f"[rollout_dump] wrote {len(rollout_samples)} samples to {dump_path}")

                if rollout_samples:
                    client_states = {
                        "episode": ep,
                        "total_consumed_prompts": total_consumed_prompts,
                        "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                        "rollout_generator_state_dict": self.samples_generator.state_dict(),
                    }
                    self.rollout_queue.put(
                        (rollout_samples, client_states, rollout_metrics, generation_time, vllm_idle_wait),
                        block=True,
                    )
                    if prompts_consumed:
                        pbar.update(prompts_consumed)
                else:
                    # Nothing enqueued => trainer will never consume this slot.
                    self.rollout_slots.put(global_step, block=True)

                if is_exhausted:
                    # For an exact full final batch, StatefulDataLoader only sets
                    # _iterator_finished on this following zero-sample probe. The
                    # optimizer batch is already queued, so carry the post-probe
                    # state as a distinct FIFO message instead of losing it.
                    terminal_client_states = {
                        "episode": ep,
                        "total_consumed_prompts": total_consumed_prompts,
                        "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                        "rollout_generator_state_dict": self.samples_generator.state_dict(),
                    }
                    break

            pbar.close()

        if terminal_client_states is not None:
            self.rollout_queue.put(make_terminal_payload(terminal_client_states), block=True)
        self.rollout_queue.put("done", block=True)


@ray.remote
class TrainingActor(BaseRLTrainer):
    def __init__(
        self,
        pretrain,
        strategy,
        actor_model_group,
        reference_model_group,
        vllm_engines,
        *,
        vllm_lock,
        rollout_queue,
        rollout_slots,
        critic_model_group=None,
    ):
        tokenizer = get_tokenizer(pretrain, None, "left", use_fast=not strategy.args.data.disable_fast_tokenizer)

        super().__init__(
            strategy,
            actor_model_group,
            reference_model_group,
            vllm_engines,
            tokenizer,
            critic_model_group=critic_model_group,
        )

        self.vllm_lock = vllm_lock
        self._prefix_caching_enabled = getattr(strategy.args.vllm, "enable_prefix_caching", False)
        self.rollout_queue = rollout_queue
        self.rollout_slots = rollout_slots

    def fit(self, global_step: int = 0) -> None:
        step_start_time = time.time()
        self._latest_client_states = {}
        self._last_checkpoint_client_states = None
        pending_payloads = deque()
        while True:
            # Time the block — in this async split, the trainer stuck here means the
            # actor side is sitting IDLE waiting for a rollout to be produced (e.g. during
            # a long eval, or if generation becomes the bottleneck). True actor-idle signal.
            _queue_wait_t0 = time.time()
            if pending_payloads:
                payload = pending_payloads.popleft()
                actor_idle_wait = 0.0
            else:
                payload = self.rollout_queue.get(block=True)
                actor_idle_wait = time.time() - _queue_wait_t0
            if payload == "done":
                break

            # A short final batch can already have saved this exact exhausted
            # state on the regular cadence. An exact-full final batch cannot:
            # its cadence save predates the StopIteration probe and therefore
            # differs at _iterator_finished, so overwrite it with the safe
            # terminal state (or create it when cadence was not due).
            terminal_client_states = handle_terminal_payload(
                payload,
                global_step=global_step,
                best_eval_metric_key=self.best_eval_metric_key,
                best_eval_metric_value=self.best_eval_metric_value,
                last_checkpoint_client_states=self._last_checkpoint_client_states,
                save_terminal_checkpoint=self.save_terminal_checkpoint,
            )
            if terminal_client_states is not None:
                self._latest_client_states = terminal_client_states
                self._last_checkpoint_client_states = dict(terminal_client_states)
                continue

            if payload[0] == "eval":
                _, eval_step, eval_metrics = payload
                # Periodic checkpoint barriers publish a dedicated slot for
                # eval.  Consuming that slot must not mint another token or the
                # producer can run progressively farther ahead after each eval.
                if not checkpoint_aligned_eval_due(self.args, eval_step):
                    self.rollout_slots.put(global_step, block=True)
                eval_metrics = dict(eval_metrics)
                current_policy_versions = (
                    ray.get([engine.get_weight_version.remote() for engine in self.vllm_engines])
                    if self.vllm_engines
                    else []
                )
                if current_policy_versions:
                    eval_metrics["eval_checkpoint_policy_version_min"] = float(min(current_policy_versions))
                    eval_metrics["eval_checkpoint_policy_version_max"] = float(max(current_policy_versions))
                eval_metrics["eval_checkpoint_policy_matches"] = float(
                    eval_matches_current_policy(eval_metrics, current_policy_versions)
                )
                logger.info(f"Eval at step {eval_step}: {eval_metrics}")
                if self.wandb_logger:
                    self.wandb_logger.log_eval(eval_step, eval_metrics)
                if self.tensorboard_logger:
                    self.tensorboard_logger.log_eval(eval_step, eval_metrics)
                client_states = dict(self._latest_client_states)
                client_states["global_step"] = global_step
                self.save_best_checkpoint(eval_metrics, eval_step, client_states)
                step_start_time = time.time()
                continue

            rollout_samples, client_states, rollout_metrics, generation_time, vllm_idle_wait = payload

            # Batch consumed => free one token to allow generator to produce next batch.
            # --train.force_sync_mode defers this until AFTER train_step (which updates the
            # actor and refits vLLM), so the generator waits for the fresh weights before
            # producing the next batch -> the next rollout is generated with the same
            # weights the trainer recomputes it under (strictly on-policy). This removes the
            # 1-step-stale rollout that inflates vllm_kl on routing-sensitive MoE
            # checkpoints, at the cost of the generate/train overlap. Default off.
            force_sync = getattr(self.args.train, "force_sync_mode", False)
            force_on_policy = bool(self.args.train.force_on_policy)
            if not force_sync and not force_on_policy:
                # Async mode keeps its existing overlap: consuming a batch frees
                # the producer immediately, carrying the pre-update logical step.
                self.rollout_slots.put(global_step, block=True)
            if force_on_policy:
                current_policy_versions = ray.get(
                    [engine.get_weight_version.remote() for engine in (self.vllm_engines or [])]
                )
                validate_force_on_policy_batch(rollout_samples, current_policy_versions)

            status, global_step = self.train_step(rollout_samples, global_step)
            if force_sync or force_on_policy:
                # train_step returns only after the optimizer step and vLLM
                # broadcast. Release the sole slot with the updated logical step,
                # so the next batch starts from the newly loaded weights.
                self.rollout_slots.put(global_step, block=True)
            status["timing/generation"] = generation_time
            # Async idle accounting (the real "which side is wasted" signal): in the split
            # actor/vLLM topology these directly attribute the idle the reaper sees.
            #   vllm_idle_wait  = generator blocked for a train slot  -> vLLM GPUs idle
            #   actor_idle_wait = trainer blocked for a rollout       -> actor GPUs idle
            # gen<<train => vllm_idle_wait large (vLLM wasted); long eval => actor_idle_wait large.
            status["timing/vllm_idle_wait"] = vllm_idle_wait
            status["timing/actor_idle_wait"] = actor_idle_wait
            status["timing/step_total"] = time.time() - step_start_time
            step_start_time = time.time()

            # rollout/dropped/<reason> counts + dynamic_filtering_pass_rate (when enabled).
            status.update(rollout_metrics)

            log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
            logger.info(f"Global step {global_step}: {log_status}")

            client_states.update({"global_step": global_step})
            self._latest_client_states = client_states
            if self.save_logs_and_checkpoints(global_step, status, client_states):
                self._last_checkpoint_client_states = dict(client_states)

            if checkpoint_aligned_eval_due(self.args, global_step):
                # Async mode normally releases a pre-update logical step before
                # training. Publish this exact post-update step as an additional
                # slot so the generator starts the scheduled eval, then stop the
                # optimizer while that frozen-policy eval is in flight. Any
                # already-prefetched train batch stays buffered until the eval
                # result has been logged and (if best) checkpointed.
                logger.info("Waiting for checkpoint-aligned evaluation at step %s", global_step)
                self.rollout_slots.put(global_step, block=True)
                if not buffer_until_eval_payload(self.rollout_queue, pending_payloads, global_step):
                    logger.info(
                        "Skipping inline evaluation at terminal step %s because the rollout producer is exhausted",
                        global_step,
                    )

        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    def broadcast_to_vllm(self):
        # Keep new generation calls out while existing requests are paused and
        # refitted. Report lock wait separately from the weight transfer. The lock
        # release sits in finally so a failed refit (NCCL errors do happen mid-run)
        # crashes cleanly instead of leaving vllm_lock held, which would silently
        # hang the trainer and eval (both acquire it).
        _t0 = time.time()
        ray.get(self.vllm_lock.acquire.remote())
        self._broadcast_lock_wait_s = time.time() - _t0
        _t0 = time.time()
        try:
            batch_vllm_engine_call(self.vllm_engines, "pause_generation")
            super().broadcast_to_vllm()
            if self._prefix_caching_enabled:
                batch_vllm_engine_call(self.vllm_engines, "reset_prefix_cache")
            batch_vllm_engine_call(self.vllm_engines, "resume_generation")
        finally:
            ray.get(self.vllm_lock.release.remote())
            self._broadcast_transfer_s = time.time() - _t0


def reseed_rollout_slots(rollout_slots, queue_size: int, global_step: int) -> None:
    """Replace constructor-time slot values before either worker starts.

    On checkpoint resume the queues still contain zeros from ``RLTrainer``
    construction. Replacing every token ensures eval-only and the first resumed
    rollout carry the restored step rather than being mislabeled as policy 0.
    """
    for _ in range(queue_size):
        rollout_slots.get(block=True)
    for _ in range(queue_size):
        rollout_slots.put(int(global_step), block=True)


@ray.remote
class RLTrainer:
    """Async-only RL controller."""

    def __init__(
        self,
        pretrain: str,
        strategy: FsdpStrategy,
        actor_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines,
        critic_model_group: RayActorGroup = None,
        router_url: str = None,
        **generate_kwargs,
    ) -> None:
        if strategy.args.eval.steps == -1:
            strategy.args.eval.steps = float("inf")
        if strategy.args.ckpt.save_steps == -1:
            strategy.args.ckpt.save_steps = float("inf")

        queue_size = getattr(strategy.args.train, "async_queue_size", 1)
        validate_force_on_policy_runtime_config(strategy.args.train)
        if queue_size <= 0:
            raise ValueError(f"async_queue_size must be positive, got {queue_size}")
        logger.info(f"async_queue_size={queue_size}")
        self._queue_size = queue_size
        self._eval_only = bool(getattr(strategy.args.eval, "eval_only", False))

        self.rollout_queue = Queue(maxsize=queue_size)
        # Slot tokens are flow-control signals, not data.  The rollout payload
        # queue remains bounded and provides the actual backpressure.  Keeping
        # slots unbounded prevents an exact-full terminal probe from deadlocking
        # with the trainer while both actors try to return their final token or
        # terminal payload.
        self.rollout_slots = Queue()
        for _ in range(queue_size):
            self.rollout_slots.put(0, block=True)

        vllm_lock = VLLMLock.remote()

        self.generator_actor = GenerateSamplesActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            vllm_lock=vllm_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
            router_url=router_url,
            version_source=vllm_engines[0] if vllm_engines else None,
            **generate_kwargs,
        )

        self.trainer_actor = TrainingActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            actor_model_group=actor_model_group,
            reference_model_group=reference_model_group,
            vllm_engines=vllm_engines,
            vllm_lock=vllm_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
            critic_model_group=critic_model_group,
        )

    def fit(self) -> None:
        checkpoint_states = ray.get(self.trainer_actor.load_checkpoint_states_or_default.remote())
        ray.get(self.trainer_actor.restore_best_metric_tracker.remote(checkpoint_states))

        # .get with defaults: an interrupted save can leave model/ without extra_state.pt, so
        # load_ckpt returns states={} (weights load, scalars empty) — resume at 0, not KeyError.
        start_episode, resume_dataloader_state = resolve_rollout_resume_position(checkpoint_states)
        global_step = checkpoint_states.get("global_step", 0)
        total_consumed_prompts = checkpoint_states.get("total_consumed_prompts", 0)
        if global_step > 0:
            resume_refs = [
                self.trainer_actor.broadcast_to_vllm.remote(),
            ]
            if self._eval_only:
                logger.info("eval-only resume: skipping training dataloader state restoration")
            elif resume_dataloader_state is not None:
                resume_refs.append(
                    self.generator_actor.load_dataloader_state_dict.remote(
                        resume_dataloader_state,
                        checkpoint_states.get("rollout_generator_state_dict"),
                    )
                )
            else:
                logger.info(
                    "checkpoint dataloader finished episode %s; resuming at episode %s with a fresh iterator",
                    checkpoint_states.get("episode", 0),
                    start_episode,
                )
            ray.get(resume_refs)

        reseed_rollout_slots(self.rollout_slots, self._queue_size, global_step)

        ray.get(
            [
                self.generator_actor.fit.remote(episode=start_episode, total_consumed_prompts=total_consumed_prompts),
                self.trainer_actor.fit.remote(global_step=global_step),
            ]
        )

    def get_max_steps(self):
        return ray.get(self.generator_actor.get_max_steps.remote())
