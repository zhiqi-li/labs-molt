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

"""Task-shared, suite-stratified progressive difficulty sampler for RL rollouts.

Groups dataset rows that share the same underlying task across different initial states / seeds
(e.g. ``label.episode.native.cell`` in LIBERO, or ``task_id`` / ``cell`` in generic datasets),
tracks each task's online EMA pass rate from valid pre-filter rollout groups, and progressively
shifts sampling probability from mastered easy tasks toward harder learnable frontier tasks.

Key invariants:
- **Suite quota preservation** (``suite_stratified=True``): preserves the original dataset's
  row-weighted suite/datasource distribution (e.g. 50% LIBERO-90 / 50% LIBERO-40) while
  adapting task sampling weights *within* each suite.
- **Anti-forgetting floor** (``replay_floor``): mastered tasks (pass rate -> 1) keep a minimum
  weight so the policy does not regress on solved skills.
- **Anti-deadlock cooling + explicit probe**: tasks that repeatedly score 0/N across valid
  groups (``num_rollouts >= min_rollouts``) decay exponentially instead of hijacking zero-gradient
  rollouts; cooled tasks past ``probe_interval_draws`` are explicitly probed once so skill
  transfer can unlock them without relying on low-probability random draws.
- **Resume-safe episode budget**: mid-episode checkpoint restores preserve ``draws_in_episode``
  and support offset-aligned partial warm-resume skipping without duplicating completed prompts.
- **Within-task seed rotation & inflight deduplication**: draws cycle through shuffled seeds per
  task and divide weight by ``(1 + inflight_ct)`` across concurrent ``partial_rollout`` slots.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def extract_task_key(label: Any, datasource: str = "default", prompt: Any = "") -> str:
    """Derive a stable shared task identifier across seeds/initial states of the same task."""
    parsed = label
    if isinstance(label, str):
        stripped = label.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except ValueError:
                parsed = None
    if isinstance(parsed, dict):
        episode = parsed.get("episode")
        if isinstance(episode, dict):
            native = episode.get("native")
            if isinstance(native, dict) and native.get("cell"):
                return str(native["cell"])
            if episode.get("cell"):
                return str(episode["cell"])
        for key in ("cell", "task_id", "task", "task_name", "env_id"):
            val = parsed.get(key)
            if isinstance(val, str) and val:
                return val
    return f"{datasource}::{prompt}"


@dataclass(frozen=True)
class AdaptiveDifficultyConfig:
    """Hyperparameters for progressive task-shared difficulty sampling."""

    enabled: bool = False
    max_alpha: float = 2.0
    ramp_steps: int = 40
    ema_gamma: float = 0.35
    replay_floor: float = 0.05
    frontier_beta: float = 1.0
    frontier_delta: float = 0.05
    zero_cooldown_tau: float = 3.0
    probe_interval_draws: int = 250
    prior_pass_rate: float = 0.5
    suite_stratified: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_alpha < 0:
            raise ValueError(f"max_alpha must be >= 0, got {self.max_alpha}")
        if self.ramp_steps < 0:
            raise ValueError(f"ramp_steps must be >= 0, got {self.ramp_steps}")
        if not 0.0 < self.ema_gamma <= 1.0:
            raise ValueError(f"ema_gamma must be in (0, 1], got {self.ema_gamma}")
        if self.replay_floor < 0:
            raise ValueError(f"replay_floor must be >= 0, got {self.replay_floor}")
        if self.zero_cooldown_tau <= 0:
            raise ValueError(f"zero_cooldown_tau must be > 0, got {self.zero_cooldown_tau}")

    @classmethod
    def from_args(cls, args: Any) -> "AdaptiveDifficultyConfig":
        rollout = getattr(args, "rollout", None)
        train = getattr(args, "train", None)
        cli_enable = bool(getattr(rollout, "adaptive_difficulty_enable", False))
        enabled = cli_enable or _env_enabled("MOLT_ADAPTIVE_DIFFICULTY", False)
        return cls(
            enabled=enabled,
            max_alpha=float(getattr(rollout, "adaptive_difficulty_alpha", 2.0)),
            ramp_steps=int(getattr(rollout, "adaptive_difficulty_ramp_steps", 40)),
            ema_gamma=float(getattr(rollout, "adaptive_difficulty_ema", 0.35)),
            replay_floor=float(getattr(rollout, "adaptive_difficulty_replay_floor", 0.05)),
            seed=int(getattr(train, "seed", 42)),
        )


@dataclass
class _TaskState:
    indices: List[int]
    suite: str
    ema_pass_rate: float
    update_ct: int = 0
    zero_streak: int = 0
    last_draw_seq: int = -10**9
    inflight_ct: int = 0
    seed_order: List[int] = field(default_factory=list)
    seed_cursor: int = 0

    def next_index(self, rng: random.Random) -> int:
        if not self.seed_order or self.seed_cursor >= len(self.seed_order):
            self.seed_order = list(self.indices)
            rng.shuffle(self.seed_order)
            self.seed_cursor = 0
        idx = self.seed_order[self.seed_cursor]
        self.seed_cursor += 1
        return idx


class AdaptiveDifficultySampler:
    """Online task-shared, suite-stratified progressive difficulty sampler."""

    def __init__(
        self,
        task_to_indices: Dict[str, Sequence[int]],
        config: AdaptiveDifficultyConfig,
        *,
        task_to_suite: Optional[Dict[str, str]] = None,
        epoch_budget: Optional[int] = None,
    ) -> None:
        if not task_to_indices:
            raise ValueError("AdaptiveDifficultySampler requires at least one task")
        self.config = config
        self._rng = random.Random(config.seed)
        self._task_ids: List[str] = sorted(task_to_indices.keys())
        self._tasks: Dict[str, _TaskState] = {}
        self._suite_tasks: Dict[str, List[str]] = {}
        suite_rows: Dict[str, int] = {}
        total_rows = 0
        for tid in self._task_ids:
            idxs = [int(i) for i in task_to_indices[tid]]
            if not idxs:
                continue
            suite = str((task_to_suite or {}).get(tid, "default"))
            self._tasks[tid] = _TaskState(indices=idxs, suite=suite, ema_pass_rate=config.prior_pass_rate)
            self._suite_tasks.setdefault(suite, []).append(tid)
            suite_rows[suite] = suite_rows.get(suite, 0) + len(idxs)
            total_rows += len(idxs)
        self._task_ids = [tid for tid in self._task_ids if tid in self._tasks]
        if not self._task_ids:
            raise ValueError("AdaptiveDifficultySampler requires at least one non-empty task")

        self._suites: List[str] = sorted(self._suite_tasks.keys())
        self._suite_weights: List[float] = [float(suite_rows[s]) for s in self._suites]

        self.epoch_budget = int(epoch_budget if epoch_budget is not None else total_rows)
        self.step_ct: int = 0
        self.draw_seq: int = 0
        self.draws_in_episode: int = 0
        self.episode_active: bool = False
        self._batch_sampled_ps: List[float] = []

    @classmethod
    def from_prompt_dataset(
        cls,
        prompt_dataset: Any,
        config: AdaptiveDifficultyConfig,
    ) -> "AdaptiveDifficultySampler":
        """Build task buckets & suite quotas from a ``PromptDataset`` without rendering images/templates."""
        raw_ds = getattr(prompt_dataset, "dataset", None)
        if raw_ds is None:
            raise ValueError("PromptDataset must expose its underlying .dataset for adaptive sampling")
        n_rows = len(raw_ds)
        label_key = getattr(prompt_dataset, "label_key", None)
        input_key = getattr(prompt_dataset, "input_key", None)
        col_names = set(getattr(raw_ds, "column_names", ()) or ())

        labels = raw_ds[label_key] if (label_key and label_key in col_names) else [None] * n_rows
        datasources = raw_ds["datasource"] if "datasource" in col_names else ["default"] * n_rows
        prompts = raw_ds[input_key] if (input_key and input_key in col_names) else [""] * n_rows

        buckets: Dict[str, List[int]] = {}
        suites: Dict[str, str] = {}
        for idx in range(n_rows):
            ds_name = str(datasources[idx])
            tid = extract_task_key(labels[idx], ds_name, prompts[idx])
            buckets.setdefault(tid, []).append(idx)
            suites.setdefault(tid, ds_name)
        return cls(buckets, config, task_to_suite=suites, epoch_budget=n_rows)

    @property
    def num_tasks(self) -> int:
        return len(self._task_ids)

    @property
    def exhausted(self) -> bool:
        return self.draws_in_episode >= self.epoch_budget

    @property
    def current_alpha(self) -> float:
        if self.config.ramp_steps <= 0:
            return self.config.max_alpha
        progress = min(1.0, max(0.0, self.step_ct / float(self.config.ramp_steps)))
        return self.config.max_alpha * progress

    def begin_round(self) -> None:
        """Prepare the sampler for a round of ``generate_samples()``.

        Resets ``draws_in_episode`` ONLY when starting a genuinely fresh episode (not mid-episode
        after a checkpoint resume where ``load_state_dict`` restored ``episode_active=True``).
        """
        if not self.episode_active or self.exhausted:
            self.draws_in_episode = 0
            self.episode_active = True
        self.reset_inflight()

    def start_episode(self) -> None:
        """Explicitly start a brand-new episode (resets episode draw budget, keeps learned EMAs)."""
        self.draws_in_episode = 0
        self.episode_active = True
        self.reset_inflight()

    def reset_inflight(self) -> None:
        for st in self._tasks.values():
            st.inflight_ct = 0

    def advance_step(self) -> None:
        """Mark one training batch complete to advance the progressive difficulty ramp."""
        self.step_ct += 1

    def _is_probe_due(self, st: _TaskState) -> bool:
        return (
            st.zero_streak >= 3
            and st.inflight_ct == 0
            and (self.draw_seq - st.last_draw_seq) >= self.config.probe_interval_draws
        )

    def task_weight(self, task_id: str, *, include_inflight: bool = True) -> float:
        st = self._tasks[task_id]
        cfg = self.config
        alpha = self.current_alpha
        p = min(1.0, max(0.0, st.ema_pass_rate))

        # Unimodal frontier score: with alpha=2.0, beta=1.0, delta=0.05 this peaks at p=0.30
        # (~94% mixed-group probability for n=8 rollouts) while suppressing mastered tasks (p -> 1).
        frontier = ((1.0 - p) ** alpha) * ((p + cfg.frontier_delta) ** cfg.frontier_beta)

        if st.zero_streak > 0 and not self._is_probe_due(st):
            cooldown = math.exp(-st.zero_streak / cfg.zero_cooldown_tau)
        else:
            cooldown = 1.0

        weight = cfg.replay_floor + frontier * cooldown
        if include_inflight and st.inflight_ct > 0:
            weight /= 1.0 + st.inflight_ct
        return max(weight, 1e-9)

    def sampling_probabilities(self, *, include_inflight: bool = True) -> Dict[str, float]:
        if not self.config.suite_stratified or len(self._suites) <= 1:
            weights = {tid: self.task_weight(tid, include_inflight=include_inflight) for tid in self._task_ids}
            total = sum(weights.values())
            return {tid: w / total for tid, w in weights.items()}

        suite_total = sum(self._suite_weights)
        probs: Dict[str, float] = {}
        for suite, s_weight in zip(self._suites, self._suite_weights):
            s_prob = s_weight / suite_total
            tids = self._suite_tasks[suite]
            t_weights = [self.task_weight(tid, include_inflight=include_inflight) for tid in tids]
            t_sum = sum(t_weights)
            for tid, tw in zip(tids, t_weights):
                probs[tid] = s_prob * (tw / t_sum)
        return probs

    def _select_one_task(self) -> str:
        # First honor explicit probe budget for cooled tasks past probe_interval_draws.
        due = [tid for tid in self._task_ids if self._is_probe_due(self._tasks[tid])]
        if due:
            # Pick the task waiting longest since its last draw (deterministic fairness).
            return min(due, key=lambda tid: self._tasks[tid].last_draw_seq)

        if self.config.suite_stratified and len(self._suites) > 1:
            suite = self._rng.choices(self._suites, weights=self._suite_weights, k=1)[0]
            candidate_tids = self._suite_tasks[suite]
        else:
            candidate_tids = self._task_ids
        weights = [self.task_weight(tid, include_inflight=True) for tid in candidate_tids]
        return self._rng.choices(candidate_tids, weights=weights, k=1)[0]

    def _draw_one(self, *, mark_inflight: bool) -> Tuple[int, str]:
        tid = self._select_one_task()
        st = self._tasks[tid]
        idx = st.next_index(self._rng)
        if mark_inflight:
            st.inflight_ct += 1
            self._batch_sampled_ps.append(st.ema_pass_rate)
        st.last_draw_seq = self.draw_seq
        self.draw_seq += 1
        self.draws_in_episode += 1
        if self.exhausted:
            self.episode_active = False
        return idx, tid

    def draw(self, num_prompts: int) -> Tuple[List[Tuple[int, str]], bool]:
        """Sample up to ``num_prompts`` ``(dataset_index, task_id)`` pairs respecting epoch budget."""
        self.episode_active = True
        remaining = max(0, self.epoch_budget - self.draws_in_episode)
        count = min(num_prompts, remaining)
        drawn = [self._draw_one(mark_inflight=True) for _ in range(count)]
        return drawn, self.exhausted

    def draw_unskipped(
        self,
        num_prompts: int,
        start_offset: int,
        skip_offsets: set[int],
    ) -> Tuple[List[Tuple[int, str, int]], int, bool]:
        """Offset-aligned draw mirroring ``_collect_unskipped_prompts`` for partial warm-resume.

        Offsets present in ``skip_offsets`` still advance the sampler's draw/RNG/seed sequence
        without dispatching or incrementing ``inflight_ct``, preventing duplicate prompts on resume.
        """
        self.episode_active = True
        collected: List[Tuple[int, str, int]] = []
        drawn_total = 0
        while len(collected) < num_prompts and not self.exhausted:
            offset = start_offset + drawn_total
            drawn_total += 1
            if offset in skip_offsets:
                self._draw_one(mark_inflight=False)
                continue
            idx, tid = self._draw_one(mark_inflight=True)
            collected.append((idx, tid, offset))
        return collected, drawn_total, self.exhausted

    def consume_contiguous_skipped(self, start_offset: int, skip_offsets: set[int]) -> int:
        """Advance past trailing skipped offsets at batch end so the next batch does not replay them."""
        consumed = 0
        while not self.exhausted and (start_offset + consumed) in skip_offsets:
            self._draw_one(mark_inflight=False)
            consumed += 1
        return consumed

    def update(
        self,
        task_id: Optional[str],
        mean_score: Optional[float],
        *,
        min_score: float = 0.0,
        num_rollouts: int = 1,
        min_rollouts: int = 1,
    ) -> None:
        """Record a finished group's pre-filter mean score and release its inflight slot.

        Always releases ``inflight_ct``. Only updates EMA / ``zero_streak`` when the group met
        ``min_rollouts`` valid scored rollouts (so infra-damaged 0/1 groups never cool down a task).
        """
        if task_id is None or task_id not in self._tasks:
            return
        st = self._tasks[task_id]
        st.inflight_ct = max(0, st.inflight_ct - 1)
        if mean_score is None or not math.isfinite(mean_score) or num_rollouts < min_rollouts:
            return

        clamped = min(1.0, max(0.0, float(mean_score)))
        gamma = max(self.config.ema_gamma, 0.5) if st.update_ct == 0 else self.config.ema_gamma
        st.ema_pass_rate = (1.0 - gamma) * st.ema_pass_rate + gamma * clamped
        st.update_ct += 1
        if clamped <= min_score:
            st.zero_streak += 1
        else:
            st.zero_streak = 0

    def pop_batch_metrics(self) -> Dict[str, float]:
        """Return aggregate sampler telemetry for the completed step and reset batch accumulators."""
        ps = [st.ema_pass_rate for st in self._tasks.values()]
        metrics: Dict[str, float] = {
            "rollout/sampler_alpha": float(self.current_alpha),
            "rollout/sampler_mean_p": sum(ps) / len(ps),
            "rollout/sampler_num_easy": float(sum(1 for st in self._tasks.values() if st.ema_pass_rate >= 0.85)),
            "rollout/sampler_num_frontier": float(
                sum(1 for st in self._tasks.values() if 0.05 < st.ema_pass_rate < 0.85)
            ),
            "rollout/sampler_num_cooled": float(sum(1 for st in self._tasks.values() if st.zero_streak >= 3)),
            "rollout/sampler_visited_tasks": float(sum(1 for st in self._tasks.values() if st.update_ct > 0)),
        }
        if self._batch_sampled_ps:
            metrics["rollout/sampler_sampled_p"] = sum(self._batch_sampled_ps) / len(self._batch_sampled_ps)
            self._batch_sampled_ps.clear()
        return metrics

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "step_ct": self.step_ct,
            "draw_seq": self.draw_seq,
            "draws_in_episode": self.draws_in_episode,
            "episode_active": self.episode_active,
            "rng_state": self._rng.getstate(),
            "tasks": {
                tid: {
                    "ema_pass_rate": st.ema_pass_rate,
                    "update_ct": st.update_ct,
                    "zero_streak": st.zero_streak,
                    "last_draw_seq": st.last_draw_seq,
                    "seed_order": list(st.seed_order),
                    "seed_cursor": st.seed_cursor,
                }
                for tid, st in self._tasks.items()
            },
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state or not isinstance(state, dict):
            return
        self.step_ct = int(state.get("step_ct", 0))
        self.draw_seq = int(state.get("draw_seq", 0))
        self.draws_in_episode = int(state.get("draws_in_episode", 0))
        self.episode_active = bool(state.get("episode_active", 0 < self.draws_in_episode < self.epoch_budget))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                self._rng.setstate(rng_state)
            except Exception:
                pass
        saved_tasks = state.get("tasks") or {}
        for tid, st in self._tasks.items():
            st.inflight_ct = 0
            rec = saved_tasks.get(tid)
            if not isinstance(rec, dict):
                continue
            st.ema_pass_rate = float(rec.get("ema_pass_rate", st.ema_pass_rate))
            st.update_ct = int(rec.get("update_ct", 0))
            st.zero_streak = int(rec.get("zero_streak", 0))
            st.last_draw_seq = int(rec.get("last_draw_seq", -10**9))
            saved_order = rec.get("seed_order")
            if isinstance(saved_order, list) and len(saved_order) == len(st.indices) and set(saved_order) == set(st.indices):
                st.seed_order = [int(i) for i in saved_order]
                st.seed_cursor = min(max(0, int(rec.get("seed_cursor", 0))), len(st.seed_order))
