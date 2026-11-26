# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit and regression tests for AdaptiveDifficultySampler."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from molt.trainer.rollout.adaptive_sampler import (
    AdaptiveDifficultyConfig,
    AdaptiveDifficultySampler,
    extract_task_key,
)


class _FakeArrowDataset:
    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self._rows]
        return self._rows[key]


class _FakePromptDataset:
    def __init__(self, rows: list[dict]):
        self.dataset = _FakeArrowDataset(rows)
        self.label_key = "label"
        self.input_key = "prompt"


class TestAdaptiveDifficultySampler(unittest.TestCase):
    def test_extract_task_key_shares_across_libero_seeds(self):
        l0 = json.dumps({"episode": {"native": {"cell": "libero_90/KITCHEN_SCENE1_open_drawer", "init_index": 0, "seed": 0}}})
        l17 = json.dumps({"episode": {"native": {"cell": "libero_90/KITCHEN_SCENE1_open_drawer", "init_index": 17, "seed": 17}}})
        self.assertEqual(extract_task_key(l0, "libero_90", "p0"), "libero_90/KITCHEN_SCENE1_open_drawer")
        self.assertEqual(extract_task_key(l0, "libero_90", "p0"), extract_task_key(l17, "libero_90", "p17"))
        self.assertEqual(extract_task_key({"task_id": "math/algebra_1"}, "math", "x"), "math/algebra_1")
        self.assertEqual(extract_task_key("plain-answer", "gsm8k", "2+2=?"), "gsm8k::2+2=?")

    def test_from_prompt_dataset_groups_seeds_and_rotates_without_repeats(self):
        rows = []
        for cell in ("suite/task_a", "suite/task_b"):
            for seed in range(5):
                rows.append(
                    {
                        "datasource": "libero",
                        "prompt": f"{cell}_s{seed}",
                        "label": json.dumps({"episode": {"native": {"cell": cell, "init_index": seed, "seed": seed}}}),
                    }
                )
        ds = _FakePromptDataset(rows)
        cfg = AdaptiveDifficultyConfig(enabled=True, max_alpha=0.0, ramp_steps=1, seed=7)
        sampler = AdaptiveDifficultySampler.from_prompt_dataset(ds, cfg)
        self.assertEqual(sampler.num_tasks, 2)
        self.assertEqual(sampler.epoch_budget, 10)

        seen_a = []
        for _ in range(10):
            pairs, _ = sampler.draw(1)
            idx, tid = pairs[0]
            sampler.update(tid, 0.5)
            if tid == "suite/task_a":
                seen_a.append(idx)
        self.assertEqual(len(set(seen_a[:5])), len(seen_a[:5]))
        self.assertTrue(all(0 <= i < 5 for i in seen_a))

    def test_progressive_ramp_peaks_at_learnable_frontier_and_suppresses_easy(self):
        cfg = AdaptiveDifficultyConfig(enabled=True, max_alpha=2.0, ramp_steps=20, ema_gamma=0.5, replay_floor=0.05)
        sampler = AdaptiveDifficultySampler(
            {"easy": [0, 1], "medium": [2, 3], "frontier": [4, 5]},
            cfg,
        )
        for _ in range(12):
            sampler.update("easy", 1.0)
            sampler.update("medium", 0.55)
            sampler.update("frontier", 0.30)

        self.assertAlmostEqual(sampler.current_alpha, 0.0)
        p_early = sampler.sampling_probabilities(include_inflight=False)

        for _ in range(25):
            sampler.advance_step()
        self.assertAlmostEqual(sampler.current_alpha, 2.0)
        p_late = sampler.sampling_probabilities(include_inflight=False)

        self.assertGreater(p_late["frontier"], p_late["medium"])
        self.assertGreater(p_late["medium"], p_late["easy"] * 2.5)
        self.assertGreater(p_late["frontier"], p_late["easy"] * 3.5)
        self.assertGreater(p_late["easy"], 0.05)
        self.assertLess(p_late["easy"], p_early["easy"])

    def test_p2_suite_stratified_preserves_original_suite_quotas(self):
        # 90 L90 tasks x 16 seeds = 1440 rows (50%) vs 40 L40 tasks x 36 seeds = 1440 rows (50%).
        rows = []
        for i in range(90):
            for s in range(16):
                rows.append({"datasource": "libero/train/libero_90", "prompt": f"p90_{i}_{s}", "label": {"cell": f"l90/t{i}"}})
        for suite_i, suite_name in enumerate(("libero_10", "libero_spatial", "libero_goal", "libero_object")):
            for j in range(10):
                for s in range(36):
                    rows.append({"datasource": f"libero/train/{suite_name}", "prompt": f"p40_{suite_i}_{j}_{s}", "label": {"cell": f"{suite_name}/t{j}"}})
        ds = _FakePromptDataset(rows)
        cfg = AdaptiveDifficultyConfig(enabled=True, max_alpha=2.0, ramp_steps=1, suite_stratified=True)
        sampler = AdaptiveDifficultySampler.from_prompt_dataset(ds, cfg)
        sampler.advance_step()

        # Even after updating difficulty asymmetrically, suite quotas must remain exact 50/50!
        for i in range(45):
            for _ in range(5):
                sampler.update(f"l90/t{i}", 1.0)
        probs = sampler.sampling_probabilities(include_inflight=False)
        l90_mass = sum(p for tid, p in probs.items() if tid.startswith("l90/"))
        l40_mass = sum(p for tid, p in probs.items() if not tid.startswith("l90/"))
        self.assertAlmostEqual(l90_mass, 0.5, places=6)
        self.assertAlmostEqual(l40_mass, 0.5, places=6)
        # Within L90, hard/unvisited tasks still outrank mastered ones.
        self.assertGreater(probs["l90/t89"], probs["l90/t0"] * 2.5)

    def test_p2_zero_streak_ignores_incomplete_groups_and_explicit_probe_fires(self):
        cfg = AdaptiveDifficultyConfig(
            enabled=True,
            max_alpha=2.0,
            ramp_steps=1,
            ema_gamma=0.5,
            zero_cooldown_tau=2.0,
            probe_interval_draws=20,
        )
        sampler = AdaptiveDifficultySampler({"learnable": [0], "dead_wall": [1]}, cfg)
        sampler.advance_step()

        # Infra-damaged 0/1 group (num_rollouts=1 < min_rollouts=6) must NOT cool or mutate EMA!
        sampler._tasks["dead_wall"].inflight_ct = 1
        ema0 = sampler._tasks["dead_wall"].ema_pass_rate
        sampler.update("dead_wall", 0.0, num_rollouts=1, min_rollouts=6)
        self.assertEqual(sampler._tasks["dead_wall"].inflight_ct, 0)
        self.assertEqual(sampler._tasks["dead_wall"].zero_streak, 0)
        self.assertEqual(sampler._tasks["dead_wall"].ema_pass_rate, ema0)

        # Valid 0/8 groups cool down dead_wall.
        for _ in range(6):
            sampler.update("learnable", 0.30, num_rollouts=8, min_rollouts=6)
            sampler.update("dead_wall", 0.0, num_rollouts=8, min_rollouts=6)
        sampler._tasks["dead_wall"].last_draw_seq = sampler.draw_seq
        self.assertLess(
            sampler.task_weight("dead_wall", include_inflight=False),
            sampler.task_weight("learnable", include_inflight=False) * 0.6,
        )

        # After probe_interval_draws elapse, explicit probe deterministically picks dead_wall next!
        sampler.draw_seq += cfg.probe_interval_draws
        pairs, _ = sampler.draw(1)
        self.assertEqual(pairs[0][1], "dead_wall")

        # One valid group with score > 0 resets zero_streak immediately.
        sampler.update("dead_wall", 0.25, num_rollouts=8, min_rollouts=6)
        self.assertEqual(sampler._tasks["dead_wall"].zero_streak, 0)

    def test_p1_resume_preserves_remaining_episode_budget(self):
        cfg = AdaptiveDifficultyConfig(enabled=True, max_alpha=2.0, ramp_steps=10, seed=123)
        s1 = AdaptiveDifficultySampler({"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]}, cfg, epoch_budget=8)
        s1.begin_round()
        pairs, exhausted = s1.draw(6)
        self.assertEqual(len(pairs), 6)
        self.assertFalse(exhausted)
        self.assertEqual(s1.draws_in_episode, 6)
        s1.update(pairs[0][1], 1.0, num_rollouts=8, min_rollouts=6)
        s1.advance_step()
        saved = s1.state_dict()

        # Simulate checkpoint resume: load_state_dict followed by begin_round() on first generate_samples().
        s2 = AdaptiveDifficultySampler({"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]}, cfg, epoch_budget=8)
        s2.load_state_dict(saved)
        s2.begin_round()
        self.assertEqual(s2.draws_in_episode, 6)
        self.assertTrue(all(st.inflight_ct == 0 for st in s2._tasks.values()))

        # Only 2 draws remain in this episode before exhausting!
        rem, exhausted2 = s2.draw(5)
        self.assertEqual(len(rem), 2)
        self.assertTrue(exhausted2)
        self.assertEqual(s2.draws_in_episode, 8)

        # Next episode's begin_round() resets draws_in_episode to 0 while preserving learned EMA/update_ct.
        s2.begin_round()
        self.assertEqual(s2.draws_in_episode, 0)
        self.assertFalse(s2.exhausted)
        self.assertEqual(s2._tasks[pairs[0][1]].update_ct, 1)

    def test_p1_partial_warm_resume_skips_completed_offsets_without_duplicates(self):
        cfg = AdaptiveDifficultyConfig(enabled=True, max_alpha=2.0, ramp_steps=10, seed=55)
        # Baseline uninterrupted draw of 6 prompts.
        ref = AdaptiveDifficultySampler({"a": [0, 1, 2], "b": [3, 4, 5]}, cfg, epoch_budget=6)
        ref.begin_round()
        # Simulate non-inflight sequence equivalence when offsets 0 and 2 were already completed in warm cache.
        resumed = AdaptiveDifficultySampler({"a": [0, 1, 2], "b": [3, 4, 5]}, cfg, epoch_budget=6)
        resumed.begin_round()
        skip = {0, 2, 5}
        drawn_items, drawn_total, exhausted = resumed.draw_unskipped(3, start_offset=0, skip_offsets=skip)
        returned_offsets = [off for _, _, off in drawn_items]
        self.assertEqual(returned_offsets, [1, 3, 4])
        self.assertEqual(drawn_total, 5)
        self.assertFalse(exhausted)
        # Trailing contiguous offset 5 is consumed cleanly at batch end.
        trailing = resumed.consume_contiguous_skipped(drawn_total, skip)
        self.assertEqual(trailing, 1)
        self.assertTrue(resumed.exhausted)
        # Returned indices must never include skipped duplicates from the same task cycle.
        self.assertEqual(len({idx for idx, _, _ in drawn_items}), 3)

    def test_config_from_args_and_env(self):
        args = SimpleNamespace(
            rollout=SimpleNamespace(adaptive_difficulty_enable=False, adaptive_difficulty_alpha=3.0),
            train=SimpleNamespace(seed=99),
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOLT_ADAPTIVE_DIFFICULTY", None)
            self.assertFalse(AdaptiveDifficultyConfig.from_args(args).enabled)
        with mock.patch.dict(os.environ, {"MOLT_ADAPTIVE_DIFFICULTY": "1"}, clear=False):
            cfg = AdaptiveDifficultyConfig.from_args(args)
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.max_alpha, 3.0)
            self.assertEqual(cfg.seed, 99)


if __name__ == "__main__":
    unittest.main()
