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

import sys
import types
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

if "ray" not in sys.modules:
    fake_ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(obj):
            return obj

        return decorator

    fake_ray.remote = remote
    fake_ray.get = MagicMock()
    fake_ray.put = MagicMock()
    fake_ray.wait = MagicMock()
    fake_ray.cancel = MagicMock()
    fake_util = types.ModuleType("ray.util")
    fake_placement_group = types.ModuleType("ray.util.placement_group")
    fake_placement_group.PlacementGroup = type("PlacementGroup", (), {})
    fake_placement_group.placement_group = MagicMock()
    fake_util.placement_group = fake_placement_group
    fake_scheduling = types.ModuleType("ray.util.scheduling_strategies")
    fake_scheduling.PlacementGroupSchedulingStrategy = type("PlacementGroupSchedulingStrategy", (), {})
    fake_util.scheduling_strategies = fake_scheduling
    fake_queue = types.ModuleType("ray.util.queue")
    fake_queue.Queue = MagicMock()
    fake_ray.util = fake_util
    sys.modules["ray"] = fake_ray
    sys.modules["ray.util"] = fake_util
    sys.modules["ray.util.placement_group"] = fake_placement_group
    sys.modules["ray.util.queue"] = fake_queue
    sys.modules["ray.util.scheduling_strategies"] = fake_scheduling
if "vllm" not in sys.modules:
    fake_vllm = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_vllm.SamplingParams = SamplingParams
    sys.modules["vllm"] = fake_vllm

from molt.agents.base import Trajectory
from molt.trainer.rollout import samples_generator
from molt.trainer.rollout.samples_generator import EvalExperience, SamplesGenerator, _compact_eval_experience


def _sample(group_id):
    return SimpleNamespace(group_ids=[group_id])


def _prompt_loader(num_prompts):
    """A dataloader yielding one prompt per item, as (index, prompts, labels, images, tools)."""
    return [(i, [f"p{i}"], [f"l{i}"], [None], [None]) for i in range(num_prompts)]


def _wire_fake_vllm(generator, monkeypatch, to_sample):
    """Wire the streaming generator to an in-memory vLLM that finishes rollouts FIFO.

    Each dispatched prompt becomes one in-flight rollout handle tagged with its
    prompt string; ray.wait hands them back in dispatch order and ray.get turns a
    handle into its single response, which `to_sample` maps to an Experience.
    """
    generator._dispatch_to_agent_runners = lambda prompts, labels, images, **kw: [
        SimpleNamespace(group_id=prompt) for prompt in prompts
    ]
    generator._process_response_into_experience = lambda response, **kw: (to_sample(response.group_id), None)
    monkeypatch.setattr(
        samples_generator.ray, "wait", lambda handles, num_returns=1: ([handles[0]], list(handles[1:]))
    )
    monkeypatch.setattr(samples_generator.ray, "get", lambda handle: [handle])


def test_generate_eval_samples_refills_independent_eval_pool(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        eval=SimpleNamespace(batch_size=3),
        rollout=SimpleNamespace(batch_size=1, n_samples_per_prompt=1),
    )
    generator.eval_dataloader = _prompt_loader(5)
    dispatches = []

    def fake_dispatch(prompts, labels, images=None, tools=None, **_kwargs):
        dispatches.append(list(prompts))
        return [SimpleNamespace(group_id=prompt) for prompt in prompts]

    generator._dispatch_to_agent_runners = fake_dispatch
    generator._process_response_into_experience = lambda response, **_kwargs: (
        _sample(response.group_id),
        None,
    )
    wait_fetch_local = []

    def fake_wait(handles, num_returns=1, timeout=None, fetch_local=True):
        wait_fetch_local.append(fetch_local)
        return [handles[0]], list(handles[1:])

    monkeypatch.setattr(
        samples_generator.ray,
        "wait",
        fake_wait,
    )
    monkeypatch.setattr(samples_generator.ray, "get", lambda handle: [handle])

    samples = generator.generate_eval_samples()

    assert all(isinstance(sample, EvalExperience) for sample in samples)
    assert [sample.group_ids[0] for sample in samples] == ["p0", "p1", "p2", "p3", "p4"]
    assert dispatches == [["p0", "p1", "p2"], ["p3"], ["p4"]]
    assert wait_fetch_local and not any(wait_fetch_local)


def test_compact_eval_experience_keeps_metrics_without_training_tensors():
    sample = SimpleNamespace(
        sequences=torch.ones(1, 1024),
        action_log_probs=torch.ones(1, 1023),
        prompts=["prompt"],
        group_ids=["group"],
        rollout_ids=["rollout"],
        rewards=torch.tensor([0.75]),
        response_length=torch.tensor([17]),
        truncated=torch.tensor([False]),
        info={
            "policy_version_start": torch.tensor([4]),
            "vln_ce_success": torch.tensor([1.0]),
            "non_numeric": "drop me",
        },
    )

    compact = _compact_eval_experience(sample)

    assert compact == EvalExperience(
        prompts=["prompt"],
        group_ids=["group"],
        rollout_ids=["rollout"],
        rewards=[0.75],
        response_length=[17],
        truncated=[False],
        info={"policy_version_start": 4, "vln_ce_success": 1.0},
    )
    assert not hasattr(compact, "sequences")
    assert not hasattr(compact, "action_log_probs")


def test_generate_samples_returns_batch_as_rollouts_finish_and_keeps_pool_saturated(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    samples, rollout_metrics, prompts_dispatched, exhausted = generator.generate_samples()

    # Returns exactly batch_size (3) finished groups, in completion (= dispatch) order.
    assert [sample.group_ids[0] for sample in samples] == ["p0", "p1", "p2"]
    # Pool stays saturated: 5 dispatched up front, then one refill per completion →
    # 7 dispatched total, the 4 unclaimed rollouts stay in flight for the next call.
    assert prompts_dispatched == 7
    assert [handle.group_id for handle in generator._inflight_rollouts] == ["p3", "p4", "p5", "p6"]
    assert generator._finished_samples == []
    # No drops and no dynamic filtering → no rollout metrics emitted.
    assert rollout_metrics == {}
    assert exhausted is False


def test_force_on_policy_generation_has_no_slow_tail_at_batch_boundary(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        train=SimpleNamespace(force_on_policy=True),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    first, _, first_dispatched, first_exhausted = generator.generate_samples()
    assert [sample.group_ids[0] for sample in first] == ["p0", "p1", "p2"]
    assert first_dispatched == 3
    assert first_exhausted is False
    assert generator._inflight_rollouts == []
    assert generator._finished_samples == []

    second, _, second_dispatched, _ = generator.generate_samples()
    assert [sample.group_ids[0] for sample in second] == ["p3", "p4", "p5"]
    assert second_dispatched == 3
    assert generator._inflight_rollouts == []
    assert generator._finished_samples == []


def test_force_on_policy_generation_drains_final_short_batch(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=4, n_samples_per_prompt=1, vllm_generate_batch_size=8),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        train=SimpleNamespace(force_on_policy=True),
    )
    generator.prompts_dataloader = _prompt_loader(2)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    samples, _, prompts_dispatched, exhausted = generator.generate_samples()

    assert [sample.group_ids[0] for sample in samples] == ["p0", "p1"]
    assert prompts_dispatched == 2
    assert exhausted is True
    assert generator._inflight_rollouts == []
    assert generator._finished_samples == []


def test_generate_samples_emits_short_batch_when_dataloader_exhausted(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=4, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(2)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    samples, _, prompts_dispatched, exhausted = generator.generate_samples()

    assert [sample.group_ids[0] for sample in samples] == ["p0", "p1"]
    assert prompts_dispatched == 2
    assert exhausted is True
    assert generator._inflight_rollouts == []


def test_generate_samples_pool_persists_across_calls(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    first, *_ = generator.generate_samples()
    second, _, prompts_dispatched, _ = generator.generate_samples()

    assert [sample.group_ids[0] for sample in first] == ["p0", "p1", "p2"]
    # The second batch is served from rollouts already in flight after the first call
    # (p3-p5) — vLLM never drained between steps — and the pool is topped back up.
    assert [sample.group_ids[0] for sample in second] == ["p3", "p4", "p5"]
    assert prompts_dispatched == 3  # only the 3 refills, not a fresh batch of 5
    assert [handle.group_id for handle in generator._inflight_rollouts] == ["p6", "p7", "p8", "p9"]


def test_generate_samples_drains_tail_before_starting_next_episode(monkeypatch):
    """Exhausting the iterator must not restart it while a slow tail remains.

    Seven prompts with a five-prompt in-flight pool and three-prompt train batch
    leave p6 in flight after the second update.  The old lifecycle keyed only on
    ``_dataloader_iter is None`` dropped p6 and restarted at p0 on the next call,
    making the outer episode loop run past 100% forever.
    """
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
    )
    generator.prompts_dataloader = _prompt_loader(7)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    first, _, first_dispatched, first_exhausted = generator.generate_samples()
    second, _, second_dispatched, second_exhausted = generator.generate_samples()
    tail, _, tail_dispatched, tail_exhausted = generator.generate_samples()

    assert [sample.group_ids[0] for sample in first] == ["p0", "p1", "p2"]
    assert [sample.group_ids[0] for sample in second] == ["p3", "p4", "p5"]
    assert [sample.group_ids[0] for sample in tail] == ["p6"]
    assert (first_dispatched, second_dispatched, tail_dispatched) == (7, 0, 0)
    assert (first_exhausted, second_exhausted, tail_exhausted) == (False, False, True)

    # Only after the tail reports exhaustion does the following outer episode
    # start a fresh iterator at p0.
    next_episode, _, _, next_exhausted = generator.generate_samples()
    assert [sample.group_ids[0] for sample in next_episode] == ["p0", "p1", "p2"]
    assert next_exhausted is False


def test_generator_keeps_no_checkpoint_state_and_resumes_from_dataloader(monkeypatch):
    """The in-flight pool is intentionally NOT persisted.

    Persisting the in-flight (prompt, label, images) payloads bloated checkpoints
    ~1000x (22-78 MB vs ~7 KB) and crashed the driver on resume, so the generator
    is stateless across checkpoints: state_dict() is empty and load_state_dict is a
    no-op tolerant of None/{}. The StatefulDataLoader cursor already points past the
    in-flight prefetch, so on resume those few prompts are skipped (a bounded loss,
    negligible for multi-epoch RL) rather than redispatched.
    """
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    first, *_ = generator.generate_samples()
    assert [sample.group_ids[0] for sample in first] == ["p0", "p1", "p2"]
    # p3-p6 are in flight at the checkpoint; the generator carries no state for them.
    assert generator.state_dict() == {}

    restored = object.__new__(SamplesGenerator)
    restored.args = generator.args
    # The StatefulDataLoader cursor in the checkpoint already points past the
    # prefetched prompts (p0-p6 were read), so resume starts at p7.
    restored.prompts_dataloader = [(i, [f"p{i}"], [f"l{i}"], [None], [None]) for i in range(7, 10)]
    _wire_fake_vllm(restored, monkeypatch, _sample)
    restored.load_state_dict(None)  # tolerate a missing payload
    restored.load_state_dict({})  # and an empty one

    second, _, newly_dispatched, _ = restored.generate_samples()
    # Resume continues from the dataloader cursor; the in-flight p3-p6 are not retrained.
    assert [sample.group_ids[0] for sample in second] == ["p7", "p8", "p9"]
    assert newly_dispatched == 3


def test_generate_samples_drops_filtered_groups_and_refills_their_slots(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=2, n_samples_per_prompt=1, vllm_generate_batch_size=2),
        algo=SimpleNamespace(dynamic_filtering_enable=True, dynamic_filtering_range=(0.0, 1.0)),
    )
    generator.prompts_dataloader = _prompt_loader(10)

    # p1's mean score (1.0) sits on the boundary of the open range (0, 1) → filtered out.
    group_score = {"p0": 0.5, "p1": 1.0, "p2": 0.5, "p3": 0.5}

    def scored_sample(group_id):
        return SimpleNamespace(group_ids=[group_id], scores=[torch.tensor(group_score[group_id])])

    _wire_fake_vllm(generator, monkeypatch, scored_sample)

    samples, rollout_metrics, prompts_dispatched, _ = generator.generate_samples()

    # p1 is dropped; p2 (refilled into p1's freed slot) completes the batch.
    assert [sample.group_ids[0] for sample in samples] == ["p0", "p2"]
    assert prompts_dispatched == 4  # p0,p1 up front; p2,p3 refilled one per completion
    assert rollout_metrics["dynamic_filtering_pass_rate"] == 2 / 4 * 100
    # The filtered group is tallied by reason for observability.
    assert rollout_metrics["rollout/dropped/dynamic_filter"] == 1.0
    assert rollout_metrics["rollout/dropped/total"] == 1.0


def test_process_response_counts_only_action_tokens_for_multiturn_lengths():
    generator = object.__new__(SamplesGenerator)

    experience, drop_reason = generator._process_response_into_experience(
        Trajectory(
            prompt="p",
            label="l",
            images=None,
            observation_text="",
            observation_tokens=list(range(8)),
            action_ranges=[(2, 4), (6, 7)],
            rollout_log_probs=[float(i) for i in range(8)],
            reward=1.0,
            scores=1.0,
        ),
        max_len=8,
    )

    assert drop_reason is None
    assert experience.response_length.item() == 3
    assert experience.action_mask.sum().item() == 3
    torch.testing.assert_close(
        experience.action_mask,
        torch.tensor([[False, True, True, False, False, True, False]]),
    )
    torch.testing.assert_close(
        experience.rollout_log_probs,
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]),
    )


def test_process_response_drops_raw_images_after_multimodal_preprocessing():
    generator = object.__new__(SamplesGenerator)
    generator.tokenizer = None
    mm_train_inputs = {"pixel_values": torch.ones(2, 3, 4, 4)}

    experience, drop_reason = generator._process_response_into_experience(
        Trajectory(
            prompt="p",
            label="l",
            images=["raw-image-copy"],
            observation_text="",
            observation_tokens=[0, 1, 2],
            action_ranges=[(1, 3)],
            rollout_log_probs=[0.0, 0.0, 0.0],
            reward=1.0,
            scores=1.0,
            mm_train_inputs=mm_train_inputs,
        ),
        max_len=8,
    )

    assert drop_reason is None
    assert experience.images == []
    assert experience.mm_train_inputs == [mm_train_inputs]
    assert experience.mm_train_input_specs == []


def test_process_response_compacts_vlm_payload_to_lossless_images(monkeypatch):
    monkeypatch.setenv("MOLT_COMPACT_VLM_EXPERIENCE", "1")
    generator = object.__new__(SamplesGenerator)
    generator.tokenizer = SimpleNamespace(image_processor=object())
    mm_train_inputs = {
        "pixel_values": torch.arange(24, dtype=torch.float32).reshape(2, 12),
        "image_grid_thw": torch.tensor([[1, 2, 3], [1, 4, 5]]),
    }
    pil_images = [object(), object()]

    experience, drop_reason = generator._process_response_into_experience(
        Trajectory(
            prompt="p",
            label="l",
            images=["source-ref"],
            observation_text="",
            observation_tokens=[0, 1, 2],
            action_ranges=[(1, 3)],
            rollout_log_probs=[0.0, 0.0, 0.0],
            reward=1.0,
            scores=1.0,
            mm_train_inputs=mm_train_inputs,
            pil_images=pil_images,
        ),
        max_len=8,
    )

    assert drop_reason is None
    assert experience.images == [pil_images]
    assert experience.mm_train_inputs == []
    assert len(experience.mm_train_input_specs) == 1
    assert set(experience.mm_train_input_specs[0]) == set(mm_train_inputs)


def test_process_response_keeps_generation_truncation_separate_from_episode_horizon():
    generator = object.__new__(SamplesGenerator)

    episode_only, drop_reason = generator._process_response_into_experience(
        Trajectory(
            prompt="p",
            label="l",
            images=None,
            observation_text="",
            observation_tokens=[0, 1, 2],
            action_ranges=[(1, 3)],
            rollout_log_probs=[0.0, 0.0, 0.0],
            reward=0.0,
            scores=0.0,
            truncated=True,
            extra_logs={"generation_truncated": False, "truncated": True},
        ),
        max_len=8,
    )

    assert drop_reason is None
    assert episode_only.truncated.item() is True
    assert episode_only.info["generation_truncated"].item() is False
    assert episode_only.info["truncated"].item() is True

    clipped, drop_reason = generator._process_response_into_experience(
        Trajectory(
            prompt="p",
            label="l",
            images=None,
            observation_text="",
            observation_tokens=[0, 1, 2, 3, 4, 5],
            action_ranges=[(1, 6)],
            rollout_log_probs=[0.0] * 6,
            reward=0.0,
            scores=0.0,
            extra_logs={"generation_truncated": False, "truncated": False},
        ),
        max_len=4,
    )

    assert drop_reason is None
    assert clipped.truncated.item() is True
    assert clipped.info["generation_truncated"].item() is True
    assert clipped.info["truncated"].item() is False


def test_process_response_rejects_action_ranges_outside_trajectory():
    generator = object.__new__(SamplesGenerator)

    with pytest.raises(ValueError, match="Invalid action range"):
        generator._process_response_into_experience(
            Trajectory(
                prompt="p",
                label="l",
                images=None,
                observation_text="",
                observation_tokens=[0, 1, 2],
                action_ranges=[(2, 4)],
                rollout_log_probs=[0.0, 0.0, 0.0],
                reward=1.0,
                scores=1.0,
            ),
            max_len=8,
        )


def test_process_response_skips_misaligned_rollout_logprobs():
    generator = object.__new__(SamplesGenerator)

    experience, drop_reason = generator._process_response_into_experience(
        Trajectory(
            prompt="p",
            label="l",
            images=None,
            observation_text="",
            observation_tokens=[0, 1, 2],
            action_ranges=[(1, 3)],
            rollout_log_probs=[0.0, 0.0],
            reward=1.0,
            scores=1.0,
        ),
        max_len=8,
    )

    assert experience is None
    assert drop_reason == "logprob_misalign"


def test_process_response_drops_backend_errors_only_from_training():
    generator = object.__new__(SamplesGenerator)
    trajectory = Trajectory(
        prompt="p",
        label="l",
        images=None,
        observation_text="",
        observation_tokens=[0, 1, 2],
        action_ranges=[(1, 3)],
        rollout_log_probs=[0.0, 0.0, 0.0],
        reward=0.0,
        scores=0.0,
        extra_logs={"nanobot_backend_error": True, "esibench_backend_error": True},
    )

    experience, drop_reason = generator._process_response_into_experience(trajectory, max_len=8, rollout_kind="train")
    assert experience is None
    assert drop_reason == "backend_error"

    # Eval must retain the row so exact-count/backend-error audits can see and
    # reject it instead of silently reporting a cleaner metric.
    experience, drop_reason = generator._process_response_into_experience(trajectory, max_len=8, rollout_kind="eval")
    assert experience is not None
    assert drop_reason is None
    assert experience.info["nanobot_backend_error"].item() is True


def test_filter_group_drops_all_training_siblings_after_one_backend_error(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(rollout=SimpleNamespace(n_samples_per_prompt=4))
    responses = [SimpleNamespace(i=i) for i in range(4)]

    def process(response, **_kwargs):
        if response.i == 1:
            return None, "backend_error"
        return SimpleNamespace(rollout_ids=[f"r{response.i}"], scores=torch.tensor([0.0])), None

    generator._process_response_into_experience = process
    monkeypatch.setattr(samples_generator.ray, "get", lambda _ref: responses)
    # defaultdict is the production type; use it here to assert exact tallies.
    drop_counts = defaultdict(int)
    kept = generator._filter_group(
        object(),
        dynamic_filtering=False,
        drop_counts=drop_counts,
        rollout_kind="train",
        n_samples_per_prompt=4,
    )

    assert kept == []
    assert dict(drop_counts) == {"backend_error": 1, "incomplete_group": 3}


def test_force_on_policy_filter_group_fails_immediately_on_backend_error(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        train=SimpleNamespace(force_on_policy=True),
        rollout=SimpleNamespace(n_samples_per_prompt=4),
    )

    generator._process_response_into_experience = lambda _response, **_kwargs: (None, "backend_error")
    monkeypatch.setattr(samples_generator.ray, "get", lambda _ref: [object()])

    with pytest.raises(RuntimeError, match="refusing to skip the affected prompt"):
        generator._filter_group(
            object(),
            dynamic_filtering=False,
            drop_counts=defaultdict(int),
            rollout_kind="train",
            n_samples_per_prompt=4,
        )


def test_dispatch_forwards_rollout_and_policy_metadata():
    calls = []

    class RemoteRunGroup:
        @staticmethod
        def remote(*args, **kwargs):
            calls.append((args, kwargs))
            return "ref"

    actor = SimpleNamespace(run_group=RemoteRunGroup())
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(n_samples_per_prompt=4),
        algo=SimpleNamespace(advantage=SimpleNamespace(is_correction_level="off")),
    )
    generator.agent_runners = [actor]
    generator._rr = 0

    refs = generator._dispatch_to_agent_runners(
        ["p"],
        ["l"],
        images=[None],
        max_len=128,
        n_samples_per_prompt=4,
        rollout_kind="eval",
        policy_version=17,
        policy_frozen=True,
    )

    assert refs == ["ref"]
    args, kwargs = calls[0]
    assert args[0:3] == ("p", "l", None)
    assert args[4:] == (128, 4)
    assert kwargs["tools"] is None
    assert (kwargs["rollout_kind"], kwargs["policy_version"], kwargs["policy_frozen"]) == ("eval", 17, True)
