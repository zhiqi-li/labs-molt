# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from molt.trainer import rl_trainer
from molt.trainer.algorithm.experience import Experience


def test_first_batch_bootstrap_continues_at_the_saved_prompt(tmp_path, monkeypatch):
    bootstrap = str(tmp_path / "first-rollout.pt")
    monkeypatch.setattr(rl_trainer.ray, "get", lambda value: value)
    monkeypatch.setattr(rl_trainer.ray, "put", lambda value: value)

    def run(name, *, produce=False, seed=42):
        dataset_path = tmp_path / f"{name}.jsonl"
        dataset_path.write_text('{"prompt": "same staged dataset"}\n')
        generator = object.__new__(rl_trainer.GenerateSamplesActor.__ray_metadata__.modified_class)
        generator.args = SimpleNamespace(
            actor=SimpleNamespace(model_name_or_path="same-model"),
            train=SimpleNamespace(
                rollout_bootstrap_output=bootstrap if produce else None,
                rollout_bootstrap_input=None if produce else bootstrap,
                rollout_replay_dir=None,
                rollout_dump_dir=None,
                num_episodes=1,
                seed=seed,
                agent_path="same-agent",
            ),
            data=SimpleNamespace(prompt_dataset=str(dataset_path)),
            rollout=SimpleNamespace(batch_size=2, num_runners=2 if produce else 4),
            vllm=SimpleNamespace(num_engines=4 if produce else 8),
            reward=SimpleNamespace(),
            algo=SimpleNamespace(dynamic_filtering_enable=True, dynamic_filtering_range=(0, 1)),
            eval=SimpleNamespace(steps=float("inf")),
        )
        generator.generate_kwargs = {"temperature": 1.0}
        generator.prompts_dataloader = StatefulDataLoader(range(4), batch_size=None)
        generator.eval_dataloader = None
        generator._partial_rollout = False
        generator.vllm_lock = SimpleNamespace(
            acquire=SimpleNamespace(remote=lambda: None),
            release=SimpleNamespace(remote=lambda: None),
        )
        payloads = []
        generator.rollout_queue = SimpleNamespace(put=lambda item, *, block: payloads.append(item))
        generator.rollout_slots = SimpleNamespace(get=lambda *, block: 0)

        def sample_batch(**_):
            prompts = iter(generator.prompts_dataloader)
            rows = [next(prompts), next(prompts)]
            return [
                Experience(sequences=torch.tensor([[i]]), group_ids=[str(i)], rollout_ids=[str(i)])
                for i in rows
            ], {"count": 2}, 2, not produce

        sample = Mock(side_effect=sample_batch)
        generator.samples_generator = SimpleNamespace(
            generate_samples=sample, prompts_drained=not produce, state_dict=dict
        )
        generator.fit(0, 0)
        sample.assert_called_once()
        return payloads

    assert run("producer", produce=True) == []
    first, second, done = run("consumer")
    assert [s.reload().sequences.item() for s in first[0]] == [0, 1]
    assert first[1]["total_consumed_prompts"] == 2 and first[2] == {"count": 2}
    assert [s.sequences.item() for s in second[0]] == [2, 3]
    assert second[1]["total_consumed_prompts"] == 4 and done == "done"
    with pytest.raises(ValueError, match="Rollout bootstrap recipe differs"):
        run("wrong-seed", seed=43)
