# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from molt.trainer import rl_trainer
from molt.trainer.algorithm.experience import Experience


@pytest.mark.parametrize("start_cursor", [0, 2])
def test_first_batch_bootstrap_continues_at_the_saved_prompt(tmp_path, monkeypatch, start_cursor):
    bootstrap = str(tmp_path / "first-rollout.pt")
    resume = None
    source_loader = StatefulDataLoader(range(start_cursor + 4), batch_size=None)
    if start_cursor:
        source_iter = iter(source_loader)
        for _ in range(start_cursor):
            next(source_iter)
        anchor = tmp_path / "_actor/global_step11"
        anchor.mkdir(parents=True)
        hf = tmp_path / "_hf/global_step11"
        hf.mkdir(parents=True)
        (hf / "model.safetensors.index.json").write_text("{}")
        state = {"global_step": 11, "episode": 0, "total_consumed_prompts": start_cursor,
                 "data_loader_state_dict": source_loader.state_dict()}
        torch.save({"client_state": state}, anchor / "extra_state.pt")
        _, resume = rl_trainer._load_bootstrap_resume_source(str(anchor))
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
            rollout=SimpleNamespace(
                batch_size=2,
                num_runners=2 if produce else 4,
                vllm_generate_batch_size=2 if produce else 3,
            ),
            vllm=SimpleNamespace(
                num_engines=4 if produce else 8, router_policy="cache_aware" if produce else "consistent_hash"
            ),
            reward=SimpleNamespace(),
            algo=SimpleNamespace(dynamic_filtering_enable=True, dynamic_filtering_range=(0, 1)),
            eval=SimpleNamespace(steps=float("inf")),
        )
        generator.generate_kwargs = {"temperature": 1.0}
        generator.prompts_dataloader = StatefulDataLoader(range(start_cursor + 4), batch_size=None)
        if resume:
            generator.prompts_dataloader.load_state_dict(source_loader.state_dict())
        generator.eval_dataloader = None
        generator._partial_rollout = not produce
        generator.vllm_lock = SimpleNamespace(
            acquire=SimpleNamespace(remote=Mock()),
            release=SimpleNamespace(remote=Mock()),
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
        generator.fit(0, start_cursor, resume)
        sample.assert_called_once()
        if not produce:
            generator.vllm_lock.acquire.remote.assert_not_called()
            generator.vllm_lock.release.remote.assert_not_called()
        return payloads

    assert run("producer", produce=True) == []
    saved = torch.load(bootstrap, weights_only=False, map_location="cpu")
    assert "router_policy" not in saved["recipe"]["vllm"]
    if resume:
        assert saved["recipe"]["resume"] == resume
    assert "vllm_generate_batch_size" not in saved["recipe"]["rollout"]
    saved["recipe"]["vllm"]["router_policy"] = "cache_aware"  # bundle from before routing was excluded
    saved["recipe"]["rollout"]["vllm_generate_batch_size"] = 2
    torch.save(saved, bootstrap)
    first, second, done = run("consumer")
    assert [s.reload().sequences.item() for s in first[0]] == [start_cursor, start_cursor + 1]
    assert first[1]["total_consumed_prompts"] == start_cursor + 2 and first[2] == {"count": 2}
    assert [s.sequences.item() for s in second[0]] == [start_cursor + 2, start_cursor + 3]
    assert second[1]["total_consumed_prompts"] == start_cursor + 4 and done == "done"
    with pytest.raises(ValueError, match="Rollout bootstrap recipe differs"):
        run("wrong-seed", seed=43)
