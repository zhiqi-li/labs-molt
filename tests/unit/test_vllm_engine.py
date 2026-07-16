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
import sys
from dataclasses import dataclass
from types import ModuleType


def _install_vllm_test_stub():
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.21.0"

    @dataclass
    class AsyncEngineArgs:
        model: str
        dtype: str = "auto"
        enforce_eager: bool = False
        disable_custom_all_reduce: bool = False

    vllm.AsyncEngineArgs = AsyncEngineArgs
    vllm.AsyncLLMEngine = object

    inputs = ModuleType("vllm.inputs")

    class TokensPrompt(dict):
        pass

    inputs.TokensPrompt = TokensPrompt

    utils = ModuleType("vllm.utils")
    utils.random_uuid = lambda: "test-request-id"

    sys.modules["vllm"] = vllm
    sys.modules["vllm.inputs"] = inputs
    sys.modules["vllm.utils"] = utils


try:
    import vllm.inputs  # noqa: F401
except Exception:
    _install_vllm_test_stub()

import molt.trainer.vllm.vllm_engine as vllm_engine  # noqa: E402


def test_vllm_ray_executor_uses_worker_gpu_even_when_actor_is_cpu_only():
    assert vllm_engine._vllm_worker_num_gpus("ray", 0) == 1
    assert vllm_engine._vllm_worker_num_gpus("mp", 8) == 8
    assert vllm_engine._vllm_worker_num_gpus("uni", 1) == 1


def test_format_ray_gpu_ids_keeps_all_visible_devices():
    assert vllm_engine._format_ray_gpu_ids([0.0, 2.0]) == "0,2"
    assert vllm_engine._format_ray_gpu_ids(["GPU-abc"]) == "GPU-abc"


def test_filter_vllm_engine_kwargs_drops_unsupported_optional_args(monkeypatch):
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        dtype: str = "auto"
        enforce_eager: bool = False

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs(
        {
            "model": "model-path",
            "dtype": "bfloat16",
            "gdn_prefill_backend": "triton",
        }
    )

    assert filtered == {"model": "model-path", "dtype": "bfloat16"}


def test_filter_vllm_engine_kwargs_keeps_speculative_config(monkeypatch):
    # MTP rollout passes speculative_config; it is a real AsyncEngineArgs field,
    # so it must survive the whitelist filter (not be dropped like unknown kwargs).
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        dtype: str = "auto"
        speculative_config: object = None

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs(
        {"model": "m", "speculative_config": {"num_speculative_tokens": 1}}
    )

    assert filtered == {"model": "m", "speculative_config": {"num_speculative_tokens": 1}}


def test_filter_vllm_engine_kwargs_keeps_disable_custom_all_reduce(monkeypatch):
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        disable_custom_all_reduce: bool = False

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs({"model": "m", "disable_custom_all_reduce": True})

    assert filtered == {"model": "m", "disable_custom_all_reduce": True}


def test_ray_visible_device_flag_is_cuda_only():
    assert vllm_engine.ray_noset_visible_devices({"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"})
    assert not vllm_engine.ray_noset_visible_devices({"RAY_EXPERIMENTAL_NOSET_OTHER_VISIBLE_DEVICES": "1"})


def _rollout_actor_instance(llm):
    metadata = getattr(vllm_engine.RolloutRayActor, "__ray_metadata__", None)
    actor_class = getattr(metadata, "modified_class", vllm_engine.RolloutRayActor)
    actor = actor_class.__new__(actor_class)
    actor.llm = llm
    actor._weight_version = 0
    return actor


def test_weight_version_advances_only_after_successful_packed_update():
    class FakeLLM:
        def __init__(self):
            self.fail = False

        async def collective_rpc(self, method, args):
            assert method == "update_weights_packed"
            if self.fail:
                raise RuntimeError("broadcast failed")
            return "ok"

    llm = FakeLLM()
    actor = _rollout_actor_instance(llm)

    assert asyncio.run(actor.get_weight_version()) == 0
    assert asyncio.run(actor.update_weights_packed([("a", "bf16", (1,))])) == "ok"
    assert asyncio.run(actor.get_weight_version()) == 1
    assert asyncio.run(actor.update_weights_packed([("b", "bf16", (1,))])) == "ok"
    assert asyncio.run(actor.get_weight_version()) == 2

    llm.fail = True
    try:
        asyncio.run(actor.update_weights_packed([("c", "bf16", (1,))]))
    except RuntimeError as exc:
        assert str(exc) == "broadcast failed"
    else:
        raise AssertionError("expected the fake collective to fail")
    assert asyncio.run(actor.get_weight_version()) == 2
