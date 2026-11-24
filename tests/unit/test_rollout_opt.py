# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rollout optimizations: delta mm kwargs_data, direct engine routing, and incremental chat decode."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import aiohttp
from PIL import Image

from molt.trainer.rollout.router import RouterGenerateClient


class _FakeResp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=SimpleNamespace(real_url="http://fake"),
                history=(),
                status=self.status,
                message="cache miss",
            )

    async def json(self):
        return self._p


class _FakeCtx:
    def __init__(self, resp):
        self._r = resp

    async def __aenter__(self):
        return self._r

    async def __aexit__(self, *a):
        return False


class _DirectDeltaFakeHttp:
    def __init__(self):
        self.calls = []
        self.fail_next_delta_500 = False

    def post(self, url, json=None, headers=None, data=None):
        self.calls.append((url, json, headers))
        if url.endswith("/v1/chat/completions/render"):
            n_imgs = len(json["messages"][0]["content"])
            idx_start = sum(1 for u, _, _ in self.calls[:-1] if u.endswith("/render")) * 10
            return _FakeCtx(
                _FakeResp(
                    {
                        "features": {
                            "mm_placeholders": {"image": [{"offset": i, "length": 1} for i in range(n_imgs)]},
                            "kwargs_data": {"image": [f"b64_item_{idx_start + i}" for i in range(n_imgs)]},
                            "mm_hashes": {"image": [f"hash_{idx_start + i}" for i in range(n_imgs)]},
                        }
                    }
                )
            )
        if url.endswith("/inference/v1/generate"):
            kw = json["features"]["kwargs_data"]["image"]
            if self.fail_next_delta_500 and any(x is None for x in kw):
                self.fail_next_delta_500 = False
                return _FakeCtx(_FakeResp({"error": "miss"}, status=500))
            return _FakeCtx(
                _FakeResp(
                    {
                        "choices": [
                            {
                                "token_ids": [42],
                                "finish_reason": "stop",
                                "logprobs": {"content": [{"token": "token_id:42", "logprob": -0.1}]},
                            }
                        ]
                    }
                )
            )
        raise AssertionError(f"Unexpected url {url}")


def _sp():
    return SimpleNamespace(max_tokens=16, temperature=1.0, top_p=1.0, top_k=-1, logprobs=1)


def test_direct_routing_and_delta_mm_with_500_fallback(monkeypatch):
    monkeypatch.setenv("MOLT_RENDER_CACHE", "1")
    monkeypatch.setenv("MOLT_VLLM_MM_CACHE_GB", "8")

    http = _DirectDeltaFakeHttp()
    engines = [f"http://10.0.0.{i}:8000" for i in range(4)]
    client = RouterGenerateClient(http, model_name="policy", image_token_id=99, engine_urls=engines)

    img1 = Image.new("RGB", (16, 16), color=(1, 2, 3))
    img2 = Image.new("RGB", (16, 16), color=(4, 5, 6))
    img3 = Image.new("RGB", (16, 16), color=(7, 8, 9))

    # Turn 1: 1 image -> full kwargs_data sent to pinned engine
    asyncio.run(client.generate([1, 99, 2], _sp(), multi_modal_data={"image": [img1]}, session_id="sess-A"))
    gen_calls = [(u, j) for u, j, _ in http.calls if u.endswith("/inference/v1/generate")]
    assert len(gen_calls) == 1
    pinned_url = gen_calls[0][0]
    assert any(pinned_url.startswith(e) for e in engines)
    assert gen_calls[0][1]["features"]["kwargs_data"]["image"] == ["b64_item_0"]

    # Turn 2: img1 + img2 -> img1 elided as None, only img2 sent!
    asyncio.run(client.generate([1, 99, 2, 99, 3], _sp(), multi_modal_data={"image": [img1, img2]}, session_id="sess-A"))
    gen_calls = [(u, j) for u, j, _ in http.calls if u.endswith("/inference/v1/generate")]
    assert len(gen_calls) == 2
    assert gen_calls[1][0] == pinned_url  # deterministic engine stickiness
    assert gen_calls[1][1]["features"]["kwargs_data"]["image"][0] is None
    assert gen_calls[1][1]["features"]["kwargs_data"]["image"][1] is not None

    # Turn 3: simulate post-refit P1 cache clear (500 on delta) -> auto immediate full resend succeeds!
    http.fail_next_delta_500 = True
    asyncio.run(
        client.generate(
            [1, 99, 2, 99, 3, 99, 4], _sp(), multi_modal_data={"image": [img1, img2, img3]}, session_id="sess-A"
        )
    )
    gen_calls = [(u, j) for u, j, _ in http.calls if u.endswith("/inference/v1/generate")]
    assert len(gen_calls) == 4  # 1 failed delta + 1 full resend
    assert gen_calls[2][1]["features"]["kwargs_data"]["image"][:2] == [None, None]
    assert all(x is not None for x in gen_calls[3][1]["features"]["kwargs_data"]["image"])

    # Turn 4: after full resend repopulated cache, delta works again!
    asyncio.run(
        client.generate(
            [1, 99, 2, 99, 3, 99, 4, 5], _sp(), multi_modal_data={"image": [img1, img2, img3]}, session_id="sess-A"
        )
    )
    gen_calls = [(u, j) for u, j, _ in http.calls if u.endswith("/inference/v1/generate")]
    assert len(gen_calls) == 5
    assert gen_calls[4][1]["features"]["kwargs_data"]["image"] == [None, None, None]
