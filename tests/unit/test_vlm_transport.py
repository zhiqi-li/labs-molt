# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from molt.utils.vlm_utils import make_mm_train_input_spec, rebuild_mm_train_inputs


class _ImageProcessor:
    def __init__(self, outputs):
        self.outputs = outputs

    def __call__(self, images, return_tensors):
        assert images == ["frame-0", "frame-1"]
        assert return_tensors == "pt"
        return {key: value.clone() for key, value in self.outputs.items()}


def test_rebuild_mm_train_inputs_matches_exact_fingerprint():
    original = {
        "pixel_values": torch.arange(240, dtype=torch.float32).reshape(20, 12),
        "image_grid_thw": torch.tensor([[1, 2, 3], [1, 4, 5]]),
    }
    processor = type("Processor", (), {"image_processor": _ImageProcessor(original)})()

    rebuilt = rebuild_mm_train_inputs(
        processor,
        ["frame-0", "frame-1"],
        make_mm_train_input_spec(original),
    )

    assert set(rebuilt) == set(original)
    for key in original:
        torch.testing.assert_close(rebuilt[key], original[key], rtol=0, atol=0)


def test_rebuild_mm_train_inputs_rejects_image_order_or_processor_drift():
    original = {
        "pixel_values": torch.arange(240, dtype=torch.float32).reshape(20, 12),
        "image_grid_thw": torch.tensor([[1, 2, 3], [1, 4, 5]]),
    }
    changed = {key: value.clone() for key, value in original.items()}
    changed["pixel_values"].view(-1)[0] += 1
    processor = type("Processor", (), {"image_processor": _ImageProcessor(changed)})()

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        rebuild_mm_train_inputs(
            processor,
            ["frame-0", "frame-1"],
            make_mm_train_input_spec(original),
        )
