# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for media-aligned bounded FFD packing in NaiveReplayBuffer."""

from __future__ import annotations

from types import SimpleNamespace

from molt.trainer.algorithm.replay_buffer import _bounded_ffd_bins, _expand_bins_to_count, _experience_media_keys


def test_bounded_ffd_packing_and_dp_expansion():
    lengths = [249000, 170000, 150000, 60000, 55000, 48000, 42000, 35000, 32000]
    indices = list(range(len(lengths)))
    cap = 128000
    bins = _bounded_ffd_bins(indices, lengths, cap)

    assert sorted(i for b in bins for i in b) == indices
    for b in bins:
        if len(b) > 1:
            assert sum(lengths[i] for i in b) <= cap

    expanded = _expand_bins_to_count(bins, lengths, len(bins) + 2)
    assert len(expanded) == len(bins) + 2
    assert sorted(i for b in expanded for i in b) == indices
    sums = [sum(lengths[i] for i in b) for b in expanded]
    assert sums == sorted(sums, reverse=True)
    for b in expanded:
        if len(b) > 1:
            assert sum(lengths[i] for i in b) <= cap


def test_experience_media_keys_handles_both_media_and_text_only():
    text_exp = SimpleNamespace(mm_train_inputs=[])
    vision_exp = SimpleNamespace(mm_train_inputs=[{"pixel_values": None, "image_grid_thw": None}])
    assert _experience_media_keys(text_exp) == ()
    assert _experience_media_keys(vision_exp) == ("image_grid_thw", "pixel_values")
