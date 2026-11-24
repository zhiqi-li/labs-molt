# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FLA coarse NB autotune bucketing."""

from __future__ import annotations

from molt.trainer.fsdp.fla_opt import _coarse_nb


def test_coarse_nb_single_bucket_across_262k():
    assert _coarse_nb(1000, 1048576) == 1
    assert _coarse_nb(65536, 1048576) == 1
    assert _coarse_nb(262144, 1048576) == 1


def test_coarse_nb_power_of_two_buckets():
    assert _coarse_nb(16384, 16384) == 1
    assert _coarse_nb(16385, 16384) == 2
    assert _coarse_nb(49152, 16384) == 4
