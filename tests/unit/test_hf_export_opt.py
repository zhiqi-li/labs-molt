# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for fast cached-mmap and async HF safetensors export."""

from __future__ import annotations

import argparse

from molt.cli.common_args import add_ckpt_args
from molt.trainer.fsdp.hf_export import _prune_hf_dirs


def test_prune_hf_dirs_ignores_hidden_staging(tmp_path):
    for name in (".staging_global_step30", "global_step28", "global_step29", "global_step30"):
        (tmp_path / name).mkdir()
    _prune_hf_dirs(str(tmp_path), "global_step30", max_num=2, max_mem=float("inf"))
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert ".staging_global_step30" in remaining
    assert "global_step30" in remaining
    assert len([n for n in remaining if not n.startswith(".")]) == 2


def test_ckpt_cli_registers_save_hf_steps_and_async_save_hf():
    parser = argparse.ArgumentParser()
    add_ckpt_args(parser, default_ckpt_path="./ckpt")
    args = parser.parse_args(["--ckpt.save_hf", "--ckpt.save_hf_steps", "2", "--ckpt.async_save_hf"])
    assert getattr(args, "ckpt.save_hf") is True
    assert getattr(args, "ckpt.save_hf_steps") == 2
    assert getattr(args, "ckpt.async_save_hf") is True
