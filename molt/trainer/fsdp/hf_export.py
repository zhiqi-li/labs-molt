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

"""Fast cached-mmap and asynchronous HF safetensors export from DCP model shards.

Stock AutoModel safetensors consolidation opens and mmaps every input shard file once
per tensor (~640 tensors x 16 shards = ~10,240 Lustre open/mmap/close cycles) and writes
consolidated files directly over Lustre while blocking all trainer GPUs (~200-240s/step).

This module provides:
1. ``install_fast_safetensors_consolidation()``: keeps one open mmap per input shard for
   the duration of each output shard pass, cutting 10,240 Lustre opens down to N_shards.
2. Standalone CLI ``python3 -m molt.trainer.fsdp.hf_export`` invoked asynchronously in a
   background subprocess by rank 0 after DCP ``save_ckpt`` writes ``_actor/<tag>/model``,
   staging outputs on local ``/tmp`` before atomically publishing ``_hf/<tag>``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import glob
import json
import logging
import mmap
import os
from pathlib import Path
import shutil
import time
from typing import Any

logger = logging.getLogger(__name__)

_FAST_PATCHED = False


def install_fast_safetensors_consolidation() -> bool:
    """Patch AutoModel's ``_process_output_file`` to reuse one open mmap per input shard."""
    global _FAST_PATCHED
    if _FAST_PATCHED:
        return True
    try:
        from nemo_automodel.components.checkpoint._backports import consolidate_hf_safetensors as mod
    except ImportError:
        return False

    if getattr(mod, "_molt_fast_mmap_patched", False):
        _FAST_PATCHED = True
        return True

    def _fast_process_output_file(
        output_file: str,
        output_data: Any,
        input_files_data: dict[str, Any],
    ) -> None:
        with ExitStack() as stack:
            # Open and mmap each input shard once instead of once per tensor.
            open_mmaps: dict[str, mmap.mmap] = {}
            for safetensors_file in input_files_data:
                stream = stack.enter_context(open(safetensors_file, "rb"))
                open_mmaps[safetensors_file] = stack.enter_context(
                    mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                )

            with open(output_file, "wb") as output_stream:
                metadata = mod._compute_safetensors_metadata_and_offsets(output_data)
                mod._write_safetensors_header(output_stream, output_data, metadata)
                sorted_tensors = sorted(output_data.fqn_data.items(), key=lambda x: x[1].offset_in_file)

                for tensor_fqn, tensor_fqn_data in sorted_tensors:
                    full_tensor_mv = memoryview(
                        bytearray(
                            mod.math.prod(tensor_fqn_data.shape_in_file) * tensor_fqn_data.source_dtype_size
                        )
                    )
                    for safetensors_file, file_data in input_files_data.items():
                        file_metadata = file_data.metadata
                        if tensor_fqn not in file_metadata:
                            continue
                        tensor_meta = file_metadata[tensor_fqn]
                        start_off, end_off = tensor_meta[mod.DATA_OFFSETS_KEY]
                        mm = open_mmaps[safetensors_file]
                        abs_start = file_data.metadata_size + start_off
                        abs_end = file_data.metadata_size + end_off
                        data_to_write = bytes(mm[abs_start:abs_end])
                        fqn_custom_metadata = mod._get_dcp_custom_metadata(file_metadata)[tensor_fqn]
                        offsets = fqn_custom_metadata[mod.SAVED_OFFSETS_KEY]
                        mod._write_sub_tensor_to_file_optimized(
                            full_tensor_mv,
                            data_to_write,
                            tensor_fqn_data.source_dtype_size,
                            tensor_fqn_data.shape_in_file,
                            offsets,
                            tensor_meta[mod.SHAPE_KEY],
                        )
                    if tensor_fqn_data.source_dtype_str == tensor_fqn_data.dtype_str:
                        output_stream.write(full_tensor_mv)
                    else:
                        output_stream.write(
                            mod._cast_tensor_bytes(
                                full_tensor_mv,
                                tensor_fqn_data.shape_in_file,
                                tensor_fqn_data.source_dtype_str,
                                tensor_fqn_data.dtype_str,
                            )
                        )

    mod._process_output_file = _fast_process_output_file
    mod._molt_fast_mmap_patched = True
    _FAST_PATCHED = True
    return True


def _collect_shard_fqns(shard_dir: str) -> list[str]:
    from nemo_automodel.components.checkpoint._backports.hf_utils import (
        DEFAULT_EXTRA_METADATA_KEY,
        SUFFIX,
        _get_safetensors_file_metadata,
    )

    fqns: dict[str, None] = {}
    for path in sorted(glob.glob(os.path.join(shard_dir, f"*{SUFFIX}"))):
        with open(path, "rb") as f:
            meta, _ = _get_safetensors_file_metadata(f)
        for key in meta:
            if key != DEFAULT_EXTRA_METADATA_KEY:
                fqns[key] = None
    return list(fqns)


def _resolve_mappings(
    shard_dir: str,
    reference_model_dir: str | None,
    forced_dtypes: dict[str, str] | None = None,
) -> tuple[dict[str, int], dict[str, str] | None]:
    from nemo_automodel.components.checkpoint.checkpointing import (
        get_fqn_to_dtype_mapping,
        get_fqn_to_file_index_mapping,
    )

    present_fqns = _collect_shard_fqns(shard_dir)
    present_set = set(present_fqns)
    fqn_to_index: dict[str, int] = {}
    fqn_to_dtype: dict[str, str] = {}

    if reference_model_dir and os.path.isdir(reference_model_dir):
        try:
            ref_idx = get_fqn_to_file_index_mapping(reference_model_dir, None) or {}
            fqn_to_index = {k: v for k, v in ref_idx.items() if k in present_set}
        except Exception as exc:
            logger.warning("Failed to load reference shard index mapping from %s: %s", reference_model_dir, exc)
        try:
            ref_dtype = get_fqn_to_dtype_mapping(reference_model_dir, None) or {}
            fqn_to_dtype = {k: v for k, v in ref_dtype.items() if k in present_set}
        except Exception as exc:
            logger.warning("Failed to load reference dtype mapping from %s: %s", reference_model_dir, exc)

    default_idx = max(fqn_to_index.values()) if fqn_to_index else 1
    for fqn in present_fqns:
        fqn_to_index.setdefault(fqn, default_idx)

    if forced_dtypes:
        for fqn, dtype_str in forced_dtypes.items():
            if fqn in present_set:
                fqn_to_dtype[fqn] = dtype_str

    return fqn_to_index, (fqn_to_dtype or None)


def _prune_hf_dirs(hf_root: str, keep_tag: str, max_num: int, max_mem: float) -> None:
    if not os.path.isdir(hf_root):
        return
    is_best = keep_tag.startswith("best")
    if is_best:
        for name in os.listdir(hf_root):
            path = os.path.join(hf_root, name)
            if name.startswith("best") and name != keep_tag and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        return

    max_size_bytes = None
    if max_mem is not None and max_mem > 0 and max_mem != float("inf"):
        max_size_bytes = max_mem * 1024**3

    def _dir_size(path: str) -> int:
        total = 0
        for dp, _, fns in os.walk(path):
            for fn in fns:
                try:
                    total += os.path.getsize(os.path.join(dp, fn))
                except OSError:
                    pass
        return total

    while True:
        subdirs = [
            (os.path.join(hf_root, name), os.path.getmtime(os.path.join(hf_root, name)))
            for name in os.listdir(hf_root)
            if not name.startswith(".") and os.path.isdir(os.path.join(hf_root, name))
        ]
        regular = [
            (p, m)
            for p, m in subdirs
            if not os.path.basename(p).startswith("best") and os.path.basename(p) != keep_tag
        ]
        current_regular = sum(1 for p, _ in subdirs if not os.path.basename(p).startswith("best"))
        overflow_num = max(0, current_regular - max_num) if max_num and max_num > 0 else 0
        overflow_mem = max_size_bytes is not None and sum(_dir_size(p) for p, _ in subdirs) > max_size_bytes
        if overflow_num == 0 and not overflow_mem:
            break
        candidates = sorted(regular, key=lambda item: item[1])
        if not candidates:
            break
        shutil.rmtree(candidates[0][0], ignore_errors=True)


def run_consolidation_job(
    *,
    staging_root: str,
    hf_root: str,
    tag: str,
    reference_model_dir: str | None,
    cast_dtype: str | None,
    num_threads: int,
    max_num: int,
    max_mem: float,
    forced_dtypes_path: str | None = None,
    local_tmp_dir: str | None = None,
) -> None:
    """Consolidate hardlinked DCP model shards into ``hf_root/tag`` atomically."""
    install_fast_safetensors_consolidation()
    from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
        consolidate_safetensors_files,
        resolve_dtype_cast,
    )

    t0 = time.time()
    shard_dir = os.path.join(staging_root, "shards")
    out_dir = os.path.join(staging_root, "out")
    final_dir = os.path.join(hf_root, tag)

    forced_dtypes = None
    if forced_dtypes_path and os.path.isfile(forced_dtypes_path):
        with open(forced_dtypes_path) as f:
            forced_dtypes = json.load(f)

    fqn_to_index, fqn_to_dtype = _resolve_mappings(shard_dir, reference_model_dir, forced_dtypes)
    resolved_cast = resolve_dtype_cast(cast_dtype) if (cast_dtype and fqn_to_dtype is None) else None
    tmp_stage = local_tmp_dir or os.environ.get("TMPDIR") or "/tmp"
    os.makedirs(tmp_stage, exist_ok=True)

    consolidate_safetensors_files(
        input_dir=shard_dir,
        output_dir=out_dir,
        fqn_to_index_mapping=fqn_to_index,
        num_threads=num_threads,
        use_staging=True,
        staging_dir=tmp_stage,
        cast_dtype=resolved_cast,
        fqn_to_dtype_mapping=fqn_to_dtype,
    )

    if os.path.exists(final_dir):
        shutil.rmtree(final_dir, ignore_errors=True)
    os.replace(out_dir, final_dir)
    shutil.rmtree(staging_root, ignore_errors=True)
    _prune_hf_dirs(hf_root, tag, max_num, max_mem)
    logger.info("Async HF export published %s in %.2fs", final_dir, time.time() - t0)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Async HF safetensors consolidation worker.")
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--hf-root", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--reference-model-dir", default=None)
    parser.add_argument("--cast-dtype", default="bf16")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-num", type=int, default=2)
    parser.add_argument("--max-mem", type=float, default=float("inf"))
    parser.add_argument("--forced-dtypes-path", default=None)
    parser.add_argument("--local-tmp-dir", default=None)
    args = parser.parse_args(argv)
    run_consolidation_job(
        staging_root=args.staging_root,
        hf_root=args.hf_root,
        tag=args.tag,
        reference_model_dir=args.reference_model_dir,
        cast_dtype=args.cast_dtype,
        num_threads=args.num_threads,
        max_num=args.max_num,
        max_mem=args.max_mem,
        forced_dtypes_path=args.forced_dtypes_path,
        local_tmp_dir=args.local_tmp_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
