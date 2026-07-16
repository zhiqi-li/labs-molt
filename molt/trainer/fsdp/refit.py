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

"""vLLM weight refit for the FSDP2/AutoModel backend.

Owns *how* to materialize each pushed parameter (``gather_full_param``): under
FSDP2, params are ``DTensor`` instances whose ``.full_tensor()`` gathers the
unsharded tensor across both FSDP shard and TP shard dims in one call.

The sender (``trainer/workers/policy_actor.py``) pushes every canonical
``state_dict`` entry, skipping only aliases proven to share the same tied
parameter with a present source. vLLM's ``load_weights`` matches the remaining
entries by name and decides which weights it accepts.
"""

from collections.abc import Mapping
from typing import Optional, Tuple

import torch
from torch.distributed.tensor import DTensor


def gather_full_param(param: torch.Tensor, dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Size]:
    """Materialize the full unsharded tensor for an FSDP2/TP-sharded parameter.

    Returns ``(full_tensor, full_shape)`` where ``full_tensor`` is on the local
    device with all mesh dims gathered. For non-DTensor params (e.g., the value
    head we don't shard, or buffers), returns ``(param.data, param.shape)``.

    Caller invokes this on each rank; ``full_tensor`` is replicated. Memory cost
    is the size of the full tensor on every participating rank — acceptable for
    weight refit (one-shot per training step). For very large models the async RL
    path uses per-tensor streaming with a ping-pong buffer to bound peak memory.
    """
    if isinstance(param, DTensor):
        full = param.full_tensor()
    else:
        full = param.data
    if dtype is not None and full.is_floating_point():
        full = full.to(dtype=dtype)
    return full, full.shape


def redundant_tied_weight_aliases(model: torch.nn.Module, state_dict: Mapping[str, object]) -> set[str]:
    """Return output aliases that duplicate an explicitly mapped tied weight.

    Modern Transformers models expose ``all_tied_weights_keys`` (or the class
    fallback ``_tied_weights_keys``) as an alias-to-source mapping.  For
    example, Qwen3-VL maps ``lm_head.weight`` to
    ``model.language_model.embed_tokens.weight``.  vLLM intentionally ignores
    the former when ``tie_word_embeddings`` is enabled, because loading the
    embedding source updates the shared output head too.  Sending that final
    alias in its own packed flush therefore produces a misleading ``loaded
    0/1`` warning even though the policy refit succeeded.

    Only skip an alias when both sides of an explicit mapping are present in
    this exact state dict.  Older list/regex forms are left untouched because
    they do not identify which entry is the authoritative source.
    """
    tied_keys = getattr(model, "all_tied_weights_keys", None)
    if tied_keys is None:
        tied_keys = getattr(model, "_tied_weights_keys", None)
    if not isinstance(tied_keys, Mapping):
        return set()
    aliases: set[str] = set()
    for alias, source in tied_keys.items():
        if (
            not isinstance(alias, str)
            or not isinstance(source, str)
            or alias == source
            or alias not in state_dict
            or source not in state_dict
        ):
            continue
        try:
            is_same_parameter = model.get_parameter(alias) is model.get_parameter(source)
        except (AttributeError, KeyError):
            is_same_parameter = False
        if is_same_parameter:
            aliases.add(alias)
    return aliases
