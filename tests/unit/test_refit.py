# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from molt.trainer.fsdp.refit import redundant_tied_weight_aliases


class _TiedModel(nn.Module):
    def __init__(self, *, tied=True):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight
        self.all_tied_weights_keys = {"lm_head.weight": "embed.weight"}


def test_redundant_tied_weight_aliases_returns_shared_output_alias():
    model = _TiedModel(tied=True)

    assert redundant_tied_weight_aliases(model, model.state_dict()) == {"lm_head.weight"}


def test_redundant_tied_weight_aliases_keeps_unshared_or_incomplete_entries():
    model = _TiedModel(tied=False)
    assert redundant_tied_weight_aliases(model, model.state_dict()) == set()

    tied_model = _TiedModel(tied=True)
    state_dict = tied_model.state_dict()
    state_dict.pop("embed.weight")
    assert redundant_tied_weight_aliases(tied_model, state_dict) == set()


def test_redundant_tied_weight_aliases_ignores_legacy_list_metadata():
    model = _TiedModel(tied=True)
    model.all_tied_weights_keys = ["lm_head.weight"]

    assert redundant_tied_weight_aliases(model, model.state_dict()) == set()
