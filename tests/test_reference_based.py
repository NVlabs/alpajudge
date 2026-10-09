# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-based model and scoring tests."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from alpajudge.reference_based import (
    AlpaJudgeRB,
    AlpaJudgeRBConfig,
    load_reference_based,
    score_reference_based,
)

_TINY_BASE_CONFIG = {
    "model_type": "deberta-v2",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "vocab_size": 64,
    "max_position_embeddings": 32,
    "num_labels": 1,
}


class _FakeTokenizer:
    cls_token = "[CLS]"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts, **kwargs):
        del kwargs
        self.texts = list(texts)
        batch = len(self.texts)
        return {
            "input_ids": torch.ones((batch, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        }


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask):
        del attention_mask
        logits = torch.arange(len(input_ids), device=input_ids.device, dtype=torch.float32)
        return SimpleNamespace(logits=logits)


def _tiny_model() -> AlpaJudgeRB:
    return AlpaJudgeRB(
        AlpaJudgeRBConfig(pretrained_model_name="tiny", base_config=_TINY_BASE_CONFIG)
    )


def test_legacy_checkpoint_loads_through_release_registration(tmp_path) -> None:
    """The published pre-rename model type remains readable in a clean process."""
    _tiny_model().save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "alpajudge"
    config["architectures"] = ["AlpaJudge"]
    config_path.write_text(json.dumps(config))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "from transformers import AutoConfig, AutoModel",
                    "from alpajudge.reference_based import AlpaJudgeRB, AlpaJudgeRBConfig",
                    f"checkpoint = {str(tmp_path)!r}",
                    "assert isinstance(AutoConfig.from_pretrained(checkpoint), AlpaJudgeRBConfig)",
                    "assert isinstance(AutoModel.from_pretrained(checkpoint), AlpaJudgeRB)",
                    "print('legacy-load-ok')",
                ]
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "legacy-load-ok"


def test_local_loader_omits_empty_subfolder() -> None:
    """A local checkpoint path does not pass the invalid ``subfolder=None`` kwarg."""
    model = _tiny_model()
    tokenizer = _FakeTokenizer()
    with (
        patch(
            "alpajudge.reference_based.inference.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as load_tokenizer,
        patch(
            "alpajudge.reference_based.inference.AutoModel.from_pretrained",
            return_value=model,
        ) as load_model,
    ):
        loaded_model, loaded_tokenizer = load_reference_based(
            "/local/checkpoint",
            subfolder=None,
            device="cpu",
        )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    load_tokenizer.assert_called_once_with("/local/checkpoint", use_fast=True)
    load_model.assert_called_once_with("/local/checkpoint")


def test_score_reference_based_uses_frozen_template_and_order() -> None:
    """The helper formats every row and returns probabilities in input order."""
    tokenizer = _FakeTokenizer()
    scores = score_reference_based(
        _FakeModel(),
        tokenizer,
        [
            {"question": "q1", "answer": "a1", "student": "s1"},
            {"question": "q2", "answer": "a2", "student": "s2"},
        ],
        batch_size=2,
    )

    assert tokenizer.texts == [
        "[CLS]\nQuestion: q1\nAnswer: a1\nStudent: s1",
        "[CLS]\nQuestion: q2\nAnswer: a2\nStudent: s2",
    ]
    torch.testing.assert_close(scores, torch.sigmoid(torch.tensor([0.0, 1.0])))
