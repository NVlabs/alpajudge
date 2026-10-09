# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-free prompt, processing, and scoring tests."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest
from torch import nn

from alpajudge.reference_free import (
    AlpaJudgeRFProcessor,
    aggregate_hard_subscores,
    aggregate_soft_subscores,
    score_reference_free,
)
from alpajudge.reference_free.processing import (
    COMPACT_CLASSIFIER_INSTRUCTION,
    prompt_contract_sha256,
    prompt_sha256,
)

EXPECTED_PROMPT_CONTRACT = "15474783891270646f087b4642f11565b0f42c7e5df697c6759c9f35b2615706"


class _FakeHFProcessor:
    def __init__(self) -> None:
        self.content = None
        self.image_count = 0

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        self.content = messages[0]["content"]
        return "rendered prompt"

    def __call__(self, *, images, do_rescale, **kwargs):
        del kwargs
        assert do_rescale is False
        self.image_count = len(images)
        return {
            "input_ids": torch.ones((1, 12), dtype=torch.long),
            "attention_mask": torch.ones((1, 12), dtype=torch.long),
            "pixel_values": torch.zeros((6, 4), dtype=torch.float32),
            "image_grid_thw": torch.ones((6, 3), dtype=torch.long),
        }


class _FakeRFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, **kwargs):
        del kwargs
        return SimpleNamespace(logits=torch.zeros((1, 5)))


def _processor() -> tuple[AlpaJudgeRFProcessor, _FakeHFProcessor]:
    hf_processor = _FakeHFProcessor()
    config = SimpleNamespace(
        student_prompt_sha256=prompt_sha256(COMPACT_CLASSIFIER_INSTRUCTION),
        prompt_contract_sha256=EXPECTED_PROMPT_CONTRACT,
        max_length=4096,
    )
    return AlpaJudgeRFProcessor(hf_processor, config), hf_processor


def test_prompt_contract_matches_published_checkpoint() -> None:
    assert prompt_contract_sha256() == EXPECTED_PROMPT_CONTRACT


def test_processor_builds_six_frame_contract() -> None:
    processor, hf_processor = _processor()
    inputs = processor(
        front_wide_frames=torch.zeros((3, 3, 8, 8), dtype=torch.uint8),
        front_tele_frames=torch.zeros((3, 3, 8, 8), dtype=torch.uint8),
        meta_action_history="history",
        meta_action_future="future",
        rationale="rationale",
    )

    assert hf_processor.image_count == 6
    assert sum(block["type"] == "image" for block in hf_processor.content) == 6
    assert inputs["input_ids"].shape == (1, 12)


def test_processor_loader_keeps_multimodal_processor_for_subfolders() -> None:
    """Sibling checkpoint loading must not fall back to a text-only tokenizer."""
    config = SimpleNamespace(
        student_prompt_sha256=prompt_sha256(COMPACT_CLASSIFIER_INSTRUCTION),
        prompt_contract_sha256=EXPECTED_PROMPT_CONTRACT,
        min_pixels=163_840,
        max_pixels=196_608,
    )
    hf_processor = _FakeHFProcessor()
    with patch(
        "alpajudge.reference_free.processing.Qwen3VLProcessor.from_pretrained",
        return_value=hf_processor,
    ) as load_processor:
        processor = AlpaJudgeRFProcessor.from_pretrained(
            "nvidia/alpajudge",
            config=config,
            subfolder="reference-free",
        )

    assert processor.processor is hf_processor
    load_processor.assert_called_once_with(
        "nvidia/alpajudge",
        trust_remote_code=True,
        subfolder="reference-free",
        min_pixels=163_840,
        max_pixels=196_608,
    )


def test_aggregation_matches_score_contract() -> None:
    probabilities = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
    soft = aggregate_soft_subscores(probabilities)
    hard = aggregate_hard_subscores((probabilities >= 0.5).float())

    torch.testing.assert_close(soft.action, torch.tensor([0.375]))
    torch.testing.assert_close(soft.component, torch.tensor([0.5]))
    torch.testing.assert_close(soft.overall, torch.tensor([0.425]))
    torch.testing.assert_close(hard.overall, torch.tensor([1.0]))


def test_score_reference_free_returns_named_subscores() -> None:
    processor, _ = _processor()
    scores = score_reference_free(
        _FakeRFModel(),
        processor,
        front_wide_frames=torch.zeros((3, 3, 8, 8), dtype=torch.uint8),
        front_tele_frames=torch.zeros((3, 3, 8, 8), dtype=torch.uint8),
        meta_action_history="history",
        meta_action_future="future",
        rationale="rationale",
    )

    assert set(scores.subscores) == {
        "action_matches_gt",
        "action_safe",
        "components_grounded",
        "components_complete",
        "component_causal",
    }
    assert scores.overall == pytest.approx(0.425)
