# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone loading and scoring for AlpaJudge-RF."""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel

from alpajudge.reference_free.model import (
    SUBSCORE_NAMES,
    AlpaJudgeRF,
    aggregate_hard_subscores,
    aggregate_soft_subscores,
)
from alpajudge.reference_free.processing import AlpaJudgeRFProcessor

DEFAULT_MODEL_ID = "nvidia/alpajudge"
DEFAULT_SUBFOLDER = "reference-free"


@dataclass(frozen=True)
class ReferenceFreeScores:
    """Five rubric probabilities and their soft/hard aggregations."""

    subscores: dict[str, float]
    action: float
    component: float
    overall: float
    hard_action: float
    hard_component: float
    hard_overall: float


def load_reference_free(
    pretrained_model_name_or_path: str = DEFAULT_MODEL_ID,
    *,
    subfolder: str | None = DEFAULT_SUBFOLDER,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | str | None = torch.bfloat16,
    processor_kwargs: dict[str, Any] | None = None,
) -> tuple[AlpaJudgeRF, AlpaJudgeRFProcessor]:
    """Load the AlpaJudge-RF model and its frozen input processor."""
    model_kwargs: dict[str, Any] = {} if subfolder is None else {"subfolder": subfolder}
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    model = AutoModel.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
    if not isinstance(model, AlpaJudgeRF):
        raise TypeError(f"expected AlpaJudgeRF, got {type(model).__name__}")
    model = model.eval().to(device)
    processor = AlpaJudgeRFProcessor.from_pretrained(
        pretrained_model_name_or_path,
        config=model.config,
        subfolder=subfolder,
        **(processor_kwargs or {}),
    )
    return model, processor


@torch.inference_mode()
def score_reference_free(
    model: AlpaJudgeRF,
    processor: AlpaJudgeRFProcessor,
    *,
    front_wide_frames: torch.Tensor,
    front_tele_frames: torch.Tensor,
    meta_action_history: str,
    meta_action_future: str,
    rationale: str,
    threshold: float = 0.5,
) -> ReferenceFreeScores:
    """Score one rationale under the frozen six-frame and meta-action contract."""
    inputs = processor(
        front_wide_frames=front_wide_frames,
        front_tele_frames=front_tele_frames,
        meta_action_history=meta_action_history,
        meta_action_future=meta_action_future,
        rationale=rationale,
    )
    device = next(model.parameters()).device
    inputs = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in inputs.items()
    }
    logits = model(**inputs).logits.float()
    if logits.shape != (1, len(SUBSCORE_NAMES)):
        raise ValueError(
            f"expected one row of {len(SUBSCORE_NAMES)} sub-score logits, got {tuple(logits.shape)}"
        )
    probabilities = torch.sigmoid(logits)
    soft = aggregate_soft_subscores(probabilities)
    hard = aggregate_hard_subscores((probabilities >= threshold).to(probabilities.dtype))
    return ReferenceFreeScores(
        subscores={
            name: probabilities[0, index].item() for index, name in enumerate(SUBSCORE_NAMES)
        },
        action=soft.action.item(),
        component=soft.component.item(),
        overall=soft.overall.item(),
        hard_action=hard.action.item(),
        hard_component=hard.component.item(),
        hard_overall=hard.overall.item(),
    )
