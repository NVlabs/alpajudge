# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone loading and batched scoring for AlpaJudge-RB."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from alpajudge.reference_based.model import AlpaJudgeRB

DEFAULT_MODEL_ID = "nvidia/alpajudge"
DEFAULT_SUBFOLDER = "reference-based"
MAX_LENGTH = 256


def load_reference_based(
    pretrained_model_name_or_path: str = DEFAULT_MODEL_ID,
    *,
    subfolder: str | None = DEFAULT_SUBFOLDER,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | str | None = None,
) -> tuple[AlpaJudgeRB, Any]:
    """Load the AlpaJudge-RB model and tokenizer from Hugging Face or a local directory."""
    location_kwargs = {} if subfolder is None else {"subfolder": subfolder}
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        use_fast=True,
        **location_kwargs,
    )
    model_kwargs: dict[str, Any] = dict(location_kwargs)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    model = AutoModel.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
    if not isinstance(model, AlpaJudgeRB):
        raise TypeError(f"expected AlpaJudgeRB, got {type(model).__name__}")
    return model.eval().to(device), tokenizer


def _format_sample(tokenizer: Any, sample: Mapping[str, str]) -> str:
    """Format one sample using the exact reference-based training template."""
    for field in ("question", "answer", "student"):
        if field not in sample:
            raise KeyError(f"sample is missing {field!r}")
    return (
        f"{tokenizer.cls_token}\n"
        f"Question: {sample['question']}\n"
        f"Answer: {sample['answer']}\n"
        f"Student: {sample['student']}"
    )


@torch.inference_mode()
def score_reference_based(
    model: AlpaJudgeRB,
    tokenizer: Any,
    samples: Sequence[Mapping[str, str]],
    *,
    batch_size: int = 64,
) -> torch.Tensor:
    """Return one alignment probability per question/reference/candidate sample."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not samples:
        return torch.empty(0, dtype=torch.float32)

    device = next(model.parameters()).device
    score_batches: list[torch.Tensor] = []
    for start in range(0, len(samples), batch_size):
        texts = [
            _format_sample(tokenizer, sample) for sample in samples[start : start + batch_size]
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        logits = model(**inputs).logits
        score_batches.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(score_batches)
