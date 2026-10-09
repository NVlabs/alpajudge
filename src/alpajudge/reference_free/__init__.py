# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-free AlpaJudge model, processor, and inference helpers."""

from alpajudge.reference_free.inference import (
    ReferenceFreeScores,
    load_reference_free,
    score_reference_free,
)
from alpajudge.reference_free.model import (
    SUBSCORE_NAMES,
    AlpaJudgeRF,
    AlpaJudgeRFConfig,
    AlpaJudgeRFModelOutput,
    aggregate_hard_subscores,
    aggregate_soft_subscores,
)
from alpajudge.reference_free.processing import AlpaJudgeRFProcessor

__all__ = [
    "SUBSCORE_NAMES",
    "AlpaJudgeRF",
    "AlpaJudgeRFConfig",
    "AlpaJudgeRFModelOutput",
    "AlpaJudgeRFProcessor",
    "ReferenceFreeScores",
    "aggregate_hard_subscores",
    "aggregate_soft_subscores",
    "load_reference_free",
    "score_reference_free",
]
