# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-based AlpaJudge model and inference helpers."""

from alpajudge.reference_based.inference import load_reference_based, score_reference_based
from alpajudge.reference_based.model import (
    AlpaJudgeRB,
    AlpaJudgeRBConfig,
    AlpaJudgeRBModelOutput,
)

__all__ = [
    "AlpaJudgeRB",
    "AlpaJudgeRBConfig",
    "AlpaJudgeRBModelOutput",
    "load_reference_based",
    "score_reference_based",
]
