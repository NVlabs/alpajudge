# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone inference package for the AlpaJudge model family."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpajudge.reference_based import AlpaJudgeRB, AlpaJudgeRBConfig
    from alpajudge.reference_free import AlpaJudgeRF, AlpaJudgeRFConfig

__all__ = ["AlpaJudgeRB", "AlpaJudgeRBConfig", "AlpaJudgeRF", "AlpaJudgeRFConfig"]


def __getattr__(name: str):
    """Lazily expose model classes without importing Torch at package import time."""
    if name in {"AlpaJudgeRB", "AlpaJudgeRBConfig"}:
        from alpajudge import reference_based

        return getattr(reference_based, name)
    if name in {"AlpaJudgeRF", "AlpaJudgeRFConfig"}:
        from alpajudge import reference_free

        return getattr(reference_free, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
