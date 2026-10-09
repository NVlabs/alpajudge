# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-surface tests for the AlpaJudge package."""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def test_top_level_import_is_lightweight() -> None:
    """Importing the package metadata surface does not eagerly import Torch."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, alpajudge; print('torch' in sys.modules)",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "False"
