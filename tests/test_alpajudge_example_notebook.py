# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone PAI inference notebook."""

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "examples" / "alpajudge_pai.ipynb"


def _load_notebook() -> dict:
    """Load the notebook JSON."""
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _notebook_source(notebook: dict) -> str:
    """Join all notebook source cells."""
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_is_clean_and_python_source_compiles() -> None:
    """Keep the checked-in notebook deterministic and free of generated output."""
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")


def test_notebook_uses_public_data_and_both_checkpoint_subfolders() -> None:
    """The example should exercise both public model variants on PhysicalAI-AV data."""
    source = _notebook_source(_load_notebook())

    for expected in (
        "PhysicalAIAVDatasetInterface",
        "CAMERA_FRONT_WIDE_120FOV",
        "CAMERA_FRONT_TELE_30FOV",
        "from IPython.display import display",
        "display(fig)",
        "snapshot_download",
        'MODEL_ID = "nvidia/alpajudge"',
        'MODEL_REVISION = os.environ.get("ALPAJUDGE_MODEL_REVISION", "main")',
        '"reference-based/*"',
        '"reference-free/*"',
        "load_reference_based",
        "score_reference_based",
        "load_reference_free",
        "score_reference_free",
    ):
        assert expected in source


class _FakeCamera:
    """Return timestamp-coded RGB frames."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.requested_timestamps: np.ndarray | None = None

    def decode_images_from_timestamps(
        self, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.requested_timestamps = timestamps.copy()
        frames = np.full((len(timestamps), 2, 4, 3), self.value, dtype=np.uint8)
        return frames, timestamps + self.value


class _FakeAVDI:
    """Expose the camera feature surface used by the notebook."""

    def __init__(self) -> None:
        self.features = SimpleNamespace(
            CAMERA=SimpleNamespace(
                CAMERA_FRONT_WIDE_120FOV="camera/front-wide",
                CAMERA_FRONT_TELE_30FOV="camera/front-tele",
            )
        )
        self.cameras = {
            "camera/front-wide": _FakeCamera(11),
            "camera/front-tele": _FakeCamera(22),
        }

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool) -> _FakeCamera:
        assert clip_id == "clip"
        assert maybe_stream is True
        return self.cameras[feature]


def test_pai_loader_builds_the_frozen_six_frame_contract() -> None:
    """Decode wide then tele frames at minus one, minus one-half, and zero seconds."""
    notebook = _load_notebook()
    loader_cell = next(
        cell for cell in notebook["cells"] if "def load_pai_frames" in "".join(cell["source"])
    )
    namespace = {"np": np, "torch": torch}
    exec(compile("".join(loader_cell["source"]), "load_pai_frames", "exec"), namespace)
    avdi = _FakeAVDI()

    wide, tele, decoded_timestamps = namespace["load_pai_frames"](avdi, "clip", 5_100_000)

    expected_timestamps = np.asarray([4_100_000, 4_600_000, 5_100_000], dtype=np.int64)
    np.testing.assert_array_equal(
        avdi.cameras["camera/front-wide"].requested_timestamps, expected_timestamps
    )
    np.testing.assert_array_equal(
        avdi.cameras["camera/front-tele"].requested_timestamps, expected_timestamps
    )
    assert wide.shape == tele.shape == (3, 3, 2, 4)
    assert wide.dtype == tele.dtype == torch.uint8
    assert wide.unique().item() == 11
    assert tele.unique().item() == 22
    assert decoded_timestamps == {
        "front_wide": [4_100_011, 4_600_011, 5_100_011],
        "front_tele": [4_100_022, 4_600_022, 5_100_022],
    }


def test_readme_and_dev_environment_expose_the_notebook() -> None:
    """The documented setup should install and link the notebook dependencies."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["dependency-groups"]["dev"]

    assert "[PAI inference notebook](examples/alpajudge_pai.ipynb)" in readme
    assert "[`nvidia/alpajudge`](https://huggingface.co/nvidia/alpajudge)" in readme
    assert '"  Lateral: [-1.6,0.0) Go Straight\\n"' in readme
    for prefix in (
        "ipykernel",
        "ipywidgets",
        "jupyterlab",
        "matplotlib",
        "numpy",
        "physical-ai-av",
    ):
        assert any(dependency.startswith(prefix) for dependency in dependencies)
