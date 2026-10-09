# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen prompt construction and image processing for AlpaJudge-RF inference."""

import hashlib
from typing import Any

import torch
from transformers import PretrainedConfig, Qwen3VLProcessor

COMPACT_CLASSIFIER_INSTRUCTION = """Task: score one autonomous-driving rationale from 0 to 1.
Use the synchronized labeled camera frames as visual truth and read GT_ACTION HISTORY together
with FUTURE. Internally judge five scoring checks: action matches ground truth, action is safe,
decisive components are grounded, components are complete, and causality is correct. Action is
zero when the action does not match; otherwise it is the mean of match and safety. Component is
the mean of grounded, complete, and causal. Overall is 0.60 * action + 0.40 * component.
Rationale coherence is diagnostic only and does not gate the score."""

CAMERA_DISPLAY_NAMES = ("FRONT-WIDE 120°FOV", "FRONT-TELE 30°FOV")
FRAME_OFFSETS_SECONDS = (-1.0, -0.5, 0.0)
GT_ACTION_HISTORY_HEADER = "GT_ACTION HISTORY (past):"
GT_ACTION_FUTURE_HEADER = "GT_ACTION FUTURE (from now):"
PREDICTED_RATIONALE_HEADER = "PREDICTED RATIONALE:"
PROMPT_TEMPLATE_FLAGS = {"add_generation_prompt": True, "add_vision_id": False}


def prompt_sha256(prompt: str) -> str:
    """Return the SHA-256 of a prompt body."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def frame_label(offset_seconds: float) -> str:
    """Return the frozen per-frame time label."""
    label = f"t={offset_seconds:.1f}s"
    return f"{label} (current)" if offset_seconds == 0.0 else label


def build_prompt_content(
    *,
    meta_action_history: str,
    meta_action_future: str,
    rationale: str,
    include_images: bool = True,
) -> list[dict[str, Any]]:
    """Build the exact user-message content used to train AlpaJudge-RF."""
    frame_content: list[dict[str, Any]] = []
    if include_images:
        for camera in CAMERA_DISPLAY_NAMES:
            for offset in FRAME_OFFSETS_SECONDS:
                frame_content.extend(
                    (
                        {"type": "text", "text": f"{camera} {frame_label(offset)}:"},
                        {"type": "image"},
                    )
                )

    evidence = (
        f"{GT_ACTION_HISTORY_HEADER}\n"
        f"{meta_action_history}\n\n"
        f"{GT_ACTION_FUTURE_HEADER}\n"
        f"{meta_action_future}\n\n"
        f"{PREDICTED_RATIONALE_HEADER}\n"
        f"{rationale}"
    )
    return [
        {"type": "text", "text": COMPACT_CLASSIFIER_INSTRUCTION},
        *frame_content,
        {"type": "text", "text": evidence},
    ]


def prompt_contract_sha256() -> str:
    """Hash the frozen prompt content and chat-template flags."""
    content = build_prompt_content(
        meta_action_history="<contract:meta_action_history>",
        meta_action_future="<contract:meta_action_future>",
        rationale="<contract:cot>",
    )
    parts = [block["text"] if block["type"] == "text" else "<image>" for block in content]
    parts.extend(f"{name}={value}" for name, value in sorted(PROMPT_TEMPLATE_FLAGS.items()))
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _normalize_frames(frames: torch.Tensor, name: str) -> torch.Tensor:
    """Validate one camera's three CHW frames and scale uint8 input to ``[0, 1]``."""
    if not torch.is_tensor(frames) or frames.ndim != 4:
        raise ValueError(f"{name} must be a [3,C,H,W] tensor")
    if len(frames) != len(FRAME_OFFSETS_SECONDS):
        raise ValueError(
            f"{name} must contain {len(FRAME_OFFSETS_SECONDS)} frames ordered oldest-first"
        )
    if frames.dtype == torch.uint8:
        return frames.float().div(255.0)
    if frames.numel() and (frames.min().item() < 0.0 or frames.max().item() > 1.0):
        raise ValueError(f"float {name} must be in [0, 1]")
    return frames.float()


class AlpaJudgeRFProcessor:
    """Prepare the frozen six-frame AlpaJudge-RF input contract."""

    def __init__(self, processor: Any, config: PretrainedConfig) -> None:
        """Bind a Hugging Face processor to one AlpaJudge-RF config."""
        self.processor = processor
        self.config = config
        self._validate_contract()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        config: PretrainedConfig | None = None,
        subfolder: str | None = None,
        **kwargs: Any,
    ) -> "AlpaJudgeRFProcessor":
        """Load the Qwen processor shipped beside an AlpaJudge-RF checkpoint."""
        location_kwargs = {} if subfolder is None else {"subfolder": subfolder}
        if config is None:
            from alpajudge.reference_free.model import AlpaJudgeRFConfig

            config = AlpaJudgeRFConfig.from_pretrained(
                pretrained_model_name_or_path,
                **location_kwargs,
            )
        for name in ("min_pixels", "max_pixels"):
            value = getattr(config, name, None)
            if value is not None:
                kwargs.setdefault(name, value)
        processor = Qwen3VLProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            **location_kwargs,
            **kwargs,
        )
        return cls(processor, config)

    def _validate_contract(self) -> None:
        expected_prompt = getattr(self.config, "student_prompt_sha256", "")
        actual_prompt = prompt_sha256(COMPACT_CLASSIFIER_INSTRUCTION)
        if expected_prompt and expected_prompt != actual_prompt:
            raise ValueError(
                "local scoring instruction differs from checkpoint: "
                f"{actual_prompt} != {expected_prompt}"
            )
        expected_contract = getattr(self.config, "prompt_contract_sha256", "")
        actual_contract = prompt_contract_sha256()
        if expected_contract and expected_contract != actual_contract:
            raise ValueError(
                "local prompt contract differs from checkpoint: "
                f"{actual_contract} != {expected_contract}"
            )

    def __call__(
        self,
        *,
        front_wide_frames: torch.Tensor,
        front_tele_frames: torch.Tensor,
        meta_action_history: str,
        meta_action_future: str,
        rationale: str,
    ) -> dict[str, torch.Tensor]:
        """Tokenize one rationale and its synchronized camera/action evidence."""
        images = torch.cat(
            (
                _normalize_frames(front_wide_frames, "front_wide_frames"),
                _normalize_frames(front_tele_frames, "front_tele_frames"),
            )
        )
        content = build_prompt_content(
            meta_action_history=meta_action_history,
            meta_action_future=meta_action_future,
            rationale=rationale,
        )
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            **PROMPT_TEMPLATE_FLAGS,
        )
        inputs = dict(
            self.processor(
                text=text,
                images=list(images),
                videos=None,
                padding=False,
                return_tensors="pt",
                do_rescale=False,
            )
        )
        if len(inputs.get("image_grid_thw", ())) != len(images):
            raise ValueError("processor image count differs from the six labeled frames")
        input_length = inputs["input_ids"].shape[-1]
        max_length = getattr(self.config, "max_length", None)
        if max_length is not None and input_length > max_length:
            raise ValueError(
                f"sample tokenized to {input_length} tokens, exceeding max_length={max_length}"
            )
        return inputs
