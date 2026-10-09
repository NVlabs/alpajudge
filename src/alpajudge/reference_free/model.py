# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AlpaJudge-RF HF model implementation.

A serialized checkpoint carries its inner Qwen3-VL config, so loading one never reopens
the original Alpamayo1 / Qwen3-VL backbone it was distilled from; the backbone-resolution
helpers below enforce that self-containment.

AlpaJudge-RF is the reference-free judge: it scores one driving rationale from two
front-camera streams and the ego meta-action history/future, without ever seeing a
ground-truth rationale to compare against. The backbone is the inner VLM of a
driving-trained Alpamayo1 checkpoint (or base Qwen3-VL) with a binary classification
head over the last real token, predicting the five rubric sub-score logits::

    model = AlpaJudgeRF.from_pretrained(ckpt_path)
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3VLForConditionalGeneration,
)
from transformers.utils import ModelOutput

from alpajudge.reference_free.processing import (
    prompt_contract_sha256 as processor_prompt_contract_sha256,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


BACKBONE_KINDS = ("qwen3_vl", "alpamayo1")
DEFAULT_BACKBONE_SOURCE = "Qwen/Qwen3-VL-2B-Instruct"

_CAMERA_POLICY_DEFAULTS = {
    "include_camera_ids": None,
    "token_layout": None,
    "frame_label": None,
}
_PIXEL_POLICY_DEFAULTS = {
    "min_pixels": None,
    "max_pixels": None,
}


@dataclass(frozen=True)
class BackboneResolution:
    """Inner VLM configuration plus immutable source provenance."""

    kind: str
    source_model: str
    vlm_config: PretrainedConfig | Any
    source_config_sha256: str
    vlm_config_sha256: str
    native_camera_policy: dict[str, Any]
    native_pixel_policy: dict[str, int | None]


def normalize_source_model(source_model: str) -> str:
    """Resolve an existing local checkpoint while preserving Hugging Face identifiers."""
    expanded = Path(os.path.expanduser(source_model))
    if expanded.exists():
        return str(expanded.resolve())
    return source_model


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def config_sha256(config: PretrainedConfig | Any) -> str:
    """Hash the canonical semantic JSON representation of a model config.

    ``PretrainedConfig.to_dict()`` stamps the running ``transformers_version`` into the
    top-level dict (and ``_name_or_path`` records wherever the config was loaded from),
    so both are stripped: the digest must pin the config's *semantics*, not the library
    version or the load path — otherwise a dependency bump or a checkpoint copy would
    fail the recorded-hash comparison at load.
    """
    payload = config.to_dict() if hasattr(config, "to_dict") else config
    if isinstance(payload, dict):
        payload = {
            key: value
            for key, value in payload.items()
            if key not in ("transformers_version", "_name_or_path")
        }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, field_name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field_name} must be a 64-character SHA-256, got {value!r}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field_name} is not hexadecimal: {value!r}") from error


def _normalize_policy(
    policy: dict[str, Any] | None,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(defaults)
    if policy is not None:
        normalized.update(policy)
    return normalized


def _validate_inner_vlm(vlm_config: PretrainedConfig | Any, source_model: str) -> None:
    model_type = getattr(vlm_config, "model_type", None)
    if model_type not in (None, "qwen3_vl"):
        raise ValueError(
            f"AlpaJudgeRF requires an inner qwen3_vl VLM, but {source_model!r} "
            f"contains model_type={model_type!r}"
        )


def _infer_source_kind(source_config: PretrainedConfig | Any) -> str:
    model_type = getattr(source_config, "model_type", None)
    if model_type in BACKBONE_KINDS:
        return model_type
    # Lightweight mocked configs and legacy Qwen configs may omit model_type.
    if model_type is None and not hasattr(source_config, "vlm_config"):
        return "qwen3_vl"
    raise ValueError(
        f"unsupported AlpaJudgeRF backbone model_type={model_type!r}; "
        f"expected one of {BACKBONE_KINDS}"
    )


def _resolve_source(source_model: str, backbone_kind: str | None) -> BackboneResolution:
    source_model = normalize_source_model(source_model)
    source_config = AutoConfig.from_pretrained(source_model, trust_remote_code=True)
    detected_kind = _infer_source_kind(source_config)
    if backbone_kind is not None and backbone_kind != detected_kind:
        raise ValueError(
            f"requested backbone_kind={backbone_kind!r}, but {source_model!r} "
            f"contains {detected_kind!r}"
        )

    if detected_kind == "alpamayo1":
        if bool(getattr(source_config, "enable_expert", False)):
            raise ValueError(
                "Alpamayo1 AlpaJudgeRF backbones require enable_expert=False; "
                f"expert checkpoint rejected: {source_model}"
            )
        vlm_config = source_config.vlm_config
    else:
        vlm_config = source_config
    _validate_inner_vlm(vlm_config, source_model)

    camera_policy = {name: getattr(source_config, name, None) for name in _CAMERA_POLICY_DEFAULTS}
    pixel_policy = {name: getattr(source_config, name, None) for name in _PIXEL_POLICY_DEFAULTS}
    return BackboneResolution(
        kind=detected_kind,
        source_model=source_model,
        vlm_config=vlm_config,
        source_config_sha256=config_sha256(source_config),
        vlm_config_sha256=config_sha256(vlm_config),
        native_camera_policy=camera_policy,
        native_pixel_policy=pixel_policy,
    )


def resolve_backbone_config(
    source_model: str,
    vlm_config: PretrainedConfig | dict[str, Any] | Any | None,
    backbone_kind: str | None = None,
    source_config_sha256: str = "",
    vlm_config_sha256: str = "",
    native_camera_policy: dict[str, Any] | None = None,
    native_pixel_policy: dict[str, int | None] | None = None,
) -> BackboneResolution:
    """Resolve a new source, or restore a self-contained serialized config.

    A serialized AlpaJudge-RF checkpoint already carries its inner VLM config, so loading
    one never reopens the backbone checkpoint it was built from.
    """
    if backbone_kind is not None and backbone_kind not in BACKBONE_KINDS:
        raise ValueError(f"backbone_kind must be one of {BACKBONE_KINDS}, got {backbone_kind!r}")
    if vlm_config is None:
        return _resolve_source(source_model, backbone_kind)

    if isinstance(vlm_config, dict):
        vlm_config = AutoConfig.for_model(**vlm_config)
    source_model = normalize_source_model(source_model)
    resolved_kind = backbone_kind or "qwen3_vl"
    _validate_inner_vlm(vlm_config, source_model)
    computed_inner_hash = config_sha256(vlm_config)
    if vlm_config_sha256:
        _validate_sha256(vlm_config_sha256, "backbone_vlm_config_sha256")
        if vlm_config_sha256 != computed_inner_hash:
            raise ValueError(
                "serialized inner VLM config does not match backbone_vlm_config_sha256: "
                f"{computed_inner_hash} != {vlm_config_sha256}"
            )
    if source_config_sha256:
        _validate_sha256(source_config_sha256, "backbone_source_config_sha256")
    elif resolved_kind == "alpamayo1":
        raise ValueError("serialized alpamayo1 config is missing backbone_source_config_sha256")
    inner_hash = computed_inner_hash
    outer_hash = source_config_sha256 or inner_hash
    return BackboneResolution(
        kind=resolved_kind,
        source_model=source_model,
        vlm_config=vlm_config,
        source_config_sha256=outer_hash,
        vlm_config_sha256=inner_hash,
        native_camera_policy=_normalize_policy(
            native_camera_policy,
            _CAMERA_POLICY_DEFAULTS,
        ),
        native_pixel_policy=_normalize_policy(
            native_pixel_policy,
            _PIXEL_POLICY_DEFAULTS,
        ),
    )


def build_backbone_vlm(vlm_config: PretrainedConfig | Any):
    """Build the supported inner Qwen3-VL architecture from its config."""
    _validate_inner_vlm(vlm_config, "serialized inner VLM config")
    return Qwen3VLForConditionalGeneration(vlm_config)


DEFAULT_SCORE_CONTRACT = "vlm_rubric_c24f80fa_v1_no_direction_quality_diagnostic"
SUBSCORE_NAMES = (
    "action_matches_gt",
    "action_safe",
    "components_grounded",
    "components_complete",
    "component_causal",
)
ACTION_WEIGHT = 0.60
COMPONENT_WEIGHT = 0.40


class AlpaJudgeRFScores(NamedTuple):
    """Action, component, and overall rubric scores."""

    action: torch.Tensor
    component: torch.Tensor
    overall: torch.Tensor


def _validate_subscore_shape(subscores: torch.Tensor) -> None:
    """Require the canonical five scoring sub-scores on the final dimension."""
    if subscores.ndim == 0 or subscores.shape[-1] != len(SUBSCORE_NAMES):
        raise ValueError(
            f"subscores must have final dimension {len(SUBSCORE_NAMES)}, "
            f"got shape {tuple(subscores.shape)}"
        )


def aggregate_hard_subscores(subscores: torch.Tensor) -> AlpaJudgeRFScores:
    """Apply the rubric's binary aggregation.

    ``action_matches_gt`` hard-gates the action axis: a rationale that proposes the wrong
    action scores zero on action however safe it would have been.

    Args:
        subscores: ``(..., 5)`` hard rubric values in :data:`SUBSCORE_NAMES` order.

    Returns:
        Three tensors with shape ``subscores.shape[:-1]``.
    """
    _validate_subscore_shape(subscores)
    match, safe, grounded, complete, causal = subscores.unbind(dim=-1)
    action_mean = (match + safe) / 2.0
    action = torch.where(match == 0, torch.zeros_like(action_mean), action_mean)
    component = (grounded + complete + causal) / 3.0
    overall = ACTION_WEIGHT * action + COMPONENT_WEIGHT * component
    return AlpaJudgeRFScores(action, component, overall)


def aggregate_soft_subscores(probabilities: torch.Tensor) -> AlpaJudgeRFScores:
    """Compute the expected rubric scores from five probabilities.

    The sub-scores are treated as independent Bernoulli variables. The expected
    hard-gated action is therefore
    ``p_match * (1 + p_safe) / 2``.

    Args:
        probabilities: ``(..., 5)`` probabilities in :data:`SUBSCORE_NAMES` order.

    Returns:
        Expected action, component, and overall scores.
    """
    _validate_subscore_shape(probabilities)
    match, safe, grounded, complete, causal = probabilities.unbind(dim=-1)
    action = match * (1.0 + safe) / 2.0
    component = (grounded + complete + causal) / 3.0
    overall = ACTION_WEIGHT * action + COMPONENT_WEIGHT * component
    return AlpaJudgeRFScores(action, component, overall)


class AlpaJudgeRFConfig(PretrainedConfig):
    """Configuration for the AlpaJudgeRF model."""

    model_type = "alpajudge_rf"
    sub_configs = {"vlm_config": AutoConfig}
    # Saving a self-contained checkpoint must not resolve the default Hub backbone.
    has_no_defaults_at_init = True

    def _get_non_default_generation_parameters(self) -> dict[str, Any]:
        """The classifier wrapper has no legacy generation parameters.

        Transformers calls this hook while saving and its default implementation
        constructs ``self.__class__()``.  Our no-argument constructor represents
        a new Hub-backed model, so that behavior would perform an unrelated Hub
        lookup even though the serialized checkpoint already contains ``vlm_config``.
        """
        return {}

    def __init__(
        self,
        vlm_name_or_path: str | None = None,
        backbone_kind: str | None = None,
        backbone_source_model: str | None = None,
        backbone_source_config_sha256: str = "",
        backbone_vlm_config_sha256: str = "",
        backbone_native_camera_policy: dict[str, Any] | None = None,
        backbone_native_pixel_policy: dict[str, int | None] | None = None,
        num_labels: int | None = None,
        score_mode: str = "subscores",
        student_prompt_sha256: str = "",
        prompt_contract_sha256: str = "",
        score_contract: str = DEFAULT_SCORE_CONTRACT,
        max_length: int = 2048,
        padding_side: str = "left",
        freeze_vision: bool = True,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        dtype: str = "bfloat16",
        **kwargs: Any,
    ):
        """Initialize AlpaJudgeRF configuration.

        Args:
            vlm_name_or_path: Constructor-only alias for ``backbone_source_model``,
                accepted so older serialized configs keep loading; it is resolved into
                ``backbone_source_model`` and not stored separately.
            backbone_kind: Optional explicit ``qwen3_vl`` or ``alpamayo1``
                source type. The source config is authoritative when omitted.
            backbone_source_model: Exact source identifier or local checkpoint.
            backbone_source_config_sha256: Canonical source-config hash.
            backbone_vlm_config_sha256: Canonical inner-VLM-config hash.
            backbone_native_camera_policy: Camera formatting fields recorded
                from the source config.
            backbone_native_pixel_policy: Image resize policy recorded from
                the source processor.
            num_labels: Optional compatibility check for the output width; must be
                the five sub-scores when given.
            score_mode: Scoring-head contract identifier; the only supported value
                is ``"subscores"`` (the five canonical rubric predictions).
            student_prompt_sha256: Hash of the exact classifier prompt, if used.
            prompt_contract_sha256: Hash of the full frozen prompt contract (camera
                display names, frame labels, evidence template and separators, chat
                template flags). Defaults to the local processor contract so every config
                this class serializes carries the pin; ``AlpaJudgeRFProcessor`` enforces
                it at load.
            score_contract: Stable scoring-schema identifier.
            max_length: Maximum tokenizer sequence length (image + text tokens).
            padding_side: ``"left"`` (Qwen default) or ``"right"``. Drives the
                processor's collation and the value-head pooling index.
            freeze_vision: Freeze the Qwen vision tower (train the LLM + head).
            min_pixels: Min pixels for the image processor (caps tokens/frame).
            max_pixels: Max pixels for the image processor (caps tokens/frame).
            dtype: Parameter data type.
        """
        # Migration guards for retired knobs: PretrainedConfig.__init__ would absorb
        # them silently, and a checkpoint that actually used them (head_pool_tokens > 1
        # averaged trailing tokens into the value head) would load and score
        # differently with no error anywhere. The published checkpoint records the
        # no-op values, which are accepted and dropped (not re-serialized).
        retired_pool = kwargs.pop("head_pool_tokens", None)
        if retired_pool is not None and int(retired_pool) != 1:
            raise ValueError(
                "head_pool_tokens was removed; only the last-token read-out "
                f"(head_pool_tokens=1) is supported, got {retired_pool}"
            )
        retired_prompt_mode = kwargs.pop("prompt_mode", None)
        if retired_prompt_mode is not None and retired_prompt_mode != "compact_scoring":
            raise ValueError(
                "prompt_mode was removed; only the 'compact_scoring' prompt contract "
                f"is supported, got {retired_prompt_mode!r}"
            )
        super().__init__(dtype=dtype, **kwargs)
        if score_mode != "subscores":
            raise ValueError(f"score_mode must be 'subscores', got {score_mode!r}")
        if num_labels is not None and num_labels != len(SUBSCORE_NAMES):
            raise ValueError(f"num_labels must be {len(SUBSCORE_NAMES)}, got {num_labels}")
        resolution = resolve_backbone_config(
            source_model=backbone_source_model or vlm_name_or_path or DEFAULT_BACKBONE_SOURCE,
            vlm_config=getattr(self, "vlm_config", None),
            backbone_kind=backbone_kind,
            source_config_sha256=backbone_source_config_sha256,
            vlm_config_sha256=backbone_vlm_config_sha256,
            native_camera_policy=backbone_native_camera_policy,
            native_pixel_policy=backbone_native_pixel_policy,
        )
        self.backbone_kind = resolution.kind
        self.backbone_source_model = resolution.source_model
        self.backbone_source_config_sha256 = resolution.source_config_sha256
        self.backbone_vlm_config_sha256 = resolution.vlm_config_sha256
        self.backbone_native_camera_policy = resolution.native_camera_policy
        self.backbone_native_pixel_policy = resolution.native_pixel_policy
        self.vlm_config = resolution.vlm_config
        self.score_mode = score_mode
        self.num_labels = len(SUBSCORE_NAMES)
        self.student_prompt_sha256 = student_prompt_sha256
        self.prompt_contract_sha256 = prompt_contract_sha256 or processor_prompt_contract_sha256()
        self.score_contract = score_contract
        self.max_length = max_length
        self.padding_side = padding_side
        self.freeze_vision = freeze_vision
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Text hidden size feeds the scalar value head.
        self.hidden_size = self.vlm_config.text_config.hidden_size

        self.set_dtype(self.dtype)
        self.backbone_vlm_config_sha256 = config_sha256(self.vlm_config)

    def set_dtype(self, dtype: str | torch.dtype | None = None) -> None:
        """Propagate dtype into nested configs."""
        if dtype is None:
            dtype = self.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        # Keep the nested VLM config dtype in sync so the backbone builds in dtype.
        if hasattr(self, "vlm_config") and not isinstance(self.vlm_config, dict):
            self.vlm_config.dtype = dtype
            if hasattr(self.vlm_config, "text_config"):
                self.vlm_config.text_config.dtype = dtype
            if hasattr(self.vlm_config, "vision_config"):
                self.vlm_config.vision_config.dtype = dtype


@dataclass
class AlpaJudgeRFModelOutput(ModelOutput):
    """Raw sub-score logits, before the sigmoid and the rubric aggregation."""

    logits: torch.FloatTensor | None = None


class AlpaJudgeRF(PreTrainedModel):
    """Multimodal judge with a five-subscore binary head."""

    config_class = AlpaJudgeRFConfig
    # The attention impl is dispatched to the inner Qwen3-VL, which supports
    # all three; advertise support so ``from_pretrained(attn_implementation=...)``
    # is accepted on the wrapper and propagated down (see __init__).
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_attention_backend = True

    def __init__(self, config: AlpaJudgeRFConfig) -> None:
        """Initialize the model.

        Args:
            config: Model configuration.
        """
        config.set_dtype(config.dtype)
        super().__init__(config)
        logger.info(f"Initializing AlpaJudgeRF model with config: {self.config}")

        # Propagate the requested attention implementation into the nested Qwen
        # configs so the backbone builds with it (HF may also do this via
        # sub_configs, but be explicit).
        attn = getattr(config, "_attn_implementation", None)
        if attn is not None:
            config.vlm_config._attn_implementation = attn
            for sub in ("text_config", "vision_config"):
                if getattr(config.vlm_config, sub, None) is not None:
                    getattr(config.vlm_config, sub)._attn_implementation = attn
        config.backbone_vlm_config_sha256 = config_sha256(config.vlm_config)

        self.vlm = build_backbone_vlm(config.vlm_config)
        self.value_head = nn.Linear(config.hidden_size, config.num_labels)
        # Zero init so an untrained head predicts p=0.5 rather than an arbitrary
        # score; a distilled checkpoint overwrites both tensors when its weights load.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        if config.freeze_vision:
            self.vlm.model.visual.requires_grad_(False)

    def _pool_last_token(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Read the last real token's hidden state, shape ``(B, H)``.

        With left padding (Qwen default) the last real token is always at index
        ``-1``; with right padding it is at ``attention_mask.sum(-1) - 1``.
        """
        if self.config.padding_side == "left" or attention_mask is None:
            return hidden[:, -1, :]
        last_idx = attention_mask.long().sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]

    def forward(
        self,
        tokenized_data: dict[str, Any] | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> AlpaJudgeRFModelOutput:
        """Compute the rubric sub-score logits.

        Accepts either a ``tokenized_data`` dict or the same keys as flat keyword
        tensors — the batch shape ``AlpaJudgeRFProcessor`` produces:
        ``input_ids``/``attention_mask`` padded ``(B, L)`` on the configured
        ``padding_side``, plus Qwen-style ``pixel_values``/``image_grid_thw``.

        Args:
            tokenized_data: Dict with ``input_ids``, ``attention_mask``, and
                (optionally) ``pixel_values`` / ``image_grid_thw`` /
                ``position_ids``.
            input_ids: Token IDs ``(B, L)`` — image-placeholder tokens expanded.
            attention_mask: Attention mask ``(B, L)``.
            pixel_values: Flattened image patches for the Qwen vision tower.
            image_grid_thw: Per-image ``(T, H, W)`` patch grid, shape ``(N, 3)``.
            position_ids: Optional mRoPE position ids (None -> computed by Qwen).
        """
        if tokenized_data is not None:
            input_ids = tokenized_data.get("input_ids", input_ids)
            attention_mask = tokenized_data.get("attention_mask", attention_mask)
            pixel_values = tokenized_data.get("pixel_values", pixel_values)
            image_grid_thw = tokenized_data.get("image_grid_thw", image_grid_thw)
            position_ids = tokenized_data.get("position_ids", position_ids)

        model_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values
            model_kwargs["image_grid_thw"] = image_grid_thw
        if position_ids is not None:
            model_kwargs["position_ids"] = position_ids

        # Inner Qwen3VLModel: merges vision embeddings into the text stream and
        # returns last_hidden_state (skips the vocab lm_head — not needed here).
        outputs = self.vlm.model(**model_kwargs)
        hidden = outputs.last_hidden_state  # (B, L, H)

        pooled = self._pool_last_token(hidden, attention_mask)
        logits = self.value_head(pooled.to(self.value_head.weight.dtype))
        return AlpaJudgeRFModelOutput(logits=logits)


AutoConfig.register("alpajudge_rf", AlpaJudgeRFConfig)
AutoModel.register(AlpaJudgeRFConfig, AlpaJudgeRF)
