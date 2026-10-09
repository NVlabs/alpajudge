# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AlpaJudgeRB HF model implementation.

AlpaJudgeRB wraps an ``AutoModelForSequenceClassification`` (e.g. DeBERTa-v3-large
from ``wayveai/Lingo-Judge``) for binary text-similarity classification.

Two ways to create the model:

1. From scratch using a base HF checkpoint::

    config = AlpaJudgeRBConfig(pretrained_model_name="wayveai/Lingo-Judge")
    model = AlpaJudgeRB.from_pretrained_base(config)

2. From a saved AlpaJudgeRB checkpoint::

    model = AlpaJudgeRB.from_pretrained(ckpt_path)

Released checkpoints that still serialize ``model_type = "alpajudge"`` should be loaded
through ``AutoModel`` after importing this module so the legacy reader is registered.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import ModelOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AlpaJudgeRBConfig(PretrainedConfig):
    """Configuration for AlpaJudge-RB."""

    model_type = "alpajudge_rb"
    sub_configs = {"base_config": AutoConfig}
    # PretrainedConfig.to_diff_dict() otherwise constructs the default config, which would
    # download ``pretrained_model_name`` merely to log or serialize an already complete config.
    has_no_defaults_at_init = True

    def __init__(
        self,
        pretrained_model_name: str = "wayveai/Lingo-Judge",
        num_labels: int = 1,
        max_length: int = 256,
        dtype: str = "float32",
        base_config: dict[str, Any] | PretrainedConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize AlpaJudgeRB configuration.

        Args:
            pretrained_model_name: HuggingFace model identifier for the base classifier.
            num_labels: Number of output labels (1 for binary via BCEWithLogitsLoss).
            max_length: Maximum tokenizer sequence length.
            dtype: Parameter data type.
            base_config: Serialized or instantiated base classifier config. The source model is
                queried only when this is absent.
        """
        super().__init__(dtype=dtype, **kwargs)
        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.max_length = max_length

        if base_config is None:
            self.base_config = AutoConfig.from_pretrained(
                pretrained_model_name, num_labels=num_labels
            )
        elif isinstance(base_config, dict):
            self.base_config = AutoConfig.for_model(**base_config)
        else:
            self.base_config = base_config

        self.set_dtype(self.dtype)

    def set_dtype(self, dtype: str | torch.dtype | None = None) -> None:
        """Propagate dtype into nested configs."""
        if dtype is None:
            dtype = self.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype

    def _get_non_default_generation_parameters(self) -> dict[str, Any]:
        """Return no generation overrides because AlpaJudge-RB is a classifier."""
        return {}


@dataclass
class AlpaJudgeRBModelOutput(ModelOutput):
    """Output of the AlpaJudgeRB model."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None


class AlpaJudgeRB(PreTrainedModel):
    """Binary text-similarity judge backed by a sequence classifier."""

    config_class = AlpaJudgeRBConfig

    def __init__(self, config: AlpaJudgeRBConfig) -> None:
        """Initialize the model.

        Args:
            config: Model configuration.
        """
        config.set_dtype(config.dtype)
        super().__init__(config)
        logger.info(f"Initializing AlpaJudgeRB model with config: {self.config}")
        self.classifier = AutoModelForSequenceClassification.from_config(config.base_config)

    @classmethod
    def from_pretrained_base(
        cls,
        config: AlpaJudgeRBConfig,
        device: torch.device | str | int = "cpu",
    ) -> "AlpaJudgeRB":
        """Create an AlpaJudgeRB model with weights loaded from the base HF checkpoint.

        Args:
            config: AlpaJudgeRB configuration.
            device: Target device.
        """
        with torch.device("meta"):
            model = cls(config=config)

        pretrained = AutoModelForSequenceClassification.from_pretrained(
            config.pretrained_model_name,
            num_labels=config.num_labels,
            device_map=device,
        )
        model.classifier = pretrained
        logger.info(f"Loaded base classifier weights from {config.pretrained_model_name}")
        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.FloatTensor | None = None,
        **kwargs: Any,
    ) -> AlpaJudgeRBModelOutput:
        """Forward pass computing classification loss.

        Args:
            input_ids: Token IDs of shape ``(B, L)``.
            attention_mask: Attention mask of shape ``(B, L)``.
            labels: Binary labels of shape ``(B,)`` with values in ``{0, 1}``.
        """
        outputs = self.classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits.squeeze(-1)  # (B,)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), labels.float()
            )

        return AlpaJudgeRBModelOutput(loss=loss, logits=logits)


class _LegacyAlpaJudgeConfig(AlpaJudgeRBConfig):
    """Read legacy checkpoints serialized before the ``alpajudge_rb`` rename."""

    model_type = "alpajudge"

    def to_export_config(self) -> AlpaJudgeRBConfig:
        """Return the canonical config used when re-exporting a legacy checkpoint."""
        config_dict = self.to_dict()
        config_dict.pop("model_type", None)
        config_dict["architectures"] = ["AlpaJudgeRB"]
        return AlpaJudgeRBConfig(**config_dict)


class _LegacyAlpaJudge(AlpaJudgeRB):
    """AutoModel adapter for checkpoints whose config uses the legacy model type."""

    config_class = _LegacyAlpaJudgeConfig


AutoConfig.register("alpajudge_rb", AlpaJudgeRBConfig)
AutoModel.register(AlpaJudgeRBConfig, AlpaJudgeRB)
AutoConfig.register("alpajudge", _LegacyAlpaJudgeConfig)
AutoModel.register(_LegacyAlpaJudgeConfig, _LegacyAlpaJudge)
