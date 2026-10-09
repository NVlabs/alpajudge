# AlpaJudge

Inference package for the reference-based and reference-free AlpaJudge checkpoints.
The package contains the Hugging Face model classes, the exact scoring contracts, and small
inference helpers. It does not include training code or data pipelines.

## Table of Contents

- [Setup](#setup)
- [Checkpoints](#checkpoints)
- [PAI example notebook](#pai-example-notebook)
- [Reference-based inference](#reference-based-inference)
- [Reference-free inference](#reference-free-inference)
- [Public API](#public-api)
- [Tests](#tests)
- [Limitations](#limitations)
- [License](#license)

## Setup

Python 3.12 and PyTorch 2.8 are required. From the repository root:

```bash
uv sync --locked --dev
source .venv/bin/activate
hf auth login --token "$HF_TOKEN"
```

The Hugging Face account must have access to the model repository.

## Checkpoints

Both checkpoints live in one Hugging Face repository:

| Variant      | Hugging Face subfolder | Input                                                                  | Output                                         |
| ------------ | ---------------------- | ---------------------------------------------------------------------- | ---------------------------------------------- |
| AlpaJudge-RB | `reference-based`      | Question, reference answer, candidate answer                           | One alignment probability                      |
| AlpaJudge-RF | `reference-free`       | Six camera frames, ego meta-action history/future, candidate rationale | Five rubric probabilities and aggregate scores |

Both checkpoints are hosted at
[`nvidia/alpajudge`](https://huggingface.co/nvidia/alpajudge).

## PAI example notebook

The [PAI inference notebook](examples/alpajudge_pai.ipynb) downloads both checkpoint subfolders,
streams a fixed two-camera sample from the NVIDIA
[PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles),
and runs one RB and one RF score on a CUDA GPU:

```bash
uv run jupyter lab examples/alpajudge_pai.ipynb
```

The default sample and PAI revision are pinned in the notebook. Set `ALPAJUDGE_MODEL_CACHE` or
`ALPAJUDGE_PAI_CACHE` before starting Jupyter when the default Hugging Face cache is too small.
`ALPAJUDGE_MODEL_REVISION`, `ALPAJUDGE_PAI_REVISION`, `ALPAJUDGE_PAI_CLIP_ID`, and
`ALPAJUDGE_PAI_T0_US` provide explicit overrides.

PhysicalAI-AV supplies the images, while the example rationale and meta-action windows are manually
authored for the displayed construction scene. Replace all text fields together when using another
moment. The resulting values verify the inference path; they are not benchmark metrics or
recommended thresholds.

## Reference-based inference

AlpaJudge-RB is text-only. It compares each candidate answer against the supplied reference:

```python
from alpajudge.reference_based import load_reference_based, score_reference_based

model, tokenizer = load_reference_based(device="cuda")
scores = score_reference_based(
    model,
    tokenizer,
    [
        {
            "question": "What is the reasoning process of driving?",
            "answer": "The ego vehicle should stop and yield to the pedestrian.",
            "student": "The ego vehicle should slow down for the pedestrian and stop.",
        }
    ],
)
print(scores[0].item())
```

For driving CoT alignment, use the canonical question shown above, put the ground-truth CoT in
`answer`, and put the generated CoT in `student`. General VQA should retain the original question.

## Reference-free inference

AlpaJudge-RF consumes three front-wide frames and three front-tele frames at
`[-1.0, -0.5, 0.0]` seconds, ordered oldest-first within each camera. Each tensor must have shape
`[3, C, H, W]` and use either uint8 pixels or floating-point pixels in `[0, 1]`.

```python
import torch

from alpajudge.reference_free import load_reference_free, score_reference_free

model, processor = load_reference_free(device="cuda")

# Replace these tensors with synchronized decoded frames.
front_wide_frames = torch.zeros((3, 3, 720, 1280), dtype=torch.uint8)
front_tele_frames = torch.zeros((3, 3, 720, 1280), dtype=torch.uint8)

scores = score_reference_free(
    model,
    processor,
    front_wide_frames=front_wide_frames,
    front_tele_frames=front_tele_frames,
    meta_action_history=(
        "  Longitudinal: [-1.6,0.0) Gentle Deceleration\n"
        "  Lateral: [-1.6,0.0) Go Straight\n"
        "  Lane: [-1.6,0.0) Lane Keep"
    ),
    meta_action_future=(
        "  Longitudinal: [0.0,6.4) Gentle Deceleration\n"
        "  Lateral: [0.0,6.4) Go Straight\n"
        "  Lane: [0.0,6.4) Lane Keep"
    ),
    rationale=(
        "Continue straight and slow slightly behind the white sedan because the work zone "
        "narrows the usable lane."
    ),
)
print(scores.overall)
print(scores.subscores)
```

The five probabilities are `action_matches_gt`, `action_safe`, `components_grounded`,
`components_complete`, and `component_causal`. The soft aggregation is:

```text
action    = p_match * (1 + p_safe) / 2
component = mean(p_grounded, p_complete, p_causal)
overall   = 0.6 * action + 0.4 * component
```

The processor verifies the checkpoint's instruction and full prompt-contract hashes before
scoring.

## Public API

```python
from alpajudge.reference_based import (
    AlpaJudgeRB,
    AlpaJudgeRBConfig,
    load_reference_based,
    score_reference_based,
)
from alpajudge.reference_free import (
    AlpaJudgeRF,
    AlpaJudgeRFConfig,
    AlpaJudgeRFProcessor,
    ReferenceFreeScores,
    load_reference_free,
    score_reference_free,
)
```

Pass a local checkpoint directory and `subfolder=None` to either loader when working without the
Hugging Face repository.

## Tests

```bash
uv run pytest
```

The tests cover import isolation, legacy RB checkpoint compatibility, the frozen RF prompt hash,
input validation, and score aggregation.

## Limitations

- AlpaJudge-RB requires a reference answer and measures agreement with that reference rather than
  independent scene correctness.
- AlpaJudge-RF requires correctly synchronized frames and preformatted meta-action interval text.
  The core package does not include a video decoder or meta-action extraction pipeline; the example
  notebook uses the optional development dependency `physical-ai-av` for its fixed input.
- The RF checkpoint is intended for GPU inference. A complete model load needs approximately
  5 GB for weights before runtime activations.

## License

The source code in this repository is licensed under the [Apache License 2.0](LICENSE).
The model weights hosted at [`nvidia/alpajudge`](https://huggingface.co/nvidia/alpajudge) are
governed by the license on that model card. Third-party dependencies are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
