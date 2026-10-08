# Grounded VQA with a lightweight bridge between frozen models

CS-747 course project, Topic 30 (Multimodal Foundation Models for Visual Question Answering). Roll No. 23DCS004.

A frozen SigLIP 2 Large vision encoder and a frozen Llama 3.2 1B language model are joined by a small
trainable bridge (about 10M parameters). Only the bridge is trained, on a subset of VQA v2. The evaluation
reports answer accuracy and four measures of whether the answers actually depend on the image.

```
image -> SigLIP 2 Large (frozen, features cached) -> bridge (trained) -> Llama 3.2 1B (frozen) -> answer
```

## Layout

| Path | What it holds |
| --- | --- |
| `configs/default.yaml` | Every setting of the reference run |
| `configs/runs.yaml` | The named runs (ablations) as overrides of the default |
| `configs/smoke.yaml` | CPU smoke test with a tiny open language model |
| `vqa/bridges.py` | Query bridge and pooled-MLP bridge |
| `vqa/model.py` | Bridge + frozen language model: training loss and answer generation |
| `vqa/evaluate.py` | VQA accuracy and the groundedness measures |
| `vqa/vqa_eval.py` | Answer normalisation, adapted from the official VQA evaluation script |
| `vqa/data.py` | Split files, feature cache, batching |
| `scripts/` | Command-line steps, listed below |
| `data/`, `cache/`, `runs/` | Downloads, cached features, results (created by the scripts) |
| `docs/proposal.md` | The project proposal, current with this code |
| `tests/` | Checks for the accuracy function |

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt     # use a CUDA build of torch for the GPU session
huggingface-cli login                                 # Llama 3.2 is gated: request access on its model page first
```

## Steps without a GPU

```powershell
python scripts/prepare_data.py all        # annotations, splits, GroundProbe, and only the needed images
python scripts/download_models.py         # SigLIP 2 Large and Llama 3.2 1B into the Hugging Face cache
```

Smoke test on CPU (checks the code path end to end, not the accuracy):

```powershell
python scripts/download_models.py --config configs/smoke.yaml
python scripts/cache_features.py --config configs/smoke.yaml
python scripts/train.py --config configs/smoke.yaml --run reference
python scripts/make_report.py --config configs/smoke.yaml
```

## GPU session

One command runs everything inside a time budget (195 minutes by default):

```powershell
python scripts/run_session.py
```

It measures training speed, picks the largest training set that fits the budget, caches the vision features,
checks three learning rates, trains the five planned runs, and writes tables and figures to `runs/report/`.
Finished steps are skipped on a rerun. Add `--extras` to use spare time for the extra runs.

Single steps can also be run by hand:

```powershell
python scripts/benchmark.py --run reference
python scripts/cache_features.py
python scripts/train.py --run blind
python scripts/make_report.py
```

## Runs

| Run | Change from the reference | Question it answers |
| --- | --- | --- |
| `reference` | Query bridge, 32 visual tokens, frozen language model | Baseline and main model |
| `blind` | Image features replaced by zeros | Does the model use the image? |
| `lora` | LoRA on the language model | Frozen versus lightly fine-tuned |
| `tokens16` | 16 visual tokens | How many visual tokens are needed? |
| `mlp16` | Pooled MLP bridge, 16 tokens | Which bridge design is better at equal tokens? |

Extra runs: `seed1`, `seed2`, `tokens64`, `mlp64`.

## What is measured

- **VQA accuracy** on 5,000 held-out validation images: overall and per answer type.
- **Blind gap:** reference accuracy minus the blind twin's.
- **Image swap:** accuracy and share of changed answers when every image is replaced by another one.
- **Complementary pairs:** share of VQA v2 pairs (same question, two images, different answers) answered correctly on both.
- **GroundProbe:** balanced existence, counting and left/right questions generated from COCO boxes.

## Own work and reused components

Reused: pretrained SigLIP 2 and Llama 3.2 weights (Hugging Face Transformers), the PEFT library for LoRA, and the
answer normalisation of the official VQA evaluation script. Everything else in this repository is own code.
