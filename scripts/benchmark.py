"""Measure training throughput and inference latency for one run configuration. Needs no data.

  python scripts/benchmark.py --run reference
  python scripts/benchmark.py --config configs/smoke.yaml --steps 3
Prints and saves samples/s (training) and ms per question (inference, batch sizes 1 and 64).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vqa.config import load_config  # noqa: E402
from vqa.model import BridgedVQA, resolve_device  # noqa: E402

QUESTION, ANSWER = "What is the man holding in his hand?", "tennis racket"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="reference")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.run, args.set)
    device = resolve_device(cfg["device"])
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count()))
    model = BridgedVQA(cfg, device)
    v, bs = cfg["vision"], cfg["train"]["batch_size"]
    feats = torch.randn(bs, v["tokens"], v["dim"]).half()
    opt = torch.optim.AdamW(model.trainable())

    def train_step():
        model.bridge.train()
        with model.autocast():
            loss = model.loss(feats, [QUESTION] * bs, [ANSWER] * bs)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):  # warm-up
        train_step()
    sync(device)
    tic = time.time()
    for _ in range(args.steps):
        train_step()
    sync(device)
    result = {"run": args.run, "device": str(device), "batch_size": bs, "trainable_params": model.count_trainable(),
              "train_samples_per_sec": round(args.steps * bs / (time.time() - tic), 1)}

    model.bridge.eval()
    for n, reps in ((1, 10), (64, 3)):
        with model.autocast():
            model.answer(feats[:n] if n <= bs else feats.repeat(n // bs + 1, 1, 1)[:n], [QUESTION] * n)
            sync(device)
            tic = time.time()
            for _ in range(reps):
                model.answer(feats[:n] if n <= bs else feats.repeat(n // bs + 1, 1, 1)[:n], [QUESTION] * n)
            sync(device)
        result[f"latency_ms_per_question_bs{n}"] = round(1000 * (time.time() - tic) / (reps * n), 2)
    if device.type == "cuda":
        result["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(json.dumps(result, indent=1))
    out = args.out or os.path.join(cfg["paths"]["runs"], f"benchmark_{args.run}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
