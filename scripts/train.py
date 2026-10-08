"""Train one run and evaluate it.

  python scripts/train.py --run reference
  python scripts/train.py --run mlp16 --set train.lr=1e-4
  python scripts/train.py --config configs/smoke.yaml --run reference     # CPU smoke test
Writes runs/<name>/: bridge.pt, results.json, predictions.json, train_log.jsonl.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vqa.config import load_config  # noqa: E402
from vqa.data import FeatureStore, VQASplit  # noqa: E402
from vqa.evaluate import evaluate_all, predict, vqa_scores  # noqa: E402
from vqa.model import BridgedVQA, resolve_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="reference")
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.run, args.set)
    out = os.path.join(cfg["paths"]["runs"], args.name or args.run)
    os.makedirs(out, exist_ok=True)
    seed, t, d = cfg["seed"], cfg["train"], cfg["data"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(cfg["device"])
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count()))

    store = FeatureStore(cfg["paths"]["cache"], d["preload"])
    train = VQASplit(cfg["paths"]["splits"], "train", store, d["train_images"])
    dev = VQASplit(cfg["paths"]["splits"], "dev", store, d["dev_images"])
    model = BridgedVQA(cfg, device)
    print(f"[{args.run}] {len(train)} train questions, {model.count_trainable() / 1e6:.2f}M trainable, device {device}", flush=True)

    opt = torch.optim.AdamW(model.trainable(), weight_decay=t["weight_decay"])
    total = t["epochs"] * math.ceil(len(train) / t["batch_size"])
    warm = max(1, int(t["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    eval_at = {round(total * (i + 1) / t["dev_evals"]) for i in range(t["dev_evals"])}
    rng = random.Random(seed)
    log = open(os.path.join(out, "train_log.jsonl"), "w")
    curve, step, seen, train_time, stopped = [], 0, 0, 0.0, False

    for epoch in range(t["epochs"]):
        for idx in train.batches(t["batch_size"], shuffle=True, seed=seed, epoch=epoch):
            tic = time.time()
            model.bridge.train()
            records = [train.records[i] for i in idx]
            with model.autocast():
                loss = model.loss(train.features(idx), [r["q"] for r in records], [rng.choice(r["answers"]) for r in records])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], t["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step, seen = step + 1, seen + len(idx)
            train_time += time.time() - tic
            if step % t["log_every"] == 0 or step == total:
                row = {"step": step, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0],
                       "samples_per_sec": round(seen / train_time, 1)}
                print(f"  step {step}/{total}  loss {row['loss']:.3f}  {row['samples_per_sec']:.0f} samples/s", flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
            if step in eval_at:
                preds = predict(model, dev, cfg["eval"]["batch_size"], limit=t["dev_questions"])
                acc = vqa_scores(dev.records[:len(preds)], preds)["overall"]
                curve.append({"step": step, "dev_acc": acc})
                print(f"  dev accuracy at step {step}: {acc}", flush=True)
            if t["max_minutes"] and train_time / 60 > t["max_minutes"]:
                stopped = True
                break
        if stopped:
            print(f"  stopped early at step {step}: train.max_minutes reached", flush=True)
            break

    torch.save(model.trainable_state(), os.path.join(out, "bridge.pt"))
    results = {
        "run": args.name or args.run, "config": cfg, "trainable_params": model.count_trainable(),
        "train_questions": len(train), "steps": step, "planned_steps": total, "stopped_early": stopped,
        "train_minutes": round(train_time / 60, 2), "samples_per_sec": round(seen / max(train_time, 1e-9), 1),
        "dev_curve": curve,
    }
    if not args.no_eval:
        tic = time.time()
        scores, predictions = evaluate_all(model, cfg, store)
        results.update(scores)
        results["eval_minutes"] = round((time.time() - tic) / 60, 2)
        json.dump(predictions, open(os.path.join(out, "predictions.json"), "w"))
        print(f"[{args.run}] test VQA {scores['vqa']}  pairs {scores['pairs']}  probe {scores['probe']}", flush=True)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
