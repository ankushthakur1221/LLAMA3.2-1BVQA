"""The whole GPU session in one command, inside a time budget.

  python scripts/run_session.py                       # 195-minute budget
  python scripts/run_session.py --budget-minutes 180 --extras

Steps: speed test -> pick the training-set size that fits -> cache features -> learning-rate check
-> the planned runs -> extra runs if time is left -> tables and figures.
Finished steps are skipped on a rerun, so a crash costs at most one run.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vqa.config import load_config  # noqa: E402

PLAN = ["reference", "blind", "lora", "tokens16", "mlp16"]
EXTRA = ["seed1", "seed2", "tokens64", "mlp64"]
COST = {"lora": 1.3, "tokens16": 0.8, "mlp16": 0.8, "tokens64": 1.5, "mlp64": 1.5}  # relative to reference, estimates
SIZES = [40000, 30000, 20000, 15000, 10000, 5000]
QUESTIONS_PER_IMAGE = 5.36
EVAL_MINUTES, CACHE_IMAGES_PER_SEC, LR_CHECK_MINUTES = 3.0, 300.0, 6.0  # planning estimates


def script(name, *args):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", name), *map(str, args)]
    print("\n>>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def read(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--budget-minutes", type=float, default=195)
    ap.add_argument("--extras", action="store_true", help="add the extra runs when time is left")
    ap.add_argument("--skip-lr-check", action="store_true")
    ap.add_argument("--train-images", type=int, default=None, help="fix the training-set size instead of choosing it")
    args = ap.parse_args()
    cfg = load_config(args.config)
    runs_dir, epochs = cfg["paths"]["runs"], cfg["train"]["epochs"]
    t0 = time.time()
    state_path = os.path.join(runs_dir, "session.json")
    state = read(state_path) or {}

    def left():
        return args.budget_minutes - (time.time() - t0) / 60

    def save():
        os.makedirs(runs_dir, exist_ok=True)
        json.dump(state, open(state_path, "w"), indent=1)

    def run_minutes(run, images):
        return COST.get(run, 1.0) * images * QUESTIONS_PER_IMAGE * epochs / state["samples_per_sec"] / 60 + EVAL_MINUTES

    # 1. Speed test.
    bench_path = os.path.join(runs_dir, "benchmark_reference.json")
    if not read(bench_path):
        script("benchmark.py", "--config", args.config, "--run", "reference")
    state["samples_per_sec"] = read(bench_path)["train_samples_per_sec"]

    # 2. Largest training set whose planned runs fit the budget, keeping 10% in reserve.
    if "train_images" not in state:
        fixed = LR_CHECK_MINUTES * (not args.skip_lr_check) + 0.1 * args.budget_minutes
        state["train_images"] = args.train_images or next(
            (n for n in SIZES if sum(run_minutes(r, n) for r in PLAN) + (n + 7000) / CACHE_IMAGES_PER_SEC / 60 + fixed <= left()),
            SIZES[-1])
    images = state["train_images"]
    data = f"data.train_images={images}"
    print(f"\n{state['samples_per_sec']} samples/s -> {images} training images; "
          f"planned runs need about {sum(run_minutes(r, images) for r in PLAN):.0f} min", flush=True)
    save()

    # 3. Feature cache.
    script("cache_features.py", "--config", args.config, "--set", data)

    # 4. Learning-rate check: three short runs scored on the dev split.
    if "lr" not in state and not args.skip_lr_check:
        scores = {}
        for lr in (1e-4, 2e-4, 5e-4):
            name = f"lrcheck_{lr:g}"
            script("train.py", "--config", args.config, "--run", "reference", "--name", name, "--no-eval", "--set",
                   f"train.lr={lr}", f"data.train_images={min(images, 3000)}", "train.epochs=1", "train.dev_evals=1")
            scores[lr] = read(os.path.join(runs_dir, name, "results.json"))["dev_curve"][-1]["dev_acc"]
        state["lr_check"], state["lr"] = {f"{k:g}": v for k, v in scores.items()}, max(scores, key=scores.get)
        save()
    lr = state.get("lr", cfg["train"]["lr"])

    # 5. Planned runs, then extras while time allows. A run is capped so the remaining ones still fit.
    todo = [r for r in PLAN if not read(os.path.join(runs_dir, r, "results.json"))]
    for i, run in enumerate(todo):
        cap = max(2.0, (left() - EVAL_MINUTES * len(todo[i:])) / len(todo[i:]))
        script("train.py", "--config", args.config, "--run", run, "--set", data, f"train.lr={lr}", f"train.max_minutes={cap:.1f}")
    if args.extras:
        for run in EXTRA:
            if not read(os.path.join(runs_dir, run, "results.json")) and run_minutes(run, images) * 1.2 < left():
                script("train.py", "--config", args.config, "--run", run, "--set", data, f"train.lr={lr}")

    # 6. Tables and figures.
    script("make_report.py", "--config", args.config)
    state["session_minutes"] = round((time.time() - t0) / 60, 1)
    save()
    print(f"\nsession finished in {state['session_minutes']} min")


if __name__ == "__main__":
    main()
