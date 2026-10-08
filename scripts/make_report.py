"""Collect every finished run into tables and figures for the report.

  python scripts/make_report.py
Writes <runs>/report/: results.md, results.csv and the figures (PNG and PDF).
"""
import argparse
import csv
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vqa.bridges import build_bridge  # noqa: E402
from vqa.config import load_config  # noqa: E402
from vqa.data import FeatureStore, VQASplit  # noqa: E402
from vqa.vqa_eval import normalize  # noqa: E402

# Chart colours: categorical slots in fixed order, ink and chrome kept recessive.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TYPES = [("overall", "Overall"), ("yes/no", "Yes/No"), ("number", "Number"), ("other", "Other")]
LABELS = {"reference": "Query bridge, 32 tokens", "blind": "Blind twin", "lora": "With LoRA",
          "tokens16": "Query bridge, 16 tokens", "mlp16": "Pooled MLP, 16 tokens",
          "tokens64": "Query bridge, 64 tokens", "mlp64": "Pooled MLP, 64 tokens"}


def load_runs(runs_dir):
    runs = {}
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name, "results.json")
        if os.path.exists(path):
            r = json.load(open(path))
            if "vqa" in r:
                runs[name] = r
    return runs


def table(runs, out):
    cols = ["run", "bridge", "tokens", "lora", "trainable_M", "vqa", "yes/no", "number", "other", "swap_vqa",
            "answers_changed", "pair_both", "pair_same_answer", "probe_existence", "probe_counting",
            "probe_leftright", "ms_per_question", "train_min", "samples_per_sec"]
    rows = []
    for name, r in runs.items():
        c = r["config"]
        rows.append([
            name, c["bridge"]["type"] + (" (blind)" if c["bridge"]["blind"] else ""), c["bridge"]["tokens"],
            "yes" if c["llm"]["lora"]["enabled"] else "no", round(r["trainable_params"] / 1e6, 2),
            r["vqa"].get("overall"), r["vqa"].get("yes/no"), r["vqa"].get("number"), r["vqa"].get("other"),
            r["swap"]["vqa"].get("overall"), r["swap"]["answers_changed"].get("overall"),
            r["pairs"]["both_correct"], r["pairs"]["same_answer"], r["probe"].get("existence"),
            r["probe"].get("counting"), r["probe"].get("leftright"), r.get("ms_per_question"),
            r["train_minutes"], r["samples_per_sec"]])
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        csv.writer(f).writerows([cols] + rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    if "reference" in runs and "blind" in runs:
        gap = {k: round(runs["reference"]["vqa"][k] - runs["blind"]["vqa"][k], 2) for k, _ in TYPES
               if k in runs["reference"]["vqa"] and k in runs["blind"]["vqa"]}
        lines += ["", f"Blind gap G1 (reference minus blind twin, accuracy points): {gap}"]
    open(os.path.join(out, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def new_axes(n=1, width=5.2, height=3.4):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), squeeze=False)
    for ax in axes[0]:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
    return fig, axes[0]


def finish(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def fig_grounding(runs, out):
    """Accuracy per answer type: full model, the same model with swapped images, and the blind twin."""
    if "reference" not in runs:
        return
    ref = runs["reference"]
    series = [("Full model", ref["vqa"]), ("Images swapped", ref["swap"]["vqa"])]
    if "blind" in runs:
        series.append(("Blind twin", runs["blind"]["vqa"]))
    types = [(k, label) for k, label in TYPES if k in ref["vqa"]]
    fig, (ax,) = new_axes()
    width = 0.8 / len(series)
    for s, (label, scores) in enumerate(series):
        xs = np.arange(len(types)) + (s - (len(series) - 1) / 2) * width
        values = [scores.get(k) or 0 for k, _ in types]
        ax.bar(xs, values, width * 0.9, color=SERIES[s], label=label)
        ax.text(xs[0], values[0] + 1, f"{values[0]:.1f}", ha="center", va="bottom", fontsize=8, color=SECONDARY)
    ax.set_xticks(np.arange(len(types)), [label for _, label in types])
    ax.set_ylabel("VQA accuracy (%)", color=SECONDARY, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("The image carries the answer: accuracy by answer type", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY)
    finish(fig, out, "fig_grounding")


def fig_convergence(runs, out):
    """Dev accuracy against training steps."""
    shown = [n for n in ("reference", "tokens16", "mlp16", "lora", "blind") if n in runs and runs[n]["dev_curve"]]
    if not shown:
        return
    fig, (ax,) = new_axes()
    for s, name in enumerate(shown):
        curve = runs[name]["dev_curve"]
        xs, ys = [p["step"] for p in curve], [p["dev_acc"] for p in curve]
        ax.plot(xs, ys, color=SERIES[s], linewidth=2, marker="o", markersize=5, label=LABELS.get(name, name))
    ax.set_xlabel("Training step", color=SECONDARY, fontsize=9)
    ax.set_ylabel("Dev VQA accuracy (%)", color=SECONDARY, fontsize=9)
    ax.set_title("Convergence on the dev split", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY)
    finish(fig, out, "fig_convergence")


def fig_tradeoff(runs, out):
    """Accuracy against inference latency as the number of visual tokens changes. One panel per measure."""
    shown = [n for n in ("tokens16", "reference", "tokens64", "mlp16", "mlp64") if n in runs and runs[n].get("ms_per_question")]
    if len(shown) < 2:
        return
    probe = lambda r: float(np.mean([r["probe"][k] for k in ("existence", "counting", "leftright") if r["probe"].get(k) is not None] or [0]))
    panels = [("VQA accuracy (%)", lambda r: r["vqa"]["overall"]), ("GroundProbe accuracy (%)", probe)]
    fig, axes = new_axes(2, width=4.2)
    for ax, (label, value) in zip(axes, panels):
        for kind, color, marker in (("query", SERIES[0], "o"), ("mlp", SERIES[1], "s")):
            names = [n for n in shown if runs[n]["config"]["bridge"]["type"] == kind]
            if not names:
                continue
            names.sort(key=lambda n: runs[n]["ms_per_question"])
            xs, ys = [runs[n]["ms_per_question"] for n in names], [value(runs[n]) for n in names]
            ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                    label="Query bridge" if kind == "query" else "Pooled MLP")
            for n, x, y in zip(names, xs, ys):
                ax.annotate(f"{runs[n]['config']['bridge']['tokens']} tokens", (x, y), textcoords="offset points",
                            xytext=(6, 6), fontsize=8, color=SECONDARY)
        ax.set_xlabel("Inference time per question (ms)", color=SECONDARY, fontsize=9)
        ax.set_ylabel(label, color=SECONDARY, fontsize=9)
    axes[0].set_title("Accuracy against inference cost", loc="left", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY)
    finish(fig, out, "fig_tradeoff")


def fig_qualitative(runs, cfg, runs_dir, out, n=8, seed=0):
    """Sample predictions with the bridge's attention over the image; half correct, half wrong."""
    path = os.path.join(runs_dir, "reference", "bridge.pt")
    if "reference" not in runs or not os.path.exists(path):
        return
    ref_cfg = runs["reference"]["config"]
    state = torch.load(path, map_location="cpu")["bridge"]
    bridge = build_bridge(ref_cfg, state["last.weight"].shape[0])
    bridge.load_state_dict(state)
    bridge.eval()
    store = FeatureStore(ref_cfg["paths"]["cache"], preload=False)
    test = VQASplit(ref_cfg["paths"]["splits"], "test", store, ref_cfg["data"]["test_images"])
    pred = {p["qid"]: p["pred"] for p in json.load(open(os.path.join(runs_dir, "reference", "predictions.json")))["test"]}
    blind_path = os.path.join(runs_dir, "blind", "predictions.json")
    blind = {p["qid"]: p["pred"] for p in json.load(open(blind_path))["test"]} if os.path.exists(blind_path) else {}
    order = list(range(len(test)))
    random.Random(seed).shuffle(order)
    is_right = lambda i: normalize(pred[test.records[i]["qid"]]) == normalize(test.records[i]["mc"])
    picks = [i for i in order if is_right(i)][:n // 2] + [i for i in order if not is_right(i)][:n - n // 2]
    if not picks:
        return
    heat = LinearSegmentedColormap.from_list("attention", [(0.92, 0.41, 0.2, 0.0), (0.92, 0.41, 0.2, 0.8)])
    cols = (len(picks) + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(3.1 * cols, 7.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, picks):
        r = test.records[i]
        image = Image.open(os.path.join(cfg["paths"]["data"], "coco", "images", r["split"],
                                        f"COCO_{r['split']}_{r['img']:012d}.jpg")).convert("RGB")
        ax.imshow(image)
        with torch.no_grad():
            attns = bridge(test.features([i]).float(), need_attn=True)[1]
        if attns:
            grid = attns[-1][0].mean(0)
            side = int(grid.numel() ** 0.5)
            grid = (grid / grid.max()).reshape(side, side).numpy()
            ax.imshow(grid, cmap=heat, extent=(0, image.width, image.height, 0), interpolation="bilinear", vmin=0, vmax=1)
        lines = [r["q"], f"human: {r['mc']}   model: {pred[r['qid']]}"]
        if blind:
            lines.append(f"blind twin: {blind.get(r['qid'], '')}")
        ax.set_title("\n".join(lines), fontsize=8, color=INK, loc="left")
    fig.suptitle("Sample predictions with bridge attention (top row correct, bottom row wrong)", fontsize=10, color=INK, x=0.01, ha="left")
    finish(fig, out, "fig_qualitative")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    runs_dir = cfg["paths"]["runs"]
    out = os.path.join(runs_dir, "report")
    os.makedirs(out, exist_ok=True)
    runs = load_runs(runs_dir)
    if not runs:
        print("no finished runs found in", runs_dir)
        return
    table(runs, out)
    fig_grounding(runs, out)
    fig_convergence(runs, out)
    fig_tradeoff(runs, out)
    fig_qualitative(runs, cfg, runs_dir, out)
    print("report written to", out)


if __name__ == "__main__":
    main()
