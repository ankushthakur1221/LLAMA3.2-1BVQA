"""Answer accuracy and the groundedness measures (image swap, complementary pairs, GroundProbe)."""
import os
import time
from collections import defaultdict

import numpy as np
import torch

from .data import ProbeSplit, VQASplit, load_json
from .vqa_eval import normalize, vqa_accuracy


@torch.no_grad()
def predict(model, split, batch_size, limit=None, swap_rows=None):
    model.bridge.eval()
    preds = []
    for idx in split.batches(batch_size):
        if limit is not None and len(preds) >= limit:
            break
        rows = None if swap_rows is None else swap_rows[idx]
        with model.autocast():
            preds += model.answer(split.features(idx, rows), [split.records[i]["q"] for i in idx])
    return preds


def mean_pct(values):
    return round(100.0 * float(np.mean(values)), 2) if len(values) else None


def vqa_scores(records, preds):
    by_type = defaultdict(list)
    for r, p in zip(records, preds):
        acc = vqa_accuracy(p, r["answers"])
        by_type["overall"].append(acc)
        by_type[r["atype"]].append(acc)
    return {k: mean_pct(v) for k, v in by_type.items()}


def other_image_rows(rows, seed=0):
    """For every question, the feature row of a different test image."""
    unique = np.unique(rows)
    shuffled = np.random.RandomState(seed).permutation(unique)
    other = dict(zip(shuffled, np.roll(shuffled, 1)))
    return np.array([other[r] for r in rows], dtype=np.int64)


def evaluate_all(model, cfg, store):
    """Test accuracy plus G2 (image swap), G3 (complementary pairs) and G4 (GroundProbe)."""
    splits, bs = cfg["paths"]["splits"], cfg["eval"]["batch_size"]
    test = VQASplit(splits, "test", store, cfg["data"]["test_images"])
    tic = time.time()
    preds = predict(model, test, bs)
    res = {"questions": len(test), "vqa": vqa_scores(test.records, preds),
           "ms_per_question": round(1000 * (time.time() - tic) / max(1, len(test)), 3)}

    swapped = predict(model, test, bs, swap_rows=other_image_rows(test.rows))
    changed = defaultdict(list)
    for r, p, s in zip(test.records, preds, swapped):
        changed["overall"].append(p != s)
        changed[r["atype"]].append(p != s)
    res["swap"] = {"vqa": vqa_scores(test.records, swapped), "answers_changed": {k: mean_pct(v) for k, v in changed.items()}}

    right = {r["qid"]: normalize(p) == normalize(r["mc"]) for r, p in zip(test.records, preds)}
    said = {r["qid"]: normalize(p) for r, p in zip(test.records, preds)}
    pairs = [p for p in load_json(os.path.join(splits, "test_pairs.json")) if p[0] in right and p[1] in right]
    res["pairs"] = {
        "pairs": len(pairs),
        "both_correct": mean_pct([right[a] and right[b] for a, b in pairs]),
        "single_correct": mean_pct([right[q] for p in pairs for q in p]),
        "same_answer": mean_pct([said[a] == said[b] for a, b in pairs]),
    }

    probe = ProbeSplit(splits, store)
    probe_preds = predict(model, probe, bs)
    by_cat = defaultdict(list)
    for r, p in zip(probe.records, probe_preds):
        by_cat[r["cat"]].append(normalize(p) == r["mc"])
    res["probe"] = {"questions": len(probe), **{k: mean_pct(v) for k, v in by_cat.items()}}

    predictions = {
        "test": [{"qid": r["qid"], "pred": p, "swap": s} for r, p, s in zip(test.records, preds, swapped)],
        "probe": [{"pid": r["qid"], "pred": p} for r, p in zip(probe.records, probe_preds)],
    }
    return res, predictions
