"""Split files, the cached-feature store and batching."""
import json
import os
import random

import numpy as np
import torch


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_images(splits_dir, train_images, dev_images, test_images):
    """The images a configuration needs: the first N of each split, in cache order."""
    index = load_json(os.path.join(splits_dir, "images.json"))
    return index["test"][:test_images] + index["dev"][:dev_images] + index["train"][:train_images]


def image_key(split, image_id):
    return f"{split}/{image_id}"


class FeatureStore:
    """Vision features on disk: feats.npy of shape (images, tokens, dim) plus index.json."""

    def __init__(self, cache_dir, preload=False):
        keys = load_json(os.path.join(cache_dir, "index.json"))
        self.row = {k: i for i, k in enumerate(keys)}
        self.feats = np.load(os.path.join(cache_dir, "feats.npy"), mmap_mode=None if preload else "r")

    def has(self, split, image_id):
        return image_key(split, image_id) in self.row

    def get(self, rows):
        return torch.from_numpy(np.ascontiguousarray(self.feats[np.asarray(rows)]))


class VQASplit:
    """Questions of one split, limited to the first `images` images and to cached features."""

    def __init__(self, splits_dir, name, store, images=None):
        records = load_json(os.path.join(splits_dir, f"{name}.json"))
        if images is not None:
            records = [r for r in records if r["rank"] < images]
        self.records = [r for r in records if store.has(r["split"], r["img"])]
        if len(self.records) < len(records):
            print(f"[data] {name}: {len(records) - len(self.records)} questions dropped (image not cached)")
        self.rows = np.array([store.row[image_key(r["split"], r["img"])] for r in self.records], dtype=np.int64)
        self.store = store

    def __len__(self):
        return len(self.records)

    def batches(self, batch_size, shuffle=False, seed=0, epoch=0):
        order = list(range(len(self)))
        if shuffle:
            random.Random(seed * 1000 + epoch).shuffle(order)
        for i in range(0, len(order), batch_size):
            yield order[i:i + batch_size]

    def features(self, idx, rows=None):
        return self.store.get(self.rows[idx] if rows is None else rows)


class ProbeSplit(VQASplit):
    """GroundProbe questions in the same shape as a VQA split."""

    def __init__(self, splits_dir, store):
        probe = load_json(os.path.join(splits_dir, "groundprobe.json"))
        self.records = [{"qid": p["pid"], "img": p["img"], "split": "val2014", "q": p["q"], "mc": p["a"],
                         "cat": p["cat"]} for p in probe if store.has("val2014", p["img"])]
        self.rows = np.array([store.row[image_key("val2014", r["img"])] for r in self.records], dtype=np.int64)
        self.store = store
