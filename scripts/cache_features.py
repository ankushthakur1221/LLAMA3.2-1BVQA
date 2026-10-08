"""Run the frozen vision encoder once over the needed images and store its patch features.

  python scripts/cache_features.py                              # images of configs/default.yaml
  python scripts/cache_features.py --config configs/smoke.yaml
Resumes if interrupted. Output: <cache>/feats.npy (float16) and <cache>/index.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vqa.config import load_config  # noqa: E402
from vqa.data import image_key, select_images  # noqa: E402
from vqa.model import resolve_device  # noqa: E402


class Images(torch.utils.data.Dataset):
    def __init__(self, entries, todo, image_root, processor):
        self.entries, self.todo, self.root, self.processor = entries, todo, image_root, processor

    def __len__(self):
        return len(self.todo)

    def __getitem__(self, i):
        row = self.todo[i]
        e = self.entries[row]
        image = Image.open(os.path.join(self.root, e["split"], e["file"])).convert("RGB")
        return row, self.processor(images=image, return_tensors="pt")["pixel_values"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    device = resolve_device(cfg["device"])
    d, v, out = cfg["data"], cfg["vision"], cfg["paths"]["cache"]
    entries = select_images(cfg["paths"]["splits"], d["train_images"], d["dev_images"], d["test_images"])
    keys = [image_key(e["split"], e["id"]) for e in entries]
    os.makedirs(out, exist_ok=True)
    index_path, feats_path, done_path = (os.path.join(out, n) for n in ("index.json", "feats.npy", "done.npy"))
    shape = (len(keys), v["tokens"], v["dim"])

    resume = os.path.exists(done_path) and os.path.exists(index_path) and json.load(open(index_path)) == keys
    if resume:
        feats, done = np.load(feats_path, mmap_mode="r+"), np.load(done_path)
    else:
        feats = np.lib.format.open_memmap(feats_path, mode="w+", dtype=np.float16, shape=shape)
        done = np.zeros(len(keys), dtype=bool)
        json.dump(keys, open(index_path, "w"))
    todo = np.flatnonzero(~done).tolist()
    print(f"{len(keys)} images, {len(todo)} to encode on {device}; cache {shape} = {np.prod(shape) * 2 / 1e9:.1f} GB", flush=True)
    if not todo:
        return

    cuda = device.type == "cuda"
    dtype = torch.float16 if cuda else torch.float32
    model = SiglipVisionModel.from_pretrained(v["name"], torch_dtype=dtype).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(v["name"], use_fast=False)
    if not cuda:
        torch.set_num_threads(min(8, os.cpu_count()))
    loader = torch.utils.data.DataLoader(
        Images(entries, todo, os.path.join(cfg["paths"]["data"], "coco", "images"), processor),
        batch_size=args.batch_size or (256 if cuda else 8),
        num_workers=args.workers if args.workers is not None else (12 if cuda else 0))
    t0, n = time.time(), 0
    with torch.no_grad():
        for rows, pixels in loader:
            hidden = model(pixel_values=pixels.to(device, dtype), output_hidden_states=True).hidden_states[v["layer"]]
            assert tuple(hidden.shape[1:]) == shape[1:], f"unexpected feature shape {tuple(hidden.shape)}"
            assert torch.isfinite(hidden).all(), "non-finite features"
            feats[rows.numpy()] = hidden.float().cpu().numpy().astype(np.float16)
            done[rows.numpy()] = True
            n += len(rows)
            if n % (len(rows) * 20) == 0 or n == len(todo):
                feats.flush()
                np.save(done_path, done)
                print(f"  {n}/{len(todo)}  {n / (time.time() - t0):.0f} img/s", flush=True)
    feats.flush()
    np.save(done_path, done)
    print(f"cached in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
