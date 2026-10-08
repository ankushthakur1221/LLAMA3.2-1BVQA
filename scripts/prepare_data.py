"""Download VQA v2 / COCO files, build the splits and the GroundProbe set. CPU and network only.

  python scripts/prepare_data.py annotations   # question / answer / pair / instance files
  python scripts/prepare_data.py splits        # train / dev / test split files
  python scripts/prepare_data.py probe         # GroundProbe questions
  python scripts/prepare_data.py images        # only the images the splits need
  python scripts/prepare_data.py all
"""
import argparse
import csv
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vqa.data import select_images  # noqa: E402

VQA_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
VQA_ZIPS = [
    "v2_Questions_Train_mscoco.zip",
    "v2_Annotations_Train_mscoco.zip",
    "v2_Questions_Val_mscoco.zip",
    "v2_Annotations_Val_mscoco.zip",
    "v2_Complementary_Pairs_Val_mscoco.zip",
]
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_IMG_URL = "http://images.cocodataset.org/{split}/{file}"

TRAIN_POOL, DEV_IMAGES, TEST_IMAGES = 40000, 2000, 5000
MIN_AREA = 0.01  # an object must cover at least 1% of the image to be asked about
PLURAL = {
    "person": "people", "bus": "buses", "bench": "benches", "sheep": "sheep", "skis": "skis",
    "wine glass": "wine glasses", "knife": "knives", "sandwich": "sandwiches", "couch": "couches",
    "mouse": "mice", "scissors": "scissors", "toothbrush": "toothbrushes",
}
NO_ARTICLE = {"skis", "scissors"}  # already plural, skipped in yes/no templates


def download(url, dest, connections=16):
    """Download to dest; files over 50 MB are fetched as byte ranges over several connections."""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url}", flush=True)
    part = dest + ".part"
    size = int(requests.head(url, allow_redirects=True, timeout=60).headers.get("content-length", 0))
    if size < 50 << 20:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    else:
        with open(part, "wb") as f:
            f.truncate(size)
        step = -(-size // connections)

        def fetch(start):
            end = min(start + step, size) - 1
            for _ in range(5):
                try:
                    with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        with open(part, "r+b") as f:
                            f.seek(start)
                            for chunk in r.iter_content(1 << 20):
                                f.write(chunk)
                    return
                except Exception:
                    continue
            raise RuntimeError(f"range {start}-{end} of {url} failed")

        with ThreadPoolExecutor(connections) as pool:
            list(pool.map(fetch, range(0, size, step)))
    os.replace(part, dest)


def extract_json(zip_path, out_dir, only=None):
    """Extract .json members flat into out_dir (names are reduced to their basename)."""
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            name = os.path.basename(m.filename)
            if not name.endswith(".json") or (only and name not in only):
                continue
            target = os.path.join(out_dir, name)
            if not os.path.exists(target):
                with z.open(m) as src, open(target, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)


def step_annotations(root):
    vqa_dir, coco_dir = os.path.join(root, "vqa"), os.path.join(root, "coco")
    for name in VQA_ZIPS:
        download(VQA_URL + name, os.path.join(vqa_dir, "zips", name))
        extract_json(os.path.join(vqa_dir, "zips", name), vqa_dir)
    zip_path = os.path.join(coco_dir, "zips", "annotations_trainval2014.zip")
    download(COCO_ANN_URL, zip_path)
    extract_json(zip_path, coco_dir, only={"instances_val2014.json"})
    print("annotations ready")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def vqa_records(vqa_dir, split):
    questions = load_json(os.path.join(vqa_dir, f"v2_OpenEnded_mscoco_{split}_questions.json"))["questions"]
    text = {q["question_id"]: q["question"] for q in questions}
    out = []
    for a in load_json(os.path.join(vqa_dir, f"v2_mscoco_{split}_annotations.json"))["annotations"]:
        out.append({
            "qid": a["question_id"], "img": a["image_id"], "split": split, "q": text[a["question_id"]],
            "answers": [x["answer"] for x in a["answers"]], "mc": a["multiple_choice_answer"],
            "atype": a["answer_type"],
        })
    return out


def image_entry(split, image_id):
    return {"id": image_id, "split": split, "file": f"COCO_{split}_{image_id:012d}.jpg"}


def step_splits(root, seed):
    vqa_dir, out_dir = os.path.join(root, "vqa"), os.path.join(root, "splits")
    rng = random.Random(seed)

    # Test: draw complementary pairs and add both images until TEST_IMAGES is reached.
    val = vqa_records(vqa_dir, "val2014")
    img_of = {r["qid"]: r["img"] for r in val}
    pairs = [p for p in load_json(os.path.join(vqa_dir, "v2_mscoco_val2014_complementary_pairs.json"))
             if p[0] in img_of and p[1] in img_of]
    order = pairs[:]
    rng.shuffle(order)
    test_imgs = set()
    for a, b in order:
        new = {img_of[a], img_of[b]} - test_imgs
        if len(test_imgs) + len(new) <= TEST_IMAGES:
            test_imgs |= new
        if len(test_imgs) == TEST_IMAGES:
            break
    test = [r for r in val if r["img"] in test_imgs]
    test_pairs = [p for p in pairs if img_of[p[0]] in test_imgs and img_of[p[1]] in test_imgs]

    # Train pool and dev: one shuffled list of train2014 images; a run uses the first N of the pool.
    train_all = vqa_records(vqa_dir, "train2014")
    ids = sorted({r["img"] for r in train_all})
    rng.shuffle(ids)
    rank = {img: i for i, img in enumerate(ids[:TRAIN_POOL])}
    dev_ids = ids[TRAIN_POOL:TRAIN_POOL + DEV_IMAGES]
    dev_rank = {img: i for i, img in enumerate(dev_ids)}
    train = sorted((dict(r, rank=rank[r["img"]]) for r in train_all if r["img"] in rank), key=lambda r: r["rank"])
    dev = sorted((dict(r, rank=dev_rank[r["img"]]) for r in train_all if r["img"] in dev_rank), key=lambda r: r["rank"])
    test_ids = sorted(test_imgs)
    test_rank = {img: i for i, img in enumerate(test_ids)}
    test = [dict(r, rank=test_rank[r["img"]]) for r in test]

    dump_json(train, os.path.join(out_dir, "train.json"))
    dump_json(dev, os.path.join(out_dir, "dev.json"))
    dump_json(test, os.path.join(out_dir, "test.json"))
    dump_json(test_pairs, os.path.join(out_dir, "test_pairs.json"))
    dump_json({
        "test": [image_entry("val2014", i) for i in test_ids],
        "dev": [image_entry("train2014", i) for i in dev_ids],
        "train": [image_entry("train2014", i) for i in ids[:TRAIN_POOL]],
    }, os.path.join(out_dir, "images.json"))
    print(f"splits: train {len(train)} q / {TRAIN_POOL} img, dev {len(dev)} q / {len(dev_ids)} img, "
          f"test {len(test)} q / {len(test_ids)} img, intact test pairs {len(test_pairs)}")


def article(name):
    return ("an " if name[0] in "aeiou" else "a ") + name


def step_probe(root, seed, per_category=1000):
    """GroundProbe: balanced existence / counting / left-right questions built from COCO boxes."""
    out_dir = os.path.join(root, "splits")
    rng = random.Random(seed + 1)
    test_ids = [e["id"] for e in load_json(os.path.join(out_dir, "images.json"))["test"]]
    coco = load_json(os.path.join(root, "coco", "instances_val2014.json"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    size = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    anns = defaultdict(lambda: defaultdict(list))  # image -> category -> annotations
    keep = set(test_ids)
    for a in coco["annotations"]:
        if a["image_id"] in keep:
            anns[a["image_id"]][a["category_id"]].append(a)

    def big(a, img):
        w, h = size[img]
        return not a["iscrowd"] and a["area"] >= MIN_AREA * w * h

    present = {img: set(anns[img]) for img in test_ids}
    freq = Counter(c for img in test_ids for c in present[img])
    popular = [c for c, _ in freq.most_common(10)]
    cooc = defaultdict(Counter)
    for img in test_ids:
        for a in present[img]:
            for b in present[img]:
                if a != b:
                    cooc[a][b] += 1
    askable = [c for c in names if names[c] not in NO_ARTICLE]
    probe = []

    # Existence, following POPE: absent objects are sampled as random, popular and adversarial.
    cands = [img for img in test_ids if any(c in askable and any(big(a, img) for a in anns[img][c]) for c in present[img])]
    rng.shuffle(cands)
    for k, img in enumerate(cands[:per_category // 2]):
        pos = rng.choice([c for c in present[img] if c in askable and any(big(a, img) for a in anns[img][c])])
        absent = [c for c in askable if c not in present[img]]
        mode = ("random", "popular", "adversarial")[k % 3]
        if mode == "popular" and any(c in absent for c in popular):
            neg = rng.choice([c for c in popular if c in absent])
        elif mode == "adversarial":
            score = Counter()
            for c in present[img]:
                score.update(cooc[c])
            neg = max(absent, key=lambda c: (score[c], rng.random()))
        else:
            neg = rng.choice(absent)
        for cat, ans, how in ((pos, "yes", "present"), (neg, "no", mode)):
            probe.append({"img": img, "cat": "existence", "q": f"Is there {article(names[cat])} in the image?",
                          "a": ans, "meta": {"object": names[cat], "sampling": how}})

    # Counting: 1 to 4 instances, none of them crowd or tiny, equal share of each count.
    by_count = defaultdict(list)
    for img in test_ids:
        for c, lst in anns[img].items():
            if 1 <= len(lst) <= 4 and all(big(a, img) for a in lst):
                by_count[len(lst)].append((img, c))
    n = min([per_category // 4] + [len(by_count[k]) for k in (1, 2, 3, 4)])
    for k in (1, 2, 3, 4):
        for img, c in rng.sample(by_count[k], n):
            plural = PLURAL.get(names[c], names[c] + "s")
            probe.append({"img": img, "cat": "counting", "q": f"How many {plural} are in the image?",
                          "a": str(k), "meta": {"object": names[c]}})

    # Left / right: one instance of each object, boxes separated horizontally by a clear margin.
    lr = []
    for img in test_ids:
        w, _ = size[img]
        single = [(c, lst[0]) for c, lst in anns[img].items() if len(lst) == 1 and big(lst[0], img) and c in askable]
        ok = [(a, b) for a, x in single for b, y in single
              if x["bbox"][0] + x["bbox"][2] + 0.05 * w <= y["bbox"][0]]  # a is clearly left of b
        if ok:
            lr.append((img, rng.choice(ok)))
    rng.shuffle(lr)
    for k, (img, (left, right)) in enumerate(lr[:per_category]):
        a, b, ans = (left, right, "yes") if k % 2 == 0 else (right, left, "no")
        probe.append({"img": img, "cat": "leftright", "q": f"Is the {names[a]} to the left of the {names[b]}?",
                      "a": ans, "meta": {"left": names[left], "right": names[right]}})

    for i, p in enumerate(probe):
        p["pid"] = i
    dump_json(probe, os.path.join(out_dir, "groundprobe.json"))
    sheet = rng.sample(probe, min(200, len(probe)))
    with open(os.path.join(out_dir, "groundprobe_handcheck.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pid", "image_file", "category", "question", "answer", "label_correct (y/n)", "note"])
        for p in sheet:
            w.writerow([p["pid"], f"COCO_val2014_{p['img']:012d}.jpg", p["cat"], p["q"], p["a"], "", ""])
    print("GroundProbe:", dict(Counter(p["cat"] for p in probe)), "answers:", dict(Counter((p["cat"], p["a"]) for p in probe)))


def step_images(root, train_images, dev_images, test_images, workers=96):
    entries = select_images(os.path.join(root, "splits"), train_images, dev_images, test_images)
    todo = []
    for e in entries:
        dest = os.path.join(root, "coco", "images", e["split"], e["file"])
        if not os.path.exists(dest):
            todo.append((COCO_IMG_URL.format(split=e["split"], file=e["file"]), dest))
    print(f"images: {len(entries)} needed, {len(todo)} to download", flush=True)
    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers))

    def fetch(job):
        url, dest = job
        for _ in range(4):
            try:
                r = session.get(url, timeout=60)
                r.raise_for_status()
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest + ".part", "wb") as f:
                    f.write(r.content)
                os.replace(dest + ".part", dest)
                return True
            except Exception:
                continue
        return False

    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        for i, ok in enumerate(pool.map(fetch, todo), 1):
            failed += not ok
            if i % 2000 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {failed} failed", flush=True)
    print("images ready" if not failed else f"{failed} images failed; rerun to retry")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["annotations", "splits", "probe", "images", "all"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-images", type=int, default=TRAIN_POOL)
    ap.add_argument("--dev-images", type=int, default=DEV_IMAGES)
    ap.add_argument("--test-images", type=int, default=TEST_IMAGES)
    ap.add_argument("--workers", type=int, default=96, help="parallel image downloads")
    args = ap.parse_args()
    if args.step in ("annotations", "all"):
        step_annotations(args.root)
    if args.step in ("splits", "all"):
        step_splits(args.root, args.seed)
    if args.step in ("probe", "all"):
        step_probe(args.root, args.seed)
    if args.step in ("images", "all"):
        step_images(args.root, args.train_images, args.dev_images, args.test_images, args.workers)


if __name__ == "__main__":
    main()
