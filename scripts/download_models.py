"""Fetch the pretrained weights into the Hugging Face cache. Network only, no GPU.

  python scripts/download_models.py                  # vision encoder + language model from the config
  python scripts/download_models.py --config configs/smoke.yaml
"""
import argparse
import os
import sys

from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vqa.config import load_config  # noqa: E402

PATTERNS = ["*.json", "*.safetensors", "tokenizer.model", "*.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for name in (cfg["vision"]["name"], cfg["llm"]["name"]):
        try:
            path = snapshot_download(name, allow_patterns=PATTERNS)
            print(f"ok   {name} -> {path}")
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0]}")


if __name__ == "__main__":
    main()
