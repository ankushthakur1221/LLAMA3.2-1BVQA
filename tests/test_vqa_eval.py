"""Known cases for the VQA accuracy and normalisation. Run: python tests/test_vqa_eval.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vqa.vqa_eval import normalize, vqa_accuracy  # noqa: E402

CASES = [
    (vqa_accuracy("yes", ["yes"] * 10), 1.0),
    (vqa_accuracy("no", ["yes"] * 10), 0.0),
    (vqa_accuracy("2", ["two"] * 5 + ["2"] * 5), 1.0),              # number words map to digits
    (round(vqa_accuracy("a dog", ["dog"] * 2 + ["cat"] * 8), 4), 0.6),  # 2 of 10 annotators agree
    (round(vqa_accuracy("red", ["red"] * 3 + ["blue"] * 7), 4), 0.9),   # 3 of 10 annotators agree
    (normalize("The  Two dogs."), "2 dogs"),
    (normalize("dont know"), "don't know"),
]

if __name__ == "__main__":
    bad = [(got, want) for got, want in CASES if got != want]
    for got, want in bad:
        print("FAIL: got", repr(got), "expected", repr(want))
    print(f"{len(CASES) - len(bad)}/{len(CASES)} cases pass")
    sys.exit(1 if bad else 0)
