"""
Deterministic dev/test splits for CrowS-Pairs and StereoSet.

Fixes a data-leakage bug: bias-direction / SAE-feature-scoring data
(sae/extract_bias_data.py) used to be drawn from the *same* leading indices
of these datasets that eval/bias_eval.py later scored for the reported
CrowS-Pairs / StereoSet accuracy. That means the steering direction was
partly derived from the examples used to "prove" it worked.

This module builds one fixed, seeded split per dataset and persists it to
data/splits.json so every script (feature scoring, hyperparameter grid
search, final evaluation) references the exact same dev/test partition.

Usage:
    python -m eval.split_utils --out data/splits.json --dev_frac 0.4 --seed 1337
"""
import argparse
import json
import os

import numpy as np
from datasets import load_dataset

LABEL_MAP = {0: "stereotype", 1: "anti-stereotype", 2: "unrelated"}


def _stereoset_valid_indices(ds):
    """Indices of StereoSet examples that have all three labels present
    (mirrors the filtering already done inline in eval/bias_eval.py)."""
    valid = []
    for i, ex in enumerate(ds):
        S = ex.get("sentences")
        lab2sent = {}
        if isinstance(S, list):
            for d in S:
                if isinstance(d, dict):
                    lab = d.get("gold_label", d.get("label"))
                    if isinstance(lab, int):
                        lab = LABEL_MAP.get(lab)
                    if lab in ("stereotype", "anti-stereotype", "unrelated"):
                        lab2sent[lab] = d.get("sentence")
        elif isinstance(S, dict):
            for sent, lab in zip(S.get("sentence", []), S.get("gold_label", S.get("label", []))):
                lab = LABEL_MAP.get(lab, lab)
                if lab in ("stereotype", "anti-stereotype", "unrelated"):
                    lab2sent[lab] = sent
        if all(k in lab2sent for k in ("stereotype", "anti-stereotype", "unrelated")):
            valid.append(i)
    return valid


def build_splits(dev_frac=0.4, seed=1337):
    rng = np.random.RandomState(seed)

    crows_ds = load_dataset("crows_pairs", trust_remote_code=True)["test"]
    n_crows = len(crows_ds)
    crows_idx = np.arange(n_crows)
    rng.shuffle(crows_idx)
    n_dev = int(round(dev_frac * n_crows))
    crows_dev, crows_test = sorted(crows_idx[:n_dev].tolist()), sorted(crows_idx[n_dev:].tolist())

    stereo_ds = load_dataset("McGill-NLP/stereoset", "intrasentence", trust_remote_code=True)["validation"]
    stereo_valid = _stereoset_valid_indices(stereo_ds)
    stereo_valid = np.array(stereo_valid)
    rng.shuffle(stereo_valid)
    n_dev_s = int(round(dev_frac * len(stereo_valid)))
    stereo_dev = sorted(stereo_valid[:n_dev_s].tolist())
    stereo_test = sorted(stereo_valid[n_dev_s:].tolist())

    return {
        "seed": seed,
        "dev_frac": dev_frac,
        "crows": {"n_total": n_crows, "dev": crows_dev, "test": crows_test},
        "stereoset": {
            "n_total_raw": len(stereo_ds),
            "n_total_valid": len(stereo_valid),
            "dev": stereo_dev,
            "test": stereo_test,
        },
    }


def load_splits(path="data/splits.json"):
    with open(path) as f:
        return json.load(f)


def crows_subset(ds, splits, split):
    idx = splits["crows"][split]
    return ds.select(idx)


def stereoset_subset(ds, splits, split):
    idx = splits["stereoset"][split]
    return ds.select(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--dev_frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    splits = build_splits(dev_frac=args.dev_frac, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(splits, f)

    c = splits["crows"]
    s = splits["stereoset"]
    overlap_c = set(c["dev"]) & set(c["test"])
    overlap_s = set(s["dev"]) & set(s["test"])
    print(f"CrowS: {len(c['dev'])} dev / {len(c['test'])} test (total {c['n_total']}), overlap={len(overlap_c)}")
    print(f"StereoSet: {len(s['dev'])} dev / {len(s['test'])} test "
          f"(valid total {s['n_total_valid']} of raw {s['n_total_raw']}), overlap={len(overlap_s)}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
