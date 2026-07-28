"""
Alpha-only dev sweep, fixing tau/topk at the values Phi-3's fuller
coordinate-ascent search found barely mattered (tau=0.0, topk=32) --
so a second model/SAE doesn't need to re-run the full tau/topk axes to
find the same near-flat result. Only alpha, which is what actually
trades off CrowS/StereoSet accuracy against perplexity, needs a fresh
sweep per model.

Reuses the exact same eval functions as eval/dev_grid_search.py so
results are directly comparable/combinable with any points already
computed there (e.g. from a truncated run).
"""
import argparse

import pandas as pd
import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.bias_eval import run_crows, run_stereoset
from eval.split_utils import load_splits
from eval.ablation_test import load_sae_data, select_features_by_criterion, apply_sae_encoder_steering, register_steering_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--scores", default="results/gemma_sae_feature_scores_dev.npy")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0, 2.5, 3.0])
    ap.add_argument("--include_baseline", action="store_true")
    ap.add_argument("--out", default="results/gemma_alpha_only_sweep.csv")
    args = ap.parse_args()

    device = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if device in ["cuda", "mps"] else T.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    splits = load_splits(args.splits)
    crows_idx, stereo_idx = splits["crows"]["dev"], splits["stereoset"]["dev"]

    E, D, d, delta = load_sae_data(args.scores)
    feat_ids = select_features_by_criterion(d, delta, topk=args.topk, criterion="cohens_d")
    E_t = T.tensor(E, dtype=T.float32, device=device)

    rows = []

    if args.include_baseline:
        c_acc, _, _ = run_crows(model, tok, device, indices=crows_idx)
        s_acc, _, _ = run_stereoset(model, tok, device, indices=stereo_idx)
        row = {"method": "baseline", "alpha": 0.0, "tau": None, "topk": None,
               "crows_dev_acc": round(c_acc, 4), "stereo_dev_acc": round(s_acc, 4)}
        rows.append(row)
        print(row)

    for alpha in args.alphas:
        steering_fn = lambda h: apply_sae_encoder_steering(h, E_t, feat_ids, alpha, args.tau)
        handle = register_steering_hook(model, steering_fn, [], args.layer_index)
        try:
            c_acc, _, _ = run_crows(model, tok, device, indices=crows_idx)
            s_acc, _, _ = run_stereoset(model, tok, device, indices=stereo_idx)
        finally:
            handle.remove()
        row = {"method": "sae_encoder", "alpha": alpha, "tau": args.tau, "topk": args.topk,
               "crows_dev_acc": round(c_acc, 4), "stereo_dev_acc": round(s_acc, 4)}
        rows.append(row)
        print(row)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
