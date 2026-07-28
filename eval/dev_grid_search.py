"""
Hyperparameter search for encoder-SAE steering and direct-activation steering,
restricted to the DEV split only (see eval/split_utils.py). The held-out TEST
split is never touched here -- it is reserved for the single final run in
eval/final_test_eval.py, which is what the paper's significance test reports.

Coordinate-ascent style sweep (vary one axis at a time around a center point),
matching the original 27-run grid's design but now leakage-free.
"""
import argparse
import json
import os

import numpy as np
import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.bias_eval import run_crows, run_stereoset
from eval.split_utils import load_splits
from eval.ablation_test import (
    load_sae_data,
    select_features_by_criterion,
    apply_sae_encoder_steering,
    register_steering_hook,
    build_direct_steering_vector,
    apply_direct_activation_steering,
)


def eval_dev(model, tok, device, splits, samples_cap=None):
    crows_idx = splits["crows"]["dev"]
    stereo_idx = splits["stereoset"]["dev"]
    if samples_cap:
        crows_idx = crows_idx[:samples_cap]
        stereo_idx = stereo_idx[:samples_cap]
    c_acc, _, _ = run_crows(model, tok, device, indices=crows_idx)
    s_acc, _, _ = run_stereoset(model, tok, device, indices=stereo_idx)
    return c_acc, s_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--scores", default="results/sae_feature_scores_dev.npy")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--biased_txt", default="data/biased_x5.txt")
    ap.add_argument("--neutral_txt", default="data/neutral_x5.txt")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--out", default="results/dev_grid_search.csv")
    ap.add_argument("--hparams_out", default="results/dev_best_hparams.json")
    args = ap.parse_args()

    device = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if device in ["cuda", "mps"] else T.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if device != "cuda":
        model.to(device)
    model.eval()

    splits = load_splits(args.splits)
    E, D, d, delta = load_sae_data(args.scores)

    rows = []

    def run_one(alpha, tau, topk):
        feat_ids = select_features_by_criterion(d, delta, topk=topk, criterion="cohens_d")
        E_t = T.tensor(E, dtype=T.float32, device=device)
        steering_fn = lambda h: apply_sae_encoder_steering(h, E_t, feat_ids, alpha, tau)
        handle = register_steering_hook(model, steering_fn, [], args.layer_index)
        try:
            c_acc, s_acc = eval_dev(model, tok, device, splits)
        finally:
            handle.remove()
        row = {"method": "sae_encoder", "alpha": alpha, "tau": tau, "topk": topk,
               "crows_dev_acc": round(c_acc, 4), "stereo_dev_acc": round(s_acc, 4)}
        print(row)
        rows.append(row)
        return c_acc

    # baseline (dev)
    c0, s0 = eval_dev(model, tok, device, splits)
    rows.append({"method": "baseline", "alpha": 0.0, "tau": None, "topk": None,
                 "crows_dev_acc": round(c0, 4), "stereo_dev_acc": round(s0, 4)})
    print(rows[-1])

    # --- coordinate-ascent sweep for sae_encoder ---
    center_alpha, center_tau, center_topk = 2.0, 0.05, 32

    best_alpha, best_alpha_acc = center_alpha, -1
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        acc = run_one(alpha, center_tau, center_topk)
        if acc > best_alpha_acc:
            best_alpha_acc, best_alpha = acc, alpha

    best_tau, best_tau_acc = center_tau, -1
    for tau in [0.0, 0.02, 0.03, 0.05, 0.08, 0.1]:
        if tau == center_tau:
            acc = best_alpha_acc if best_alpha == center_alpha else run_one(best_alpha, tau, center_topk)
        else:
            acc = run_one(best_alpha, tau, center_topk)
        if acc > best_tau_acc:
            best_tau_acc, best_tau = acc, tau

    best_topk, best_topk_acc = center_topk, -1
    for topk in [8, 16, 24, 32, 48, 64, 128]:
        if topk == center_topk:
            acc = best_tau_acc if best_tau == center_tau else run_one(best_alpha, best_tau, topk)
        else:
            acc = run_one(best_alpha, best_tau, topk)
        if acc > best_topk_acc:
            best_topk_acc, best_topk = acc, topk

    print(f"\nBest SAE-encoder (dev): alpha={best_alpha}, tau={best_tau}, topk={best_topk}, "
          f"crows_dev_acc={best_topk_acc}")

    # --- direct-activation alpha sweep (dev) ---
    direct_vec = build_direct_steering_vector(model, tok, device, args.biased_txt, args.neutral_txt, args.layer_index)
    best_direct_alpha, best_direct_acc = 0.0, c0
    for alpha in [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]:
        vec = direct_vec.to(device)
        steering_fn = lambda h: apply_direct_activation_steering(h, vec, alpha)
        handle = register_steering_hook(model, steering_fn, [], args.layer_index)
        try:
            c_acc, s_acc = eval_dev(model, tok, device, splits)
        finally:
            handle.remove()
        row = {"method": "direct", "alpha": alpha, "tau": None, "topk": None,
               "crows_dev_acc": round(c_acc, 4), "stereo_dev_acc": round(s_acc, 4)}
        print(row)
        rows.append(row)
        if c_acc > best_direct_acc:
            best_direct_acc, best_direct_alpha = c_acc, alpha

    print(f"Best direct-activation (dev): alpha={best_direct_alpha}, crows_dev_acc={best_direct_acc}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out, index=False)

    best = {
        "sae_encoder": {"alpha": best_alpha, "tau": best_tau, "topk": best_topk},
        "direct": {"alpha": best_direct_alpha},
    }
    with open(args.hparams_out, "w") as f:
        json.dump(best, f, indent=2)
    print("saved", args.out, "and", args.hparams_out)


if __name__ == "__main__":
    main()
