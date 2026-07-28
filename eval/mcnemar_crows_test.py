"""
Paired significance test for baseline vs. encoder-SAE steering on the
held-out CrowS-Pairs test split. eval/final_test_eval.py's z-test treats
baseline and steered accuracy as independent samples, but they're actually
the same items scored twice (once unsteered, once steered) -- a paired
design. McNemar's exact test uses that pairing and is the statistically
correct (and typically more powerful) test here.

Reuses the same model load + steering hook as final_test_eval.py, but only
scores CrowS (not StereoSet) under baseline and encoder-SAE conditions,
since that's the comparison the headline significance claim is about.
"""
import argparse
import json
import math

import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.bias_eval import run_crows
from eval.split_utils import load_splits
from eval.ablation_test import (
    load_sae_data,
    select_features_by_criterion,
    apply_sae_encoder_steering,
    register_steering_hook,
)


def exact_mcnemar_one_tailed(n_01, n_10):
    """H1: condition favoring index 1 (n_01) is more common than n_10."""
    n = n_01 + n_10
    if n == 0:
        return float("nan")
    k = n_01
    p = sum(math.comb(n, i) * (0.5 ** n) for i in range(k, n + 1))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--scores", default="results/sae_feature_scores_dev.npy")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--hparams", default="results/dev_best_hparams.json")
    ap.add_argument("--layer_index", type=int, default=-2)
    args = ap.parse_args()

    device = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if device in ["cuda", "mps"] else T.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    splits = load_splits(args.splits)
    crows_test = splits["crows"]["test"]

    with open(args.hparams) as f:
        hp = json.load(f)
    alpha, tau, topk = hp["sae_encoder"]["alpha"], hp["sae_encoder"]["tau"], hp["sae_encoder"]["topk"]

    E, D, d, delta = load_sae_data(args.scores)
    feat_ids = select_features_by_criterion(d, delta, topk=topk, criterion="cohens_d")
    E_t = T.tensor(E, dtype=T.float32, device=device)

    # baseline
    _, _, base_df = run_crows(model, tok, device, indices=crows_test)
    base_win = base_df["anti_win"].tolist()

    # encoder-SAE steer
    steering_fn = lambda h: apply_sae_encoder_steering(h, E_t, feat_ids, alpha, tau)
    handle = register_steering_hook(model, steering_fn, [], args.layer_index)
    try:
        _, _, steer_df = run_crows(model, tok, device, indices=crows_test)
    finally:
        handle.remove()
    steer_win = steer_df["anti_win"].tolist()

    assert len(base_win) == len(steer_win) == len(crows_test)

    n_11 = sum(1 for b, s in zip(base_win, steer_win) if b == 1 and s == 1)
    n_10 = sum(1 for b, s in zip(base_win, steer_win) if b == 1 and s == 0)  # steering made it worse
    n_01 = sum(1 for b, s in zip(base_win, steer_win) if b == 0 and s == 1)  # steering made it better
    n_00 = sum(1 for b, s in zip(base_win, steer_win) if b == 0 and s == 0)

    print(f"2x2 table (baseline_win, steer_win): "
          f"(1,1)={n_11} (1,0)={n_10} (0,1)={n_01} (0,0)={n_00}")
    print(f"discordant pairs: {n_10 + n_01} (steering helped {n_01}, steering hurt {n_10})")

    p_one_tailed = exact_mcnemar_one_tailed(n_01, n_10)
    print(f"\nExact McNemar one-tailed test (H1: steering improves anti-stereotype rate): p={p_one_tailed:.4f}")

    chi2 = (abs(n_10 - n_01) - 1) ** 2 / (n_10 + n_01) if (n_10 + n_01) > 0 else float("nan")
    print(f"(continuity-corrected McNemar chi2 = {chi2:.3f}, two-tailed approx p = {math.erfc(math.sqrt(chi2/2)):.4f})")


if __name__ == "__main__":
    main()
