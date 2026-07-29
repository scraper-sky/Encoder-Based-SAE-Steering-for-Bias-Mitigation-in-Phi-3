"""
Paired McNemar test directly comparing encoder-direction vs decoder-direction
steering on the held-out CrowS-Pairs test split. The main result only
compares each to baseline separately, which lets a reader infer the
encoder > decoder ordering but doesn't test it directly on the same items.
This scores both conditions on the identical held-out items and tests
whether encoder's advantage over decoder is itself significant.
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
    apply_sae_decoder_steering,
    register_steering_hook,
)


def exact_mcnemar_one_tailed(n_01, n_10):
    n = n_01 + n_10
    if n == 0:
        return float("nan")
    k = n_01
    return sum(math.comb(n, i) * (0.5 ** n) for i in range(k, n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--label", default="phi3")
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
    D_t = T.tensor(D, dtype=T.float32, device=device)

    # encoder-SAE
    steering_fn = lambda h: apply_sae_encoder_steering(h, E_t, feat_ids, alpha, tau)
    handle = register_steering_hook(model, steering_fn, [], args.layer_index)
    try:
        _, _, enc_df = run_crows(model, tok, device, indices=crows_test)
    finally:
        handle.remove()
    enc_win = enc_df["anti_win"].tolist()

    # decoder ablation
    steering_fn = lambda h: apply_sae_decoder_steering(h, E_t, D_t, feat_ids, alpha, tau)
    handle = register_steering_hook(model, steering_fn, [], args.layer_index)
    try:
        _, _, dec_df = run_crows(model, tok, device, indices=crows_test)
    finally:
        handle.remove()
    dec_win = dec_df["anti_win"].tolist()

    assert len(enc_win) == len(dec_win) == len(crows_test)

    n_11 = sum(1 for e, dd in zip(enc_win, dec_win) if e == 1 and dd == 1)
    n_10 = sum(1 for e, dd in zip(enc_win, dec_win) if e == 1 and dd == 0)  # encoder wins, decoder doesn't
    n_01 = sum(1 for e, dd in zip(enc_win, dec_win) if e == 0 and dd == 1)  # decoder wins, encoder doesn't
    n_00 = sum(1 for e, dd in zip(enc_win, dec_win) if e == 0 and dd == 0)

    print(f"[{args.label}] 2x2 table (encoder_win, decoder_win): "
          f"(1,1)={n_11} (1,0)={n_10} (0,1)={n_01} (0,0)={n_00}")
    print(f"encoder acc={sum(enc_win)/len(enc_win):.4f}  decoder acc={sum(dec_win)/len(dec_win):.4f}")
    print(f"discordant pairs: {n_10 + n_01} (encoder-only-win {n_10}, decoder-only-win {n_01})")

    p = exact_mcnemar_one_tailed(n_10, n_01)
    print(f"\nExact McNemar one-tailed test (H1: encoder beats decoder): p={p:.4f}")

    with open(f"results/{args.label}_mcnemar_encoder_vs_decoder.json", "w") as f:
        json.dump({"n_11": n_11, "n_10_encoder_only": n_10, "n_01_decoder_only": n_01, "n_00": n_00,
                   "p_one_tailed": p}, f, indent=2)
    print(f"saved results/{args.label}_mcnemar_encoder_vs_decoder.json")


if __name__ == "__main__":
    main()
