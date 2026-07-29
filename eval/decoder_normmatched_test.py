"""
Norm-matched decoder-direction ablation. SAE encoder rows and decoder
columns are not norm-matched by construction -- empirically verified:
the custom Phi-3 SAE's encoder rows are ~1.73x longer than its decoder
columns on average (std of the ratio across features: 0.02, i.e. a fairly
uniform global scale factor); Gemma Scope's decoder columns are exactly
unit-norm by training constraint while its encoder rows vary 0.78x-3.6x
per feature. This rescales each selected feature's decoder column to
match its encoder row's norm before subtracting, isolating *direction*
from *edit magnitude* in the encoder-vs-decoder comparison, and reports a
direct paired McNemar test between encoder and norm-matched decoder on
the held-out test split.
"""
import argparse
import json
import math

import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.bias_eval import run_crows, run_stereoset
from eval.split_utils import load_splits
from eval.ablation_test import load_sae_data, select_features_by_criterion, apply_sae_encoder_steering, register_steering_hook


def apply_sae_decoder_steering_norm_matched(h, E_t, D_t, feat_ids, alpha, tau):
    orig_dtype = h.dtype
    h32 = h.float()
    z = T.relu(T.einsum("bsh,fh->bsf", h32, E_t))
    z_sel = z[..., feat_ids]
    if tau > 0:
        mask = (z_sel > tau).float()
    else:
        mask = (z_sel > 0).float()
    Esel = E_t[feat_ids, :]   # [K,H]
    Dsel = D_t[:, feat_ids]   # [H,K]
    E_norms = Esel.norm(dim=1)   # [K]
    D_norms = Dsel.norm(dim=0)   # [K]
    scale = E_norms / (D_norms + 1e-8)   # [K]
    Dsel_matched = Dsel * scale.unsqueeze(0)  # each column k now has norm E_norms[k]
    contrib = T.einsum("bsk,hk->bsh", mask * z_sel * (-alpha), Dsel_matched)
    return (h32 + contrib).to(orig_dtype)


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
    stereo_test = splits["stereoset"]["test"]

    with open(args.hparams) as f:
        hp = json.load(f)
    alpha, tau, topk = hp["sae_encoder"]["alpha"], hp["sae_encoder"]["tau"], hp["sae_encoder"]["topk"]

    E, D, d, delta = load_sae_data(args.scores)
    feat_ids = select_features_by_criterion(d, delta, topk=topk, criterion="cohens_d")
    E_t = T.tensor(E, dtype=T.float32, device=device)
    D_t = T.tensor(D, dtype=T.float32, device=device)

    Esel = E_t[feat_ids, :]
    Dsel = D_t[:, feat_ids]
    E_norms = Esel.norm(dim=1)
    D_norms = Dsel.norm(dim=0)
    ratio = (E_norms / D_norms)
    print(f"[{args.label}] selected-feature (top-{topk}) norm ratio E/D: "
          f"mean={float(ratio.mean()):.4f} std={float(ratio.std()):.4f} "
          f"min={float(ratio.min()):.4f} max={float(ratio.max()):.4f}")

    steering_fn = lambda h: apply_sae_encoder_steering(h, E_t, feat_ids, alpha, tau)
    handle = register_steering_hook(model, steering_fn, [], args.layer_index)
    try:
        enc_c_acc, _, enc_c_df = run_crows(model, tok, device, indices=crows_test)
        enc_s_acc, _, _ = run_stereoset(model, tok, device, indices=stereo_test)
    finally:
        handle.remove()
    print(f"[{args.label}] encoder-SAE: crows={enc_c_acc:.4f} stereo={enc_s_acc:.4f}")

    steering_fn = lambda h: apply_sae_decoder_steering_norm_matched(h, E_t, D_t, feat_ids, alpha, tau)
    handle = register_steering_hook(model, steering_fn, [], args.layer_index)
    try:
        dec_c_acc, _, dec_c_df = run_crows(model, tok, device, indices=crows_test)
        dec_s_acc, _, _ = run_stereoset(model, tok, device, indices=stereo_test)
    finally:
        handle.remove()
    print(f"[{args.label}] decoder-norm-matched: crows={dec_c_acc:.4f} stereo={dec_s_acc:.4f}")

    enc_win = enc_c_df["anti_win"].tolist()
    dec_win = dec_c_df["anti_win"].tolist()
    n_11 = sum(1 for e, dd in zip(enc_win, dec_win) if e == 1 and dd == 1)
    n_10 = sum(1 for e, dd in zip(enc_win, dec_win) if e == 1 and dd == 0)
    n_01 = sum(1 for e, dd in zip(enc_win, dec_win) if e == 0 and dd == 1)
    n_00 = sum(1 for e, dd in zip(enc_win, dec_win) if e == 0 and dd == 0)
    print(f"[{args.label}] 2x2 (encoder_win, decoder_normmatched_win): "
          f"(1,1)={n_11} (1,0)={n_10} (0,1)={n_01} (0,0)={n_00}")
    p = exact_mcnemar_one_tailed(n_10, n_01)
    print(f"[{args.label}] Exact McNemar one-tailed (H1: encoder beats norm-matched decoder): p={p:.4f}")

    with open(f"results/{args.label}_decoder_normmatched.json", "w") as f:
        json.dump({
            "norm_ratio_mean": float(ratio.mean()), "norm_ratio_std": float(ratio.std()),
            "encoder_crows": enc_c_acc, "encoder_stereo": enc_s_acc,
            "decoder_normmatched_crows": dec_c_acc, "decoder_normmatched_stereo": dec_s_acc,
            "mcnemar_2x2": {"n11": n_11, "n10_encoder_only": n_10, "n01_decoder_only": n_01, "n00": n_00},
            "mcnemar_p_one_tailed": p,
        }, f, indent=2)
    print("saved", f"results/{args.label}_decoder_normmatched.json")


if __name__ == "__main__":
    main()
