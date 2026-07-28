"""
Single, held-out-test-only evaluation of baseline / encoder-SAE / decoder-SAE /
direct-activation steering. Hyperparameters must come from a dev-only search
(eval/dev_grid_search.py) -- this script does not tune anything, it just runs
each method ONCE on the TEST split and reports the significance test that
belongs in the paper. Works for any causal LM + any SAE saved in the
{"E","D","d","delta"} schema (sae/sae_rank_features.py or
sae/gemma_scope_rank_features.py), so it's reused for both the Phi-3 rerun
and the Gemma-2-2B/Gemma-Scope replication.
"""
import argparse
import json
import math

import pandas as pd
import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.bias_eval import run_crows, run_stereoset
from eval.split_utils import load_splits
from eval.ablation_test import (
    load_sae_data,
    select_features_by_criterion,
    apply_sae_encoder_steering,
    apply_sae_decoder_steering,
    apply_direct_activation_steering,
    register_steering_hook,
    build_direct_steering_vector,
)


def two_prop_ztest_one_tailed(x1, n1, x2, n2):
    """H1: p2 > p1 (steered anti-stereotype rate exceeds baseline)."""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p2 - p1) / se
    # one-tailed p-value via error-function-based normal CDF (no scipy dependency)
    p_value = 0.5 * (1 - math.erf(z / math.sqrt(2)))
    return z, p_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--label", default="phi3", help="tag used in the output filenames/rows")
    ap.add_argument("--scores", default="results/sae_feature_scores_dev.npy")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--hparams", default="results/dev_best_hparams.json",
                     help="output of eval/dev_grid_search.py")
    ap.add_argument("--biased_txt", default="data/biased_x5.txt")
    ap.add_argument("--neutral_txt", default="data/neutral_x5.txt")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--out", default="results/final_test_eval.csv")
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
    n_crows, n_stereo = len(crows_test), len(stereo_test)

    with open(args.hparams) as f:
        hp = json.load(f)
    alpha, tau, topk = hp["sae_encoder"]["alpha"], hp["sae_encoder"]["tau"], hp["sae_encoder"]["topk"]
    direct_alpha = hp["direct"]["alpha"]

    E, D, d, delta = load_sae_data(args.scores)
    feat_ids = select_features_by_criterion(d, delta, topk=topk, criterion="cohens_d")
    E_t = T.tensor(E, dtype=T.float32, device=device)
    D_t = T.tensor(D, dtype=T.float32, device=device)
    direct_vec = build_direct_steering_vector(model, tok, device, args.biased_txt, args.neutral_txt, args.layer_index)

    rows = []
    counts = {}  # method -> (crows_anti_wins, crows_n, stereo_anti_wins, stereo_n)

    def evaluate(method_name, handle):
        try:
            c_acc, _, c_df = run_crows(model, tok, device, indices=crows_test)
            s_acc, _, s_df = run_stereoset(model, tok, device, indices=stereo_test)
        finally:
            if handle is not None:
                handle.remove()
        c_wins = int(c_df["anti_win"].sum())
        row = {"method": method_name, "crows_test_acc": round(c_acc, 4), "stereo_test_acc": round(s_acc, 4),
               "n_crows_test": n_crows, "n_stereo_test": n_stereo}
        rows.append(row)
        counts[method_name] = (c_wins, n_crows)
        print(row)

    # baseline
    evaluate("baseline", None)

    # encoder-SAE steer
    steering_fn = lambda h: apply_sae_encoder_steering(h, E_t, feat_ids, alpha, tau)
    handle = register_steering_hook(model, steering_fn, [], args.layer_index)
    evaluate(f"sae_encoder(alpha={alpha},tau={tau},topk={topk})", handle)

    # decoder-direction ablation
    steering_fn = lambda h: apply_sae_decoder_steering(h, E_t, D_t, feat_ids, alpha, tau)
    handle = register_steering_hook(model, steering_fn, [], args.layer_index)
    evaluate(f"sae_decoder(alpha={alpha},tau={tau},topk={topk})", handle)

    # direct activation steering
    steering_fn = lambda h: apply_direct_activation_steering(h, direct_vec.to(device), direct_alpha)
    handle = register_steering_hook(model, steering_fn, [], args.layer_index)
    evaluate(f"direct(alpha={direct_alpha})", handle)

    # significance: baseline vs encoder-SAE steer, CrowS test only
    x1, n1 = counts["baseline"]
    x2, n2 = counts[rows[1]["method"]]
    z, p = two_prop_ztest_one_tailed(x1, n1, x2, n2)
    print(f"\nOne-tailed z-test (baseline vs encoder-SAE steer, CrowS TEST n={n1}): z={z:.3f}, p={p:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    with open(args.out.replace(".csv", "_ztest.json"), "w") as f:
        json.dump({"z": z, "p": p, "n_crows_test": n1, "n_stereo_test": n_stereo, "label": args.label}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
