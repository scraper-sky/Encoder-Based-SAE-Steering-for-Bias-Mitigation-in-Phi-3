"""
Direct-activation-steering alpha sweep, dev split only. Mirrors the direct
sweep already run for Phi-3 (results/dev_grid_search.csv), which was never
actually run for Gemma -- Gemma's "direct" hyperparameter (alpha=0.0) was
set by assumption from the Phi-3 finding, not measured. This closes that
gap so the "direct steering does nothing" negative control is an actual
empirical result on Gemma too, not a tautological alpha=0 no-op.
"""
import argparse

import pandas as pd
import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.bias_eval import run_crows, run_stereoset
from eval.split_utils import load_splits
from eval.ablation_test import build_direct_steering_vector, apply_direct_activation_steering, register_steering_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--biased_txt", default="data/biased_x5.txt")
    ap.add_argument("--neutral_txt", default="data/neutral_x5.txt")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--alphas", type=float, nargs="+", default=[-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--out", default="results/gemma_direct_only_sweep.csv")
    args = ap.parse_args()

    device = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if device in ["cuda", "mps"] else T.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    splits = load_splits(args.splits)
    crows_idx, stereo_idx = splits["crows"]["dev"], splits["stereoset"]["dev"]

    # baseline (alpha=0, no hook) for reference
    c0, _, _ = run_crows(model, tok, device, indices=crows_idx)
    s0, _, _ = run_stereoset(model, tok, device, indices=stereo_idx)
    rows = [{"method": "baseline", "alpha": 0.0, "crows_dev_acc": round(c0, 4), "stereo_dev_acc": round(s0, 4)}]
    print(rows[-1])

    direct_vec = build_direct_steering_vector(model, tok, device, args.biased_txt, args.neutral_txt, args.layer_index)

    per_example_check_done = False
    for alpha in args.alphas:
        steering_fn = lambda h: apply_direct_activation_steering(h, direct_vec, alpha)
        handle = register_steering_hook(model, steering_fn, [], args.layer_index)

        if not per_example_check_done:
            # Mechanistic sanity check: confirm the hook actually perturbs hidden
            # states at nonzero alpha (catches "wired but not firing" bugs that
            # alpha=0 can't distinguish from a correctly-behaving no-op).
            ids = tok("The nurse comforted the patient.", return_tensors="pt").to(device)
            with T.no_grad():
                out_steered = model(**ids, output_hidden_states=True, use_cache=False)
            handle.remove()
            with T.no_grad():
                out_base = model(**ids, output_hidden_states=True, use_cache=False)
            diff = (out_steered.hidden_states[-1] - out_base.hidden_states[-1]).abs().max().item()
            print(f"[sanity check] max abs hidden-state diff at alpha={alpha}: {diff:.4f} (should be >0)")
            per_example_check_done = True
            handle = register_steering_hook(model, steering_fn, [], args.layer_index)

        try:
            c_acc, _, _ = run_crows(model, tok, device, indices=crows_idx)
            s_acc, _, _ = run_stereoset(model, tok, device, indices=stereo_idx)
        finally:
            handle.remove()
        row = {"method": "direct", "alpha": alpha, "crows_dev_acc": round(c_acc, 4), "stereo_dev_acc": round(s_acc, 4)}
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    best = df[df["method"] == "direct"].sort_values("crows_dev_acc", ascending=False).iloc[0]
    best_alpha = best["alpha"] if best["crows_dev_acc"] > c0 else 0.0
    print(f"\nBest direct-activation alpha (dev): {best_alpha} "
          f"(baseline crows_dev_acc={c0:.4f}, best tested={best['crows_dev_acc']:.4f})")
    print("saved", args.out)


if __name__ == "__main__":
    main()
