"""
Extend the encoder-SAE alpha search past the dev_grid_search.py boundary,
with a WikiText-2 perplexity guardrail: CrowS dev accuracy was still
climbing monotonically at alpha=3.0 (the edge of the first sweep) while
StereoSet dev accuracy fell, so picking alpha=3.0 outright risks reporting
a boundary-clipped "optimum" with no check on whether the model is still
coherent. This sweeps alpha upward (still dev-only -- test split untouched)
and stops once perplexity rises past --ppl_increase_cap relative to the
unsteered baseline, then writes the largest alpha within budget to
--hparams_out.

Fixes K at --topk (default 32, not the K=128 dev_grid_search picked --
that gain was ~3/603 dev examples, i.e. noise) and tau at --tau (0.0,
since the first sweep showed tau has no effect at this alpha range).
"""
import argparse
import json

import pandas as pd
import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.bias_eval import run_crows, run_stereoset
from eval.split_utils import load_splits
from eval.ablation_test import (
    load_sae_data,
    select_features_by_criterion,
    apply_sae_encoder_steering,
    register_steering_hook,
)
from sae.sae_steer_eval import compute_wikitext2_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--scores", default="results/sae_feature_scores_dev.npy")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    ap.add_argument("--ppl_max_tokens", type=int, default=20000)
    ap.add_argument("--ppl_increase_cap", type=float, default=0.30,
                     help="stop once (ppl/baseline_ppl - 1) exceeds this")
    ap.add_argument("--out", default="results/alpha_ppl_search.csv")
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
    crows_idx, stereo_idx = splits["crows"]["dev"], splits["stereoset"]["dev"]

    E, D, d, delta = load_sae_data(args.scores)
    feat_ids = select_features_by_criterion(d, delta, topk=args.topk, criterion="cohens_d")
    E_t = T.tensor(E, dtype=T.float32, device=device)

    baseline_ppl = compute_wikitext2_ppl(tok, model, device, max_len=4096, stride=2048, max_tokens=args.ppl_max_tokens)
    print({"alpha": 0.0, "ppl_wikitext2": round(baseline_ppl, 3)})

    rows = [{"alpha": 0.0, "crows_dev_acc": None, "stereo_dev_acc": None,
             "ppl_wikitext2": round(baseline_ppl, 3), "ppl_rel_increase": 0.0, "within_cap": True}]

    last_good_alpha = 0.0
    for alpha in args.alphas:
        steering_fn = lambda h: apply_sae_encoder_steering(h, E_t, feat_ids, alpha, args.tau)
        handle = register_steering_hook(model, steering_fn, [], args.layer_index)
        try:
            c_acc, _, _ = run_crows(model, tok, device, indices=crows_idx)
            s_acc, _, _ = run_stereoset(model, tok, device, indices=stereo_idx)
            ppl = compute_wikitext2_ppl(tok, model, device, max_len=4096, stride=2048, max_tokens=args.ppl_max_tokens)
        finally:
            handle.remove()

        rel_increase = ppl / baseline_ppl - 1
        within_cap = rel_increase <= args.ppl_increase_cap
        row = {"alpha": alpha, "crows_dev_acc": round(c_acc, 4), "stereo_dev_acc": round(s_acc, 4),
               "ppl_wikitext2": round(ppl, 3), "ppl_rel_increase": round(rel_increase, 4), "within_cap": within_cap}
        rows.append(row)
        print(row)

        if within_cap:
            last_good_alpha = alpha
        else:
            print(f"stopping: alpha={alpha} exceeds ppl cap ({rel_increase:.1%} > {args.ppl_increase_cap:.0%})")
            break

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    with open(args.hparams_out) as f:
        hp = json.load(f)
    hp["sae_encoder"] = {"alpha": last_good_alpha, "tau": args.tau, "topk": args.topk}
    with open(args.hparams_out, "w") as f:
        json.dump(hp, f, indent=2)

    print(f"\nFinal alpha (within {args.ppl_increase_cap:.0%} ppl cap): {last_good_alpha}, K={args.topk}, tau={args.tau}")
    print("saved", args.out, "and", args.hparams_out)


if __name__ == "__main__":
    main()
