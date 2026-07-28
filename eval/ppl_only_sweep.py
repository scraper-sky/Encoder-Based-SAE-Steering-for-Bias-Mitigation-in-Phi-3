"""
Perplexity-only sweep across the alpha range already accuracy-tested in
results/dev_grid_search.csv (0.5..3.0). eval/alpha_ppl_search.py had a gap:
it only checked perplexity starting at alpha=3.0 (the accuracy-optimal
boundary) and never measured the lower alphas, so when 3.0 failed the cap
there was no fallback data. This fills in perplexity for the full range so
we can pick the largest alpha that stays within a utility budget using
real data instead of a single untested boundary point.
"""
import argparse

import pandas as pd
import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.ablation_test import load_sae_data, select_features_by_criterion, apply_sae_encoder_steering, register_steering_hook
from sae.sae_steer_eval import compute_wikitext2_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--scores", default="results/sae_feature_scores_dev.npy")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--ppl_max_tokens", type=int, default=20000)
    ap.add_argument("--out", default="results/alpha_ppl_full.csv")
    args = ap.parse_args()

    device = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if device in ["cuda", "mps"] else T.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    E, D, d, delta = load_sae_data(args.scores)
    feat_ids = select_features_by_criterion(d, delta, topk=args.topk, criterion="cohens_d")
    E_t = T.tensor(E, dtype=T.float32, device=device)

    baseline_ppl = compute_wikitext2_ppl(tok, model, device, max_len=4096, stride=2048, max_tokens=args.ppl_max_tokens)
    print({"alpha": 0.0, "ppl_wikitext2": round(baseline_ppl, 3)})
    rows = [{"alpha": 0.0, "ppl_wikitext2": round(baseline_ppl, 3), "ppl_rel_increase": 0.0}]

    for alpha in args.alphas:
        steering_fn = lambda h: apply_sae_encoder_steering(h, E_t, feat_ids, alpha, args.tau)
        handle = register_steering_hook(model, steering_fn, [], args.layer_index)
        try:
            ppl = compute_wikitext2_ppl(tok, model, device, max_len=4096, stride=2048, max_tokens=args.ppl_max_tokens)
        finally:
            handle.remove()
        rel_increase = ppl / baseline_ppl - 1
        row = {"alpha": alpha, "ppl_wikitext2": round(ppl, 3), "ppl_rel_increase": round(rel_increase, 4)}
        rows.append(row)
        print(row)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
