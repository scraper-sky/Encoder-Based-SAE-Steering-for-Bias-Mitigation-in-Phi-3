"""
Qualitative demonstration for the "Interpretability for Discovery" workshop:
the paper claims the top-K bias-sensitive SAE features are "monosemantic,
interpretable" but only supports that by citation, never by showing an
actual feature. This script shows two things instead of just asserting them:

1. Max-activating tokens for the top-N bias-sensitive features (by |Cohen's
   d|), scanned over the dev bias/neutral corpus -- what do these features
   actually fire on?
2. Concrete CrowS-Pairs examples where baseline prefers the stereotype
   completion but encoder-SAE steering flips the preference to
   anti-stereotype, with before/after logprobs -- a worked example of what
   the intervention is doing, not just an aggregate accuracy number.

Quick/illustrative by design (small corpus sample, single feature scan),
not a new statistical claim -- the held-out test numbers are what carry the
paper's actual result.
"""
import argparse
import json

import torch as T
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.ablation_test import load_sae_data, select_features_by_criterion, apply_sae_encoder_steering, register_steering_hook
from eval.bias_eval import token_avg_logprob
from eval.split_utils import load_splits


def get_per_token_hidden(model, tok, device, text, layer_index=-2, max_tokens=64):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
    with T.no_grad():
        hs = model(**ids, output_hidden_states=True, use_cache=False).hidden_states[layer_index]
    tokens = tok.convert_ids_to_tokens(ids["input_ids"][0])
    return tokens, hs[0].float().cpu()  # [seq, H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--scores", default="results/sae_feature_scores_dev.npy")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--topn_features", type=int, default=4)
    ap.add_argument("--corpus_txt", nargs="+",
                     default=["data/biased_crows_stereoset_dev.txt", "data/neutral_crows_stereoset_dev.txt"])
    ap.add_argument("--corpus_limit", type=int, default=300)
    ap.add_argument("--topk_per_feature", type=int, default=5)
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--hparams", default="results/dev_best_hparams.json")
    ap.add_argument("--n_flip_examples", type=int, default=5)
    ap.add_argument("--flip_search_n", type=int, default=150, help="how many test-split CrowS pairs to scan for flips")
    args = ap.parse_args()

    device = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if device in ["cuda", "mps"] else T.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    E, D, d, delta = load_sae_data(args.scores)
    top_feat_ids = select_features_by_criterion(d, delta, topk=args.topn_features, criterion="cohens_d")
    E_cpu = T.tensor(E, dtype=T.float32)  # for part 1 (CPU-side per-token scan)
    E_t = T.tensor(E, dtype=T.float32, device=device)  # for part 2 (steering hook)

    print("=== Part 1: Max-activating tokens for top bias-sensitive features ===\n")

    sentences = []
    for path in args.corpus_txt:
        with open(path) as f:
            sentences.extend([l.strip() for l in f if l.strip()])
    sentences = sentences[:args.corpus_limit]

    feature_hits = {f: [] for f in top_feat_ids}
    for sent in sentences:
        tokens, h = get_per_token_hidden(model, tok, device, sent, args.layer_index)
        z = T.relu(h @ E_cpu.T)  # [seq, F]
        for f in top_feat_ids:
            acts = z[:, f]
            max_idx = int(acts.argmax())
            max_val = float(acts[max_idx])
            if max_val > 0:
                feature_hits[f].append((max_val, sent, tokens[max_idx]))

    for f in top_feat_ids:
        hits = sorted(feature_hits[f], key=lambda x: -x[0])[:args.topk_per_feature]
        print(f"--- Feature {f} (Cohen's d={d[f]:.3f}) ---")
        for val, sent, token in hits:
            print(f"  act={val:.3f} | token='{token}' | \"{sent}\"")
        print()

    print("\n=== Part 2: Concrete steering-flip examples (CrowS-Pairs, held-out test) ===\n")

    with open(args.hparams) as f:
        hp = json.load(f)
    alpha, tau, topk = hp["sae_encoder"]["alpha"], hp["sae_encoder"]["tau"], hp["sae_encoder"]["topk"]
    feat_ids = select_features_by_criterion(d, delta, topk=topk, criterion="cohens_d")

    splits = load_splits(args.splits)
    crows_ds = load_dataset("nyu-mll/crows_pairs", trust_remote_code=True)["test"]
    test_examples = crows_ds.select(splits["crows"]["test"][:args.flip_search_n])

    steering_fn = lambda h: apply_sae_encoder_steering(h, E_t, feat_ids, alpha, tau)

    found = 0
    for ex in test_examples:
        more, less = ex["sent_more"], ex["sent_less"]

        l_more_base = token_avg_logprob(model, tok, more, device)
        l_less_base = token_avg_logprob(model, tok, less, device)
        base_prefers_stereo = l_more_base > l_less_base

        handle = register_steering_hook(model, steering_fn, [], args.layer_index)
        try:
            l_more_steer = token_avg_logprob(model, tok, more, device)
            l_less_steer = token_avg_logprob(model, tok, less, device)
        finally:
            handle.remove()
        steer_prefers_stereo = l_more_steer > l_less_steer

        if base_prefers_stereo and not steer_prefers_stereo:
            found += 1
            print(f"FLIP #{found}:")
            print(f"  Stereotype : \"{more}\"")
            print(f"  Anti-stereo: \"{less}\"")
            print(f"  Baseline  logprob(stereo)={l_more_base:.3f} vs logprob(anti)={l_less_base:.3f}  -> prefers STEREOTYPE")
            print(f"  Steered   logprob(stereo)={l_more_steer:.3f} vs logprob(anti)={l_less_steer:.3f}  -> prefers ANTI-STEREOTYPE")
            print()
            if found >= args.n_flip_examples:
                break

    if found == 0:
        print("No flip examples found in the sampled subset -- try increasing --flip_search_n.")


if __name__ == "__main__":
    main()
