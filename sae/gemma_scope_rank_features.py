"""
Bias-feature scoring for google/gemma-2-2b using a pretrained Gemma Scope
JumpReLU SAE (no SAE training needed). Mirrors sae/sae_rank_features.py's
output schema ({"E","D","d","delta","cons"}) so every downstream script
(eval/dev_grid_search.py, eval/final_test_eval.py) works unmodified on
either model.

Gemma Scope params.npz layout: W_enc [H,F], W_dec [F,H], b_enc [F], b_dec [H],
threshold [F] (JumpReLU gate: feature fires iff pre-activation > threshold).
We transpose to our convention: E [F,H] (row f = encoder direction),
D [H,F] (col f = decoder direction) -- matching sae/sae_train.py's SAE.
"""
import argparse

import numpy as np
import torch as T
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_hidden(model, tok, dev, texts, layer_index=-2):
    H = []
    for t in texts:
        ids = tok(t, return_tensors="pt").to(dev)
        with T.no_grad():
            hs = model(**ids, output_hidden_states=True, use_cache=False).hidden_states[layer_index]
        H.append(hs[0, -1].detach().float().cpu().numpy())
    return np.stack(H, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--layer_index", type=int, default=-2,
                     help="hidden_states index; -2 = output of block 24/26 for gemma-2-2b, matching sae_repo layer_24")
    ap.add_argument("--sae_repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--sae_path", default="layer_24/width_16k/average_l0_38/params.npz")
    ap.add_argument("--biased_txt", default="data/biased_crows_stereoset_dev.txt")
    ap.add_argument("--neutral_txt", default="data/neutral_crows_stereoset_dev.txt")
    ap.add_argument("--out", default="results/gemma_sae_feature_scores_dev.npy")
    args = ap.parse_args()

    dev = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if dev in ["cuda", "mps"] else T.float32
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(dev)
    model.eval()

    biased = [l.strip() for l in open(args.biased_txt) if l.strip()]
    neutral = [l.strip() for l in open(args.neutral_txt) if l.strip()]
    Hb = get_hidden(model, tok, dev, biased, args.layer_index)   # [Nb,H]
    Hn = get_hidden(model, tok, dev, neutral, args.layer_index)  # [Nn,H]

    params_path = hf_hub_download(repo_id=args.sae_repo, filename=args.sae_path)
    p = np.load(params_path)
    W_enc, W_dec = p["W_enc"], p["W_dec"]     # [H,F], [F,H]
    b_enc, threshold = p["b_enc"], p["threshold"]

    def encode(H):
        z_pre = H @ W_enc + b_enc            # [N,F]
        return z_pre * (z_pre > threshold)   # JumpReLU gate

    Zb, Zn = encode(Hb), encode(Hn)
    mu_b, mu_n = Zb.mean(0), Zn.mean(0)
    var = 0.5 * (Zb.var(0) + Zn.var(0)) + 1e-8
    d = (mu_b - mu_n) / np.sqrt(var)
    cons = (mu_b > mu_n).astype(float)

    E = W_enc.T   # [F,H] row f = encoder direction
    D = W_dec.T   # [H,F] col f = decoder direction

    np.save(args.out, {"E": E, "D": D, "d": d, "delta": mu_b - mu_n, "cons": cons}, allow_pickle=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
