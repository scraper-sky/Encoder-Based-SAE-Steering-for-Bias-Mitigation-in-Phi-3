import argparse, os
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def token_avg_logprob(model, tok, text, device):
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, labels=ids["input_ids"])
    return -out.loss.item()

def run_crows(model, tok, device, n=None):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/crows_pairs", trust_remote_code=True)["test"]
    if n: ds = ds.select(range(min(n, len(ds))))
    wins = 0
    for ex in tqdm(ds, desc="CrowS"):
        more, less = ex["sent_more"], ex["sent_less"]
        l_more = token_avg_logprob(model, tok, more, device)
        l_less = token_avg_logprob(model, tok, less, device)
        wins += int(l_less > l_more)
    return wins / len(ds)

def run_stereoset(model, tok, device, n=None):
    from datasets import load_dataset
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence", trust_remote_code=True)["validation"]
    if n: ds = ds.select(range(min(n, len(ds))))
    LABEL_MAP = {0: "stereotype", 1: "anti-stereotype", 2: "unrelated"}
    anti, total = 0, 0
    for ex in tqdm(ds, desc="StereoSet"):
        S = ex.get("sentences")
        lab2sent = {}
        if isinstance(S, list):
            for d in S:
                if isinstance(d, dict):
                    lab = d.get("gold_label", d.get("label"))
                    if isinstance(lab, int): lab = LABEL_MAP.get(lab)
                    if lab in ("stereotype","anti-stereotype","unrelated"):
                        lab2sent[lab] = d.get("sentence")
        elif isinstance(S, dict):
            for sent, lab in zip(S.get("sentence", []), S.get("gold_label", S.get("label", []))):
                lab = LABEL_MAP.get(lab, lab)
                if lab in ("stereotype","anti-stereotype","unrelated"):
                    lab2sent[lab] = sent
        if not all(k in lab2sent for k in ("stereotype","anti-stereotype","unrelated")):
            continue
        s,a,r = lab2sent["stereotype"], lab2sent["anti-stereotype"], lab2sent["unrelated"]
        ls = token_avg_logprob(model, tok, s, device)
        la = token_avg_logprob(model, tok, a, device)
        lr = token_avg_logprob(model, tok, r, device)
        best = max((ls,"S"), (la,"A"), (lr,"R"), key=lambda x: x[0])[1]
        anti += int(best == "A"); total += 1
    return (anti/total) if total else 0.0

def mean_rep(model, tok, device, texts, layer_index=-2):
    reps = []
    for t in texts:
        ids = tok(t, return_tensors="pt").to(device)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True, use_cache=False).hidden_states[layer_index]
        reps.append(hs[0, -1].detach().float().cpu())
    X = torch.stack(reps, dim=0)
    return X.mean(dim=0)

def build_vector(model, tok, device, biased_txt, neutral_txt, layer_index=-2):
    with open(biased_txt) as f:
        biased = [l.strip() for l in f if l.strip()]
    with open(neutral_txt) as f:
        neutral = [l.strip() for l in f if l.strip()]
    if not biased or not neutral:
        raise ValueError(f"Empty bucket(s): {biased_txt}={len(biased)} lines, {neutral_txt}={len(neutral)} lines.")
    mu_b = mean_rep(model, tok, device, biased, layer_index)
    mu_n = mean_rep(model, tok, device, neutral, layer_index)
    v = mu_b - mu_n
    return v / (v.norm() + 1e-8)

def apply_steering_hook(model, vec, alpha, layer_index=-2):
    # Add α·v right before lm_head projection (guaranteed to affect logits)
    v = vec.to(next(model.parameters()).device)
    def pre_hook(_mod, inputs):
        h = inputs[0]              # [B, S, H]
        return (h + alpha * v,)    # same shape
    return model.lm_head.register_forward_pre_hook(pre_hook)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--biased_txt", default="data/biased.txt")
    ap.add_argument("--neutral_txt", default="data/neutral.txt")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--alphas", type=float, nargs="+", default=[-2, -1, -0.5, 0, 0.5, 1, 2])
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--out", default="results/steer_sweep.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if device != "cuda": model.to(device)
    model.eval()

    vec = build_vector(model, tok, device, args.biased_txt, args.neutral_txt, layer_index=args.layer_index)

    rows = []
    for a in args.alphas:
        handle = apply_steering_hook(model, vec, alpha=a, layer_index=args.layer_index)
        try:
            c = run_crows(model, tok, device, n=args.samples)
            s = run_stereoset(model, tok, device, n=args.samples)
        finally:
            handle.remove()
        rows.append({"alpha": a, "crows_anti_acc": round(c,4), "stereo_anti_acc": round(s,4)})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
