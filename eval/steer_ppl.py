import argparse, math, numpy as np, torch as T
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----- reuse: build mean-difference vector and lm_head pre-hook -----

def mean_rep(model, tok, dev, texts, layer_index=-2):
    reps=[]
    for t in texts:
        ids = tok(t, return_tensors="pt").to(dev)
        with T.no_grad():
            hs = model(**ids, output_hidden_states=True, use_cache=False).hidden_states[layer_index]
        reps.append(hs[0,-1].detach().float().cpu())
    X = T.stack(reps, 0)
    return X.mean(0)

def build_vector(model, tok, dev, biased_txt, neutral_txt, layer_index=-2):
    with open(biased_txt) as f:  biased = [l.strip() for l in f if l.strip()]
    with open(neutral_txt) as f: neutral = [l.strip() for l in f if l.strip()]
    if not biased or not neutral:
        raise ValueError("Buckets empty; fill data/biased*.txt and data/neutral*.txt.")
    mu_b = mean_rep(model, tok, dev, biased, layer_index)
    mu_n = mean_rep(model, tok, dev, neutral, layer_index)
    v = mu_b - mu_n
    return v / (v.norm() + 1e-8)

def apply_steering_hook(model, vec, alpha):
    v = vec.to(next(model.parameters()).device)
    def pre_hook(_mod, inputs):
        h = inputs[0]               # [B,S,H]
        return (h + alpha * v,)     # add before lm_head
    return model.lm_head.register_forward_pre_hook(pre_hook)

# ----- perplexity on WikiText-2 (test), token-capped for speed -----

def calc_ppl(model, tok, dev, max_tokens=50000):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]
    text = "\n\n".join([ex["text"] for ex in ds if ex["text"].strip()])
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0]  # [T]
    if max_tokens and input_ids.numel() > max_tokens:
        input_ids = input_ids[:max_tokens]

    input_ids = input_ids.to(dev)
    block = min(getattr(tok, "model_max_length", 4096), 4096)  # Phi-3 is 4k context
    total_loss, total_tokens = 0.0, 0

    for i in range(0, input_ids.numel() - 1, block):
        chunk = input_ids[i : i + block]
        # labels = same ids; model computes next-token loss internally
        with T.no_grad():
            out = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
        # out.loss is mean over tokens in this chunk; scale by length
        n_tok = int(chunk.numel())
        total_loss += float(out.loss) * n_tok
        total_tokens += n_tok

    if total_tokens == 0:
        return float("inf")
    nll = total_loss / total_tokens
    return math.exp(nll)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--biased_txt", default="data/biased_x5.txt")   # use amplified buckets
    ap.add_argument("--neutral_txt", default="data/neutral_x5.txt")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--max_tokens", type=int, default=20000)        # ~fast estimate
    args = ap.parse_args()

    dev = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if dev in ["cuda","mps"] else T.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if dev != "cuda": model.to(dev)
    model.eval()

    handle = None
    if abs(args.alpha) > 0:
        vec = build_vector(model, tok, dev, args.biased_txt, args.neutral_txt, layer_index=args.layer_index)
        handle = apply_steering_hook(model, vec, args.alpha)

    try:
        ppl = calc_ppl(model, tok, dev, max_tokens=args.max_tokens)
    finally:
        if handle is not None: handle.remove()

    print({"alpha": args.alpha, "ppl_wikitext2": round(ppl, 3)})

if __name__ == "__main__":
    main()
