import argparse, numpy as np, torch as T
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def grab(model, tok, dev, text, layer_index=-2):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
    with T.no_grad():
        hs = model(**ids, output_hidden_states=True, use_cache=False).hidden_states[layer_index]
    return hs[0, -1].detach().float().cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--sources", default="wikitext-2-raw-v1:test,8000;wikitext-2-raw-v1:train,42000")
    ap.add_argument("--out", default="cache/acts/lm2_train.npy")
    args = ap.parse_args()

    dev = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if dev in ["cuda","mps"] else T.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if dev != "cuda": model.to(dev); model.eval()

    acts = []
    for spec in args.sources.split(";"):
        name, split_count = spec.split(":")
        split, count = split_count.split(",")
        ds = load_dataset("Salesforce/wikitext", name)[split]
        n = min(int(count), len(ds))
        for i in range(n):
            txt = ds[i]["text"].strip()
            if not txt: continue
            acts.append(grab(model, tok, dev, txt, args.layer_index))
    X = np.stack(acts, 0)  # [N,H]
    np.save(args.out, X); print("saved", args.out, X.shape)

if __name__ == "__main__":
    main()
