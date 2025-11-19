import argparse, math, torch, pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def ppl_on_text(model, tok, device, text, chunk=1024):
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    nll, count = 0.0, 0
    for i in range(0, ids.shape[1]-1, chunk):
        inp = ids[:, i:i+chunk]
        with torch.no_grad():
            out = model(input_ids=inp, labels=inp)
        nll += out.loss.item() * inp.shape[1]
        count += inp.shape[1]
    return math.exp(nll / count)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--out", default="results/baseline_utility.csv")
    ap.add_argument("--wiki_samples", type=int, default=20000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if device != "cuda":
        model.to(device)

    wt = load_dataset("wikitext", "wikitext-2-raw-v1")["test"]
    text = "\n".join(wt[i]["text"] for i in range(min(args.wiki_samples, len(wt))))
    ppl = ppl_on_text(model, tok, device, text)
    pd.DataFrame([{"model": args.model, "ppl_wikitext2": round(ppl,3)}]).to_csv(args.out, index=False)
    print({"ppl_wikitext2": round(ppl,3)})

if __name__ == "__main__":
    main()
