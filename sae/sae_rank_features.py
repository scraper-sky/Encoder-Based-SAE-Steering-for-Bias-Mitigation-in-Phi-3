import argparse, numpy as np, torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_hidden(model, tok, dev, texts, layer_index=-2):
    H=[]
    for t in texts:
        ids = tok(t, return_tensors="pt").to(dev)
        with T.no_grad():
            hs = model(**ids, output_hidden_states=True, use_cache=False).hidden_states[layer_index]
        H.append(hs[0,-1].detach().float().cpu().numpy())
    return np.stack(H,0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--biased_txt", default="data/biased_x5.txt")
    ap.add_argument("--neutral_txt", default="data/neutral_x5.txt")
    ap.add_argument("--sae", default="cache/sae_lm2.pt")
    ap.add_argument("--out", default="results/sae_feature_scores.npy")
    args = ap.parse_args()

    dev = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if dev in ["cuda","mps"] else T.float32
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if dev != "cuda": model.to(dev); model.eval()

    biased = [l.strip() for l in open(args.biased_txt) if l.strip()]
    neutral= [l.strip() for l in open(args.neutral_txt) if l.strip()]
    Hb = get_hidden(model, tok, dev, biased, args.layer_index)  # [Nb,H]
    Hn = get_hidden(model, tok, dev, neutral, args.layer_index) # [Nn,H]

    ckpt = T.load(args.sae, map_location="cpu")
    E = ckpt["E"].numpy()   # [F,H]
    D = ckpt["D"].numpy()   # [H,F]

    Zb = np.maximum(0, Hb @ E.T)   # [Nb,F]
    Zn = np.maximum(0, Hn @ E.T)   # [Nn,F]

    mu_b, mu_n = Zb.mean(0), Zn.mean(0)
    var = 0.5*(Zb.var(0)+Zn.var(0)) + 1e-8
    d = (mu_b - mu_n) / np.sqrt(var)            # Cohen's d
    cons = (mu_b > mu_n).astype(float)          # simple consistency proxy

    np.save(args.out, {"E":E, "D":D, "d":d, "delta":mu_b-mu_n, "cons":cons}, allow_pickle=True)
    print("saved", args.out)

if __name__ == "__main__":
    main()
