import argparse, math, random
import numpy as np, pandas as pd, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def token_avg_logprob(model, tok, text, device):
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, labels=ids["input_ids"])
    return -out.loss.item()  # avg per-token logprob

def run_crows(model, tok, device, n=None, indices=None):
    ds = load_dataset("nyu-mll/crows_pairs", trust_remote_code=True)["test"]
    if indices is not None:
        ds = ds.select(indices)
    elif n:
        ds = ds.select(range(min(n, len(ds))))
    wins, ties = 0, 0; rows = []
    for ex in tqdm(ds, desc="CrowS"):
        more, less = ex["sent_more"], ex["sent_less"]  # more=stereotype, less=anti
        l_more = token_avg_logprob(model, tok, more, device)
        l_less = token_avg_logprob(model, tok, less, device)
        anti_win = 1 if l_less > l_more else 0
        tie = 1 if abs(l_less - l_more) < 1e-6 else 0
        wins += anti_win; ties += tie
        rows.append({"la_more": l_more, "lb_less": l_less, "anti_win": anti_win, "tie": tie})
    acc = wins / len(ds)
    return acc, ties/len(ds), pd.DataFrame(rows)

def run_stereoset(model, tok, device, n=None, indices=None):
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence", trust_remote_code=True)["validation"]
    if indices is not None:
        ds = ds.select(indices)
    elif n:
        ds = ds.select(range(min(n, len(ds))))
    anti, total, ties = 0, 0, 0
    rows = []

    LABEL_MAP = {0: "stereotype", 1: "anti-stereotype", 2: "unrelated"}

    for ex in tqdm(ds, desc="StereoSet"):
        lab2sent = {}

        S = ex.get("sentences")
        if isinstance(S, list):
            # Format A: list of dicts
            for d in S:
                if isinstance(d, dict):
                    label = d.get("gold_label", d.get("label"))
                    # normalize label
                    if isinstance(label, int):
                        label = LABEL_MAP.get(label)
                    if label in ("stereotype", "anti-stereotype", "unrelated"):
                        lab2sent[label] = d.get("sentence")
        elif isinstance(S, dict):
            # Format B: dict of lists
            sent_list = S.get("sentence", [])
            label_list = S.get("gold_label", S.get("label", []))
            for sent, lab in zip(sent_list, label_list):
                label = LABEL_MAP.get(lab, lab)  # int -> string if needed
                if label in ("stereotype", "anti-stereotype", "unrelated"):
                    lab2sent[label] = sent
        else:
            # Unexpected format; skip
            continue

        if not all(k in lab2sent for k in ("stereotype", "anti-stereotype", "unrelated")):
            continue

        s, a, r = lab2sent["stereotype"], lab2sent["anti-stereotype"], lab2sent["unrelated"]
        ls = token_avg_logprob(model, tok, s, device)
        la = token_avg_logprob(model, tok, a, device)
        lr = token_avg_logprob(model, tok, r, device)

        best = max((ls, "S"), (la, "A"), (lr, "R"), key=lambda x: x[0])[1]
        anti += int(best == "A"); total += 1
        ties += int(len({round(ls,6), round(la,6), round(lr,6)}) < 3)
        rows.append({"ls": ls, "la": la, "lr": lr, "best": best})

    return (anti/total if total else 0.0), (ties/total if total else 0.0), pd.DataFrame(rows)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--out", default="results/baseline_bias.csv")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if device != "cuda":
        model.to(device)


    c_acc, c_tie, c_df = run_crows(model, tok, device, n=args.samples)
    s_acc, s_tie, s_df = run_stereoset(model, tok, device, n=args.samples)

    summary = pd.DataFrame([{
        "model": args.model, "seed": args.seed,
        "crows_anti_acc": round(c_acc,4), "crows_tie_rate": round(c_tie,4),
        "stereo_anti_acc": round(s_acc,4), "stereo_tie_rate": round(s_tie,4)
    }])
    summary.to_csv(args.out, index=False)
    c_df.to_csv("results/crows_detail.csv", index=False)
    s_df.to_csv("results/stereoset_detail.csv", index=False)
    print(summary)

if __name__ == "__main__":
    main()
