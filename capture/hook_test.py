import argparse, os, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--layer_index", type=int, default=-2, help="-2 = last block output")
    ap.add_argument("--outdir", default="cache/acts/l_lastminus1")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    prompts = [
        "The doctor said that the nurse was very",
        "The programmer fixed the bug and then",
        "A teacher prepared a lesson on",
        "The engineer presented the design to",
    ]

    for idx, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple: embeddings at 0, then blocks
        target = hs[args.layer_index].detach().float().cpu().numpy()
        np.save(os.path.join(args.outdir, f"act_{idx}.npy"), target)

    print(f"saved {len(prompts)} activations to {args.outdir} (layer_index={args.layer_index}, total_layers={len(out.hidden_states)-1})")

if __name__ == "__main__":
    main()
