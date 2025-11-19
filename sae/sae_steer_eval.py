import math
import argparse, numpy as np, torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.bias_eval import run_crows, run_stereoset  # uses your working funcs (acc, tie, df)

# ----------------------------
# SAE selection + steering hook
# ----------------------------
def select_features(scores_path, topk=32, require_positive_gap=True):
    s = np.load(scores_path, allow_pickle=True).item()
    d, gap = s["d"], s["delta"]
    idx = np.argsort(-np.abs(d))
    sel=[]
    for i in idx:
        if require_positive_gap and gap[i] <= 0: 
            continue
        sel.append(int(i))
        if len(sel) >= topk: 
            break
    return sel, s["E"], s["D"]

# ----------------------------
# PPL evaluation helper
# ----------------------------
def compute_wikitext2_ppl(tok, model, device, max_len=4096, stride=2048, max_tokens=None):
    """
    Sliding-window perplexity on WikiText-2 (raw) test split.
    Avoids >context-length errors by chunking with overlap.
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_tokens if max_tokens is not None else max_len)
    input_ids = enc["input_ids"].to(device)
    if max_tokens is not None and input_ids.size(1) > max_tokens:
        input_ids = input_ids[:, :max_tokens]

    nlls, total_tokens = [], 0
    for start in range(0, input_ids.size(1) - 1, stride):
        end = min(start + max_len, input_ids.size(1))
        trg_len = end - start - 1
        if trg_len <= 0:
            break
        chunk = input_ids[:, start:end]
        labels = chunk.clone()
        with T.no_grad():
            out = model(chunk, labels=labels)
            nlls.append(out.loss * trg_len)
        total_tokens += trg_len
        if end == input_ids.size(1):
            break

    mean_nll = T.stack(nlls).sum() / total_tokens
    return float(math.exp(mean_nll))

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--scores", default="results/sae_feature_scores.npy")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--samples", type=int, default=400,
                    help="Number of examples for CrowS/StereoSet when mode=bias. Ignored for mode=ppl.")
    ap.add_argument("--mode", choices=["bias", "ppl"], default="bias",
                    help="bias = CrowS/StereoSet (default), ppl = WikiText-2 perplexity")
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="Maximum tokens for WikiText-2 perplexity (default: full dataset)")
    ap.add_argument("--layer_index", type=int, default=-2,
                    help="Layer index where SAE was trained (default: -2 = second-to-last)")
    args = ap.parse_args()

    # device / dtype
    dev = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if dev in ["cuda","mps"] else T.float32

    # model + tok
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(dev)
    model.eval()

    # SAE features + hook
    feat_ids, E, D = select_features(args.scores, topk=args.topk)
    
    # Hook into the correct layer (where SAE was trained), not lm_head
    # The SAE was trained on hidden_states[layer_index], so we need to modify that layer's output
    # Create the SAE transformation function
    E_t = T.tensor(E, dtype=T.float32, device=dev)   # [F,H]
    D_t = T.tensor(D, dtype=T.float32, device=dev)   # [H,F]
    # Use encoder directions instead of decoder for steering
    Esel = E_t[feat_ids, :]                          # [K,H] - encoder directions for selected features
    
    def apply_sae_transform(h):
        """Apply SAE transformation to hidden states h [B,S,H]"""
        # SAE encoder: ReLU(h @ E^T)
        z = T.relu(T.einsum("bsh,fh->bsf", h, E_t))  # [B,S,F]
        z_sel = z[..., feat_ids]                      # [B,S,K] - activations of selected features
        # gate selected features above tau
        mask = (z_sel > args.tau).to(h.dtype)  # [B,S,K]
        # subtract alpha * (activation * encoder_direction) for active feats
        # This removes the component of h that aligns with bias feature directions
        contrib = T.einsum("bsk,kh->bsh", mask * z_sel * (-args.alpha), Esel)
        return h + contrib
    
    # Find the transformer layer to hook into
    # Most models have: model.model.layers (Phi-3, Gemma, etc.) or model.transformer.h (GPT-2 style)
    handle = None
    hook_applied = False
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Phi-3, Gemma, Llama style
        layers = model.model.layers
        target_layer_idx = args.layer_index if args.layer_index >= 0 else len(layers) + args.layer_index
        if 0 <= target_layer_idx < len(layers):
            # Use pre_hook on the NEXT layer (target_layer_idx + 1) to modify its input
            # This ensures our modification flows through the remaining layers
            if target_layer_idx + 1 < len(layers):
                next_layer_idx = target_layer_idx + 1
                def layer_pre_hook(module, input_tuple):
                    # input_tuple is (hidden_states, ...) from previous layer
                    if not isinstance(input_tuple, tuple):
                        input_tuple = (input_tuple,)
                    h = input_tuple[0]  # [B,S,H] - this is layer -2's output
                    modified_h = apply_sae_transform(h)
                    return (modified_h,) + input_tuple[1:] if len(input_tuple) > 1 else (modified_h,)
                handle = layers[next_layer_idx].register_forward_pre_hook(layer_pre_hook)
            else:
                # If we're at the last layer, use forward hook on current layer
                def layer_hook(module, input_tuple, output):
                    if isinstance(output, tuple):
                        h = output[0]  # [B,S,H]
                        modified_h = apply_sae_transform(h)
                        return (modified_h,) + output[1:]
                    else:
                        return apply_sae_transform(output)
                handle = layers[target_layer_idx].register_forward_hook(layer_hook)
            hook_applied = True
        else:
            raise ValueError(f"Layer index {args.layer_index} out of range for {len(layers)} layers")
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style
        layers = model.transformer.h
        target_layer_idx = args.layer_index if args.layer_index >= 0 else len(layers) + args.layer_index
        if 0 <= target_layer_idx < len(layers):
            # Use pre_hook on the NEXT layer (target_layer_idx + 1) to modify its input
            if target_layer_idx + 1 < len(layers):
                next_layer_idx = target_layer_idx + 1
                def layer_pre_hook(module, input_tuple):
                    if not isinstance(input_tuple, tuple):
                        input_tuple = (input_tuple,)
                    h = input_tuple[0]  # [B,S,H] - this is layer -2's output
                    modified_h = apply_sae_transform(h)
                    return (modified_h,) + input_tuple[1:] if len(input_tuple) > 1 else (modified_h,)
                handle = layers[next_layer_idx].register_forward_pre_hook(layer_pre_hook)
            else:
                # If we're at the last layer, use forward hook on current layer
                def layer_hook(module, input_tuple, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        modified_h = apply_sae_transform(h)
                        return (modified_h,) + output[1:]
                    else:
                        return apply_sae_transform(output)
                handle = layers[target_layer_idx].register_forward_hook(layer_hook)
            hook_applied = True
        else:
            raise ValueError(f"Layer index {args.layer_index} out of range for {len(layers)} layers")
    else:
        # Fallback: hook into lm_head (original behavior, but warns about mismatch)
        import warnings
        warnings.warn(
            f"Could not find transformer layers. Hooking into lm_head, but SAE was trained on layer {args.layer_index}. "
            f"This is a LAYER MISMATCH and may cause incorrect behavior. "
            f"Model structure: {[n for n, _ in model.named_children()]}"
        )
        # For lm_head, we need to use pre_hook format
        def pre_hook_lm_head(_mod, inputs):
            h = inputs[0]
            return (apply_sae_transform(h),)
        handle = model.lm_head.register_forward_pre_hook(pre_hook_lm_head)
        hook_applied = True
    
    if not hook_applied:
        raise RuntimeError("Failed to apply hook to any layer!")

    try:
        if args.mode == "ppl":
            ppl = compute_wikitext2_ppl(tok, model, dev, max_len=4096, stride=2048, max_tokens=args.max_tokens)
            print({
                "alpha": args.alpha, "tau": args.tau, "topk": args.topk,
                "ppl_wikitext2": round(ppl, 3)
            })
            return

        # mode == "bias": CrowS + StereoSet using your working helpers
        c_acc, _, _ = run_crows(model, tok, dev, n=args.samples)
        s_acc, _, _ = run_stereoset(model, tok, dev, n=args.samples)
        print({
            "alpha": args.alpha, "tau": args.tau, "topk": args.topk,
            "crows_anti_acc": round(c_acc, 4),
            "stereo_anti_acc": round(s_acc, 4)
        })

    finally:
        handle.remove()

if __name__ == "__main__":
    main()
