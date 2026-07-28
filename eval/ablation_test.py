"""
Comprehensive ablation test for SAE-based bias steering.

This script systematically tests different components and hyperparameters of the SAE-based
bias steering method to understand what contributes to its effectiveness.

Ablation Tests:
1. Steering method: SAE encoder, SAE decoder, direct activation, random control, baseline
2. Feature selection: Different topk, selection criteria (Cohen's d, delta, random)
3. Hyperparameters: Alpha, tau combinations
4. Component ablation: With/without thresholding, different topk values

Usage Examples:
    # Run all ablation tests
    python eval/ablation_test.py --test_all
    
    # Run only method comparison
    python eval/ablation_test.py --test_methods
    
    # Run with perplexity evaluation
    python eval/ablation_test.py --test_all --eval_ppl
    
    # Run specific ablation (e.g., hyperparameter sensitivity)
    python eval/ablation_test.py --test_hyperparams --samples 800
    
    # Custom model and data paths
    python eval/ablation_test.py --test_all --model google/gemma-2-2b \\
        --scores results/sae_feature_scores.npy \\
        --biased_txt data/biased_x5.txt --neutral_txt data/neutral_x5.txt
"""

import argparse
import math
import random
import numpy as np
import pandas as pd
import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from eval.bias_eval import run_crows, run_stereoset
from sae.sae_steer_eval import compute_wikitext2_ppl


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    T.manual_seed(s)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(s)


def load_sae_data(scores_path):
    """Load SAE encoder/decoder and feature scores."""
    data = np.load(scores_path, allow_pickle=True).item()
    E = data["E"]  # [F, H] encoder
    D = data["D"]  # [H, F] decoder
    d = data["d"]  # Cohen's d
    delta = data["delta"]  # mean difference
    return E, D, d, delta


def select_features_by_criterion(d, delta, topk=32, criterion="cohens_d", require_positive_gap=True):
    """
    Select features using different criteria.
    
    Args:
        d: Cohen's d scores [F]
        delta: Mean difference scores [F]
        topk: Number of features to select
        criterion: "cohens_d", "delta", "random", "all"
        require_positive_gap: Only select features with positive delta
    
    Returns:
        List of feature indices
    """
    if criterion == "random":
        # Random selection (control)
        all_indices = np.arange(len(d))
        if require_positive_gap:
            positive_indices = np.where(delta > 0)[0]
            if len(positive_indices) == 0:
                return []
            return np.random.choice(positive_indices, size=min(topk, len(positive_indices)), replace=False).tolist()
        return np.random.choice(all_indices, size=min(topk, len(all_indices)), replace=False).tolist()
    
    elif criterion == "all":
        # Use all features (no selection)
        all_indices = np.arange(len(d))
        if require_positive_gap:
            return np.where(delta > 0)[0].tolist()
        return all_indices.tolist()
    
    elif criterion == "cohens_d":
        # Select by absolute Cohen's d
        scores = np.abs(d)
        idx = np.argsort(-scores)
    elif criterion == "delta":
        # Select by absolute delta
        scores = np.abs(delta)
        idx = np.argsort(-scores)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    selected = []
    for i in idx:
        if require_positive_gap and delta[i] <= 0:
            continue
        selected.append(int(i))
        if len(selected) >= topk:
            break
    
    return selected


def apply_sae_encoder_steering(h, E_t, feat_ids, alpha, tau):
    """Apply SAE encoder direction steering."""
    # E_t is always float32; h may be float16 (CUDA/MPS autocast-free fp16 models).
    # einsum requires matching dtypes (strictly enforced on CUDA, silently
    # tolerated on MPS), so do the math in float32 and cast back at the end.
    orig_dtype = h.dtype
    h32 = h.float()

    # Compute activations
    z = T.relu(T.einsum("bsh,fh->bsf", h32, E_t))  # [B,S,F]
    z_sel = z[..., feat_ids]  # [B,S,K]

    # Gate by threshold
    if tau > 0:
        mask = (z_sel > tau).float()  # [B,S,K]
    else:
        mask = (z_sel > 0).float()  # Always active if above 0

    # Get encoder directions for selected features
    Esel = E_t[feat_ids, :]  # [K,H]

    # Subtract alpha * (activation * encoder_direction) for active features
    contrib = T.einsum("bsk,kh->bsh", mask * z_sel * (-alpha), Esel)
    return (h32 + contrib).to(orig_dtype)


def apply_sae_decoder_steering(h, E_t, D_t, feat_ids, alpha, tau):
    """Apply SAE decoder direction steering (alternative method).

    Uses decoder columns D[:, k] instead of encoder rows E[k, :] for steering.
    Still needs E to compute activations z.
    """
    orig_dtype = h.dtype
    h32 = h.float()

    # Compute activations using encoder
    z = T.relu(T.einsum("bsh,fh->bsf", h32, E_t))  # [B,S,F]
    z_sel = z[..., feat_ids]  # [B,S,K]

    if tau > 0:
        mask = (z_sel > tau).float()
    else:
        mask = (z_sel > 0).float()

    # Use decoder columns for steering (instead of encoder rows)
    Dsel = D_t[:, feat_ids]  # [H,K] - decoder columns for selected features
    # Subtract alpha * (activation * decoder_column) for active features
    contrib = T.einsum("bsk,hk->bsh", mask * z_sel * (-alpha), Dsel)
    return (h32 + contrib).to(orig_dtype)


def apply_direct_activation_steering(h, vec, alpha):
    """Apply direct activation steering (mean-difference vector)."""
    return h + alpha * vec.to(device=h.device, dtype=h.dtype)


def apply_random_steering(h, vec_shape, alpha, device):
    """Apply random direction steering (control)."""
    random_vec = T.randn(vec_shape, device=device, dtype=h.dtype)
    random_vec = random_vec / (random_vec.norm() + 1e-8)  # Normalize
    return h + alpha * random_vec


def register_steering_hook(model, steering_fn, steering_args, layer_index=-2):
    """Register a steering hook at the specified layer."""
    handle = None
    hook_applied = False
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        target_layer_idx = layer_index if layer_index >= 0 else len(layers) + layer_index
        
        if 0 <= target_layer_idx < len(layers):
            if target_layer_idx + 1 < len(layers):
                next_layer_idx = target_layer_idx + 1
                def layer_pre_hook(module, input_tuple):
                    if not isinstance(input_tuple, tuple):
                        input_tuple = (input_tuple,)
                    h = input_tuple[0]
                    modified_h = steering_fn(h, *steering_args)
                    return (modified_h,) + input_tuple[1:] if len(input_tuple) > 1 else (modified_h,)
                handle = layers[next_layer_idx].register_forward_pre_hook(layer_pre_hook)
            else:
                def layer_hook(module, input_tuple, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        modified_h = steering_fn(h, *steering_args)
                        return (modified_h,) + output[1:]
                    else:
                        return steering_fn(output, *steering_args)
                handle = layers[target_layer_idx].register_forward_hook(layer_hook)
            hook_applied = True
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        target_layer_idx = layer_index if layer_index >= 0 else len(layers) + layer_index
        
        if 0 <= target_layer_idx < len(layers):
            if target_layer_idx + 1 < len(layers):
                next_layer_idx = target_layer_idx + 1
                def layer_pre_hook(module, input_tuple):
                    if not isinstance(input_tuple, tuple):
                        input_tuple = (input_tuple,)
                    h = input_tuple[0]
                    modified_h = steering_fn(h, *steering_args)
                    return (modified_h,) + input_tuple[1:] if len(input_tuple) > 1 else (modified_h,)
                handle = layers[next_layer_idx].register_forward_pre_hook(layer_pre_hook)
            else:
                def layer_hook(module, input_tuple, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        modified_h = steering_fn(h, *steering_args)
                        return (modified_h,) + output[1:]
                    else:
                        return steering_fn(output, *steering_args)
                handle = layers[target_layer_idx].register_forward_hook(layer_hook)
            hook_applied = True
    
    if not hook_applied:
        raise RuntimeError(f"Failed to apply hook to layer {layer_index}")
    
    return handle


def build_direct_steering_vector(model, tok, device, biased_txt, neutral_txt, layer_index=-2):
    """Build mean-difference vector for direct activation steering."""
    from eval.steer_sweep import mean_rep, build_vector
    return build_vector(model, tok, device, biased_txt, neutral_txt, layer_index)


def run_ablation_experiment(
    model, tok, device, 
    method,  # "sae_encoder", "sae_decoder", "direct", "random", "baseline"
    E=None, D=None, feat_ids=None, alpha=0.0, tau=0.05,
    direct_vec=None, layer_index=-2,
    samples=400, eval_ppl=False, max_tokens=20000
):
    """Run a single ablation experiment."""
    handle = None
    
    try:
        if method == "baseline":
            # No steering
            pass
        elif method == "sae_encoder":
            if E is None or feat_ids is None:
                raise ValueError("SAE encoder method requires E and feat_ids")
            E_t = T.tensor(E, dtype=T.float32, device=device)
            steering_fn = lambda h: apply_sae_encoder_steering(h, E_t, feat_ids, alpha, tau)
            handle = register_steering_hook(model, steering_fn, [], layer_index)
        elif method == "sae_decoder":
            if E is None or D is None or feat_ids is None:
                raise ValueError("SAE decoder method requires E, D and feat_ids")
            E_t = T.tensor(E, dtype=T.float32, device=device)
            D_t = T.tensor(D, dtype=T.float32, device=device)
            steering_fn = lambda h: apply_sae_decoder_steering(h, E_t, D_t, feat_ids, alpha, tau)
            handle = register_steering_hook(model, steering_fn, [], layer_index)
        elif method == "direct":
            if direct_vec is None:
                raise ValueError("Direct method requires direct_vec")
            vec = direct_vec.to(device)
            steering_fn = lambda h: apply_direct_activation_steering(h, vec, alpha)
            handle = register_steering_hook(model, steering_fn, [], layer_index)
        elif method == "random":
            # Random direction control
            h_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2048
            steering_fn = lambda h: apply_random_steering(h, (h_dim,), alpha, device)
            handle = register_steering_hook(model, steering_fn, [], layer_index)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Run evaluations
        results = {}
        
        if eval_ppl:
            ppl = compute_wikitext2_ppl(tok, model, device, max_len=4096, stride=2048, max_tokens=max_tokens)
            results["ppl_wikitext2"] = round(ppl, 3)
        else:
            c_acc, _, _ = run_crows(model, tok, device, n=samples)
            s_acc, _, _ = run_stereoset(model, tok, device, n=samples)
            results["crows_anti_acc"] = round(c_acc, 4)
            results["stereo_anti_acc"] = round(s_acc, 4)
        
        return results
        
    finally:
        if handle is not None:
            handle.remove()


def main():
    ap = argparse.ArgumentParser(description="Comprehensive ablation test for SAE bias steering")
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--scores", default="results/sae_feature_scores.npy")
    ap.add_argument("--biased_txt", default="data/biased_x5.txt")
    ap.add_argument("--neutral_txt", default="data/neutral_x5.txt")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--samples", type=int, default=400, help="Samples for bias eval")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="results/ablation_results.csv")
    
    # Ablation parameters
    ap.add_argument("--test_methods", action="store_true", help="Test different steering methods")
    ap.add_argument("--test_selection", action="store_true", help="Test different feature selection criteria")
    ap.add_argument("--test_hyperparams", action="store_true", help="Test hyperparameter sensitivity")
    ap.add_argument("--test_components", action="store_true", help="Test component ablation (tau, topk)")
    ap.add_argument("--test_all", action="store_true", help="Run all ablation tests")
    ap.add_argument("--eval_ppl", action="store_true", help="Also evaluate perplexity")
    ap.add_argument("--max_tokens", type=int, default=20000, help="Max tokens for PPL eval")
    
    args = ap.parse_args()
    
    if not any([args.test_methods, args.test_selection, args.test_hyperparams, args.test_components, args.test_all]):
        # Default: run all if nothing specified
        args.test_all = True
    
    set_seed(args.seed)
    
    # Setup
    device = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    dtype = T.float16 if device in ["cuda", "mps"] else T.float32
    
    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()
    
    # Load SAE data
    print(f"Loading SAE data: {args.scores}")
    E, D, d, delta = load_sae_data(args.scores)
    
    # Build direct steering vector
    print("Building direct steering vector...")
    direct_vec = build_direct_steering_vector(model, tok, device, args.biased_txt, args.neutral_txt, args.layer_index)
    
    results = []
    
    # 1. Method ablation: Compare steering methods
    if args.test_all or args.test_methods:
        print("\n=== Testing Steering Methods ===")
        methods = ["baseline", "sae_encoder", "sae_decoder", "direct", "random"]
        default_feat_ids = select_features_by_criterion(d, delta, topk=32, criterion="cohens_d")
        
        for method in tqdm(methods, desc="Methods"):
            try:
                # For decoder method, need both E and D
                needs_e = method in ["sae_encoder", "sae_decoder"]
                needs_d = method == "sae_decoder"
                res = run_ablation_experiment(
                    model, tok, device, method,
                    E=E if needs_e else None,
                    D=D if needs_d else None,
                    feat_ids=default_feat_ids if method in ["sae_encoder", "sae_decoder"] else None,
                    alpha=2.0 if method != "baseline" else 0.0,
                    tau=0.05,
                    direct_vec=direct_vec if method == "direct" else None,
                    layer_index=args.layer_index,
                    samples=args.samples,
                    eval_ppl=args.eval_ppl,
                    max_tokens=args.max_tokens
                )
                res.update({
                    "ablation_type": "method",
                    "method": method,
                    "alpha": 2.0 if method != "baseline" else 0.0,
                    "tau": 0.05,
                    "topk": 32 if method in ["sae_encoder", "sae_decoder"] else None,
                    "selection_criterion": "cohens_d" if method in ["sae_encoder", "sae_decoder"] else None,
                })
                results.append(res)
            except Exception as e:
                print(f"Error with method {method}: {e}")
    
    # 2. Selection criterion ablation
    if args.test_all or args.test_selection:
        print("\n=== Testing Feature Selection Criteria ===")
        criteria = ["cohens_d", "delta", "random"]
        for criterion in tqdm(criteria, desc="Selection criteria"):
            try:
                feat_ids = select_features_by_criterion(d, delta, topk=32, criterion=criterion)
                if len(feat_ids) == 0:
                    print(f"Skipping {criterion}: no features selected")
                    continue
                
                res = run_ablation_experiment(
                    model, tok, device, "sae_encoder",
                    E=E, feat_ids=feat_ids,
                    alpha=2.0, tau=0.05,
                    layer_index=args.layer_index,
                    samples=args.samples,
                    eval_ppl=args.eval_ppl,
                    max_tokens=args.max_tokens
                )
                res.update({
                    "ablation_type": "selection",
                    "method": "sae_encoder",
                    "alpha": 2.0,
                    "tau": 0.05,
                    "topk": 32,
                    "selection_criterion": criterion,
                })
                results.append(res)
            except Exception as e:
                print(f"Error with criterion {criterion}: {e}")
    
    # 3. Hyperparameter sensitivity
    if args.test_all or args.test_hyperparams:
        print("\n=== Testing Hyperparameter Sensitivity ===")
        alphas = [0.5, 1.0, 1.5, 2.0, 2.5]
        taus = [0.0, 0.02, 0.05, 0.08, 0.1]
        default_feat_ids = select_features_by_criterion(d, delta, topk=32, criterion="cohens_d")
        
        for alpha in tqdm(alphas, desc="Alpha"):
            for tau in taus:
                try:
                    res = run_ablation_experiment(
                        model, tok, device, "sae_encoder",
                        E=E, feat_ids=default_feat_ids,
                        alpha=alpha, tau=tau,
                        layer_index=args.layer_index,
                        samples=args.samples,
                        eval_ppl=args.eval_ppl,
                        max_tokens=args.max_tokens
                    )
                    res.update({
                        "ablation_type": "hyperparams",
                        "method": "sae_encoder",
                        "alpha": alpha,
                        "tau": tau,
                        "topk": 32,
                        "selection_criterion": "cohens_d",
                    })
                    results.append(res)
                except Exception as e:
                    print(f"Error with alpha={alpha}, tau={tau}: {e}")
    
    # 4. Component ablation
    if args.test_all or args.test_components:
        print("\n=== Testing Component Ablation ===")
        topks = [8, 16, 32, 64, 128]
        taus = [0.0, 0.05]  # With and without thresholding
        
        for topk in tqdm(topks, desc="TopK"):
            for tau in taus:
                try:
                    feat_ids = select_features_by_criterion(d, delta, topk=topk, criterion="cohens_d")
                    if len(feat_ids) == 0:
                        continue
                    
                    res = run_ablation_experiment(
                        model, tok, device, "sae_encoder",
                        E=E, feat_ids=feat_ids,
                        alpha=2.0, tau=tau,
                        layer_index=args.layer_index,
                        samples=args.samples,
                        eval_ppl=args.eval_ppl,
                        max_tokens=args.max_tokens
                    )
                    res.update({
                        "ablation_type": "components",
                        "method": "sae_encoder",
                        "alpha": 2.0,
                        "tau": tau,
                        "topk": topk,
                        "selection_criterion": "cohens_d",
                    })
                    results.append(res)
                except Exception as e:
                    print(f"Error with topk={topk}, tau={tau}: {e}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n=== Results Summary ===")
        print(f"Total experiments: {len(results)}")
        print(f"\nSaved to: {args.out}")
        print("\nFirst few results:")
        print(df.head(10).to_string(index=False))
    else:
        print("No results collected!")


if __name__ == "__main__":
    main()

