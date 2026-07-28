#!/bin/bash
# Example script to run ablation tests

# Run comprehensive ablation test
python eval/ablation_test.py \
    --test_all \
    --model google/gemma-2-2b \
    --scores results/sae_feature_scores.npy \
    --biased_txt data/biased_x5.txt \
    --neutral_txt data/neutral_x5.txt \
    --samples 400 \
    --out results/ablation_results.csv \
    --seed 1337

# Or run specific ablations:
# python eval/ablation_test.py --test_methods --samples 400
# python eval/ablation_test.py --test_hyperparams --samples 400
# python eval/ablation_test.py --test_selection --samples 400
# python eval/ablation_test.py --test_components --samples 400


