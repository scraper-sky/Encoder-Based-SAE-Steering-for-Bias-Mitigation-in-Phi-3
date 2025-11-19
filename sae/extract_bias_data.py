"""
Extract biased and neutral sentences from CrowS and StereoSet datasets
for use in SAE feature ranking.
"""
import argparse
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crows_n", type=int, default=1000, help="Number of CrowS examples")
    ap.add_argument("--stereoset_n", type=int, default=1000, help="Number of StereoSet examples")
    ap.add_argument("--biased_out", default="data/biased_crows_stereoset.txt")
    ap.add_argument("--neutral_out", default="data/neutral_crows_stereoset.txt")
    args = ap.parse_args()

    biased_sents = []
    neutral_sents = []

    # Extract from CrowS
    print("Loading CrowS...")
    crows_ds = load_dataset("crows_pairs", trust_remote_code=True)["test"]
    n_crows = min(args.crows_n, len(crows_ds))
    for ex in crows_ds.select(range(n_crows)):
        # sent_more = stereotype, sent_less = anti-stereotype
        biased_sents.append(ex["sent_more"].strip())
        neutral_sents.append(ex["sent_less"].strip())

    # Extract from StereoSet
    print("Loading StereoSet...")
    stereoset_ds = load_dataset("McGill-NLP/stereoset", "intrasentence", trust_remote_code=True)["validation"]
    n_stereo = min(args.stereoset_n, len(stereoset_ds))
    
    LABEL_MAP = {0: "stereotype", 1: "anti-stereotype", 2: "unrelated"}
    count = 0
    for ex in stereoset_ds:
        if count >= n_stereo:
            break
        
        S = ex.get("sentences")
        lab2sent = {}
        
        if isinstance(S, list):
            for d in S:
                if isinstance(d, dict):
                    label = d.get("gold_label", d.get("label"))
                    if isinstance(label, int):
                        label = LABEL_MAP.get(label)
                    if label in ("stereotype", "anti-stereotype", "unrelated"):
                        lab2sent[label] = d.get("sentence")
        elif isinstance(S, dict):
            sent_list = S.get("sentence", [])
            label_list = S.get("gold_label", S.get("label", []))
            for sent, lab in zip(sent_list, label_list):
                label = LABEL_MAP.get(lab, lab)
                if label in ("stereotype", "anti-stereotype", "unrelated"):
                    lab2sent[label] = sent
        
        if all(k in lab2sent for k in ("stereotype", "anti-stereotype")):
            biased_sents.append(lab2sent["stereotype"].strip())
            neutral_sents.append(lab2sent["anti-stereotype"].strip())
            count += 1

    # Write to files
    print(f"Writing {len(biased_sents)} biased sentences to {args.biased_out}")
    with open(args.biased_out, "w") as f:
        for s in biased_sents:
            f.write(s + "\n")
    
    print(f"Writing {len(neutral_sents)} neutral sentences to {args.neutral_out}")
    with open(args.neutral_out, "w") as f:
        for s in neutral_sents:
            f.write(s + "\n")
    
    print(f"Done! Extracted {len(biased_sents)} sentence pairs")

if __name__ == "__main__":
    main()

