#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Bryan Kyung and Katie Kim
"""
GPU zero-shot rhetoric classifier (runs locally on your A5000; no API, no hand-coding).

Auto-selects the GPU whose name contains 'A5000', classifies every debate sentence into the
three constructs with a strong zero-shot NLI model, validates against the 60-sentence expert
gold set, and writes candidate-debate counts for the regressions.

SETUP (on your machine, once):
    pip install transformers torch pandas
RUN (single command):
    python run_gpu_classify.py --corpus ../../R1/Data_Files/corpus_sentences.csv \
        --gold ../../R1/Data_Files/gold_sample.json \
        --out zeroshot_labels.csv --counts zeroshot_counts.csv

Then send me zeroshot_counts.csv and I'll merge it and re-run every regression.
The first run downloads the model (~1.5 GB) from Hugging Face (needs internet).
On an A5000, ~24,546 sentences take only a few minutes.
"""
import argparse, json, torch, pandas as pd
from transformers import pipeline

# Descriptive hypothesis labels (zero-shot works better with rich labels than single words)
LABELS = {
    "aggressive":     "a confrontational personal attack accusing an opponent of failure, lying, or incompetence",
    "inflammatory":   "emotionally provocative, fear-mongering, or morally outraged language",
    "discriminatory": "rhetoric that targets, demeans, stereotypes, or blames a racial, ethnic, religious, national, or gender group, including group-blaming, threat framing, and coded identity-charged appeals",
}
# expert gold single-labels, order matches gold_sample.json
GOLD = list("NNNIAINNNI"+"NNNINNINNN"+"NNNNNNANNN"+"ANANNNNINI"+"NNNIAINNNN"+"NINNANNNNN")

def pick_a5000():
    if not torch.cuda.is_available():
        print("WARNING: no CUDA visible; running on CPU."); return -1
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for i, nm in enumerate(names):
        print(f"  GPU {i}: {nm}")
    for i, nm in enumerate(names):
        if "A5000" in nm.upper().replace(" ", ""):
            print(f"Selected GPU {i} ({nm})"); return i
    print(f"A5000 not found by name; using GPU 0 ({names[0]})"); return 0

def classify(clf, sentences, threshold, batch=64):
    cand = list(LABELS.values()); inv = {v: k for k, v in LABELS.items()}
    out = []
    for b in range(0, len(sentences), batch):
        chunk = sentences[b:b+batch]
        res = clf(chunk, cand, multi_label=True, hypothesis_template="This text is {}.")
        if isinstance(res, dict): res = [res]
        for r in res:
            d = {inv[lab]: (sc >= threshold) for lab, sc in zip(r["labels"], r["scores"])}
            out.append({k: int(d.get(k, 0)) for k in LABELS})
        print(f"  {min(b+batch,len(sentences))}/{len(sentences)}", end="\r")
    print()
    return out

def gold_validation(preds, gold):
    print("\n=== GOLD VALIDATION (per-category binary) ===")
    for cat, L in [("aggressive","A"),("inflammatory","I"),("discriminatory","D")]:
        tp=sum(p[cat]==1 and g==L for p,g in zip(preds,gold))
        fp=sum(p[cat]==1 and g!=L for p,g in zip(preds,gold))
        fn=sum(p[cat]==0 and g==L for p,g in zip(preds,gold))
        ng=sum(g==L for g in gold)
        P=tp/(tp+fp) if tp+fp else float('nan'); R=tp/(tp+fn) if tp+fn else float('nan')
        F=2*P*R/(P+R) if (P==P and R==R and P+R>0) else float('nan')
        print(f"  {cat:14s} n_gold={ng:2d}  P={P:.2f}  R={R:.2f}  F1={F:.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--model", default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
    ap.add_argument("--out", default="zeroshot_labels.csv")
    ap.add_argument("--counts", default="zeroshot_counts.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    dev = pick_a5000()
    clf = pipeline("zero-shot-classification", model=a.model, device=dev)
    # 1) validate on gold first
    gsent = [r["sentence"] for r in json.load(open(a.gold, encoding="utf-8"))]
    gold_validation(classify(clf, gsent, a.threshold), GOLD)
    # 2) full corpus
    df = pd.read_csv(a.corpus); df["date"] = df["date"].astype(str)
    preds = classify(clf, df["sentence"].astype(str).tolist(), a.threshold)
    for c in LABELS: df[c] = [p[c] for p in preds]
    df.to_csv(a.out, index=False)
    g = df.groupby(["date","party"])[list(LABELS)].sum().reset_index()
    g.to_csv(a.counts, index=False)
    print(f"\nWrote labels -> {a.out}; counts -> {a.counts}. Send me the counts file.")

if __name__ == "__main__":
    main()
