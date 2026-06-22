#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Bryan Kyung and Katie Kim
"""
Classify debate sentences for hateful / group-directed content with a pretrained
transformer (HateBERT-family). Strong, purpose-built signal for the 'discriminatory'
construct. Needs network to download the model (and a GPU is much faster).

SETUP
  pip install transformers torch pandas
RUN
  python hatebert_classify.py --corpus ../../R1/Data_Files/corpus_sentences.csv \
        --out hatebert_labels.csv --counts hatebert_counts.csv

Default model: 'Hate-speech-CNERG/dehatebert-mono-english' (binary hate / non-hate).
Swap --model to 'GroNLP/hateBERT' (+ a fine-tuned head) or another validated checkpoint
if your lab prefers. Always re-validate the chosen model on the 60-sentence gold set
before trusting the counts.
"""
import argparse, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="Hate-speech-CNERG/dehatebert-mono-english")
    ap.add_argument("--out", default="hatebert_labels.csv")
    ap.add_argument("--counts", default="hatebert_counts.csv")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(a.model).to(dev).eval()
    df = pd.read_csv(a.corpus); sents = df["sentence"].astype(str).tolist()
    probs = []
    with torch.no_grad():
        for b in range(0, len(sents), a.batch):
            enc = tok(sents[b:b+a.batch], return_tensors="pt", truncation=True,
                      padding=True, max_length=128).to(dev)
            p = torch.softmax(mdl(**enc).logits, dim=-1)[:, -1]  # P(hate)
            probs.extend(p.cpu().tolist())
            print(f"  {min(b+a.batch,len(sents))}/{len(sents)}", end="\r")
    df["hate_prob"] = probs
    df["discriminatory"] = (df["hate_prob"] >= a.threshold).astype(int)
    df.to_csv(a.out, index=False)
    g = df.groupby(["date","party"])["discriminatory"].sum().reset_index()
    g.to_csv(a.counts, index=False)
    print(f"\nWrote labels -> {a.out}; counts -> {a.counts}")
    print("Validate on gold_sample.json before use; report per-category precision/recall/F1.")

if __name__ == "__main__":
    main()
