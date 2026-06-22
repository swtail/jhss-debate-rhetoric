#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Bryan Kyung and Katie Kim
"""
Score every debate sentence with Google's Perspective API — the purpose-built,
peer-reviewed classifier for toxicity and IDENTITY_ATTACK (the right measure for the
'discriminatory' construct). Writes per-sentence scores and binary flags at a chosen
threshold, then aggregates to candidate-debate counts for the regressions.

SETUP
  pip install google-api-python-client pandas
  Get a free Perspective API key: https://developers.perspectiveapi.com/s/docs-get-started
  export PERSPECTIVE_API_KEY=...

RUN
  python perspective_classify.py --corpus ../../R1/Data_Files/corpus_sentences.csv \
        --out perspective_scores.csv --counts perspective_counts.csv --threshold 0.5

Notes: Perspective rate-limits to ~1 query/second by default; the script sleeps to
respect that and caches to perspective_cache.jsonl so reruns are free. ~24,546 calls
≈ 7 hours at 1 QPS (request a quota increase to go faster).
"""
import os, json, time, argparse, hashlib
import pandas as pd
from googleapiclient import discovery

ATTRS = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "THREAT"]

def client():
    return discovery.build("commentanalyzer", "v1alpha1",
        developerKey=os.environ["PERSPECTIVE_API_KEY"],
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False)

def score_one(svc, text):
    req = {"comment": {"text": text[:3000]},
           "languages": ["en"],
           "requestedAttributes": {a: {} for a in ATTRS}}
    r = svc.comments().analyze(body=req).execute()
    return {a: r["attributeScores"][a]["summaryScore"]["value"] for a in ATTRS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="perspective_scores.csv")
    ap.add_argument("--counts", default="perspective_counts.csv")
    ap.add_argument("--cache", default="perspective_cache.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    df = pd.read_csv(a.corpus)
    cache = {}
    if os.path.exists(a.cache):
        for line in open(a.cache, encoding="utf-8"):
            o = json.loads(line); cache[o["h"]] = o["s"]
    svc = client(); cf = open(a.cache, "a", encoding="utf-8")
    scores = []
    for n, s in enumerate(df["sentence"].astype(str)):
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        if h in cache: scores.append(cache[h]); continue
        for attempt in range(5):
            try:
                sc = score_one(svc, s); break
            except Exception as e:
                if attempt == 4: sc = {a_: float("nan") for a_ in ATTRS}
                else: time.sleep(2*(attempt+1))
        scores.append(sc); cf.write(json.dumps({"h": h, "s": sc})+"\n"); cf.flush()
        time.sleep(1.0)
        if n % 200 == 0: print(f"  {n}/{len(df)}", end="\r")
    cf.close()
    for at in ATTRS: df[at] = [sc.get(at) for sc in scores]
    # binary constructs from Perspective:
    df["inflammatory"] = (df["TOXICITY"]   >= a.threshold).astype(int)
    df["aggressive"]   = (df["INSULT"]     >= a.threshold).astype(int)
    df["discriminatory"] = (df["IDENTITY_ATTACK"] >= a.threshold).astype(int)
    df.to_csv(a.out, index=False)
    g = df.groupby(["date","party"])[["aggressive","inflammatory","discriminatory"]].sum().reset_index()
    g.to_csv(a.counts, index=False)
    print(f"\nWrote scores -> {a.out}; counts -> {a.counts}")
    print("IDENTITY_ATTACK is the validated measure for the discriminatory construct.")

if __name__ == "__main__":
    main()
