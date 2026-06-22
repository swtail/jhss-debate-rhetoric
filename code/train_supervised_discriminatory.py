#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Bryan Kyung and Katie Kim
"""
Supervised discriminatory classifier (the only method that validates) — reproduces Section 6.10.

Pipeline (numpy/pandas only; no sklearn, no GPU, no network):
  1. Train a TF-IDF + logistic-regression classifier on the 372-sentence human gold
     (data/human_gold_372.csv), and report 5-fold cross-validated F1 (the validation number).
  2. Apply the trained classifier to the full corpus (2004-2020 and 2024) and write
     per-debate discriminatory counts (data/supervised_counts.csv).
  3. Merge the counts into the poll panel (data/panel_2008_2024.csv) and re-estimate the
     source-specific model (WLS by sample size; election-year FE; debate-clustered SE),
     writing the H3 test in results/supervised_regression_results.txt.

Run:  python code/train_supervised_discriminatory.py
"""
import os, re, csv, math
import numpy as np
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D    = lambda *p: os.path.join(ROOT, "data", *p)
R    = lambda *p: os.path.join(ROOT, "results", *p)

# ---------- text features ----------
def tok(s):
    t = re.findall(r"[a-z']+", str(s).lower())
    return t + [t[i] + "_" + t[i+1] for i in range(len(t)-1)]
def build_vocab(docs, minc=2):
    c = Counter()
    for d in docs: c.update(set(tok(d)))
    return {w: i for i, w in enumerate([w for w, n in c.items() if n >= minc])}
def idf_of(docs, v):
    df = np.zeros(len(v))
    for d in docs:
        for w in set(tok(d)):
            if w in v: df[v[w]] += 1
    return np.log((1 + len(docs)) / (1 + df)) + 1
def feats(docs, v, idf):
    X = np.zeros((len(docs), len(v)))
    for r, d in enumerate(docs):
        for w in tok(d):
            if w in v: X[r, v[w]] += 1
    X = np.log1p(X) * idf
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return X / n
def train_lr(X, y, l2=1.0, it=600, lr=0.5):
    n, p = X.shape; w = np.zeros(p); b = 0.0
    cw = np.where(y == 1, n/(2*max(y.sum(),1)), n/(2*max((1-y).sum(),1)))  # balanced
    for _ in range(it):
        pr = 1/(1+np.exp(-(X@w+b))); g = (pr-y)*cw
        w -= lr*(X.T@g/n + l2*w/n); b -= lr*g.mean()
    return w, b

# ---------- cross-validated F1 (the validation number) ----------
def cv_f1(S, y, seeds=10, k=5):
    def folds(y, rng):
        i0, i1 = np.where(y==0)[0], np.where(y==1)[0]; rng.shuffle(i0); rng.shuffle(i1)
        F = [[] for _ in range(k)]
        for i, ix in enumerate(i1): F[i%k].append(ix)
        for i, ix in enumerate(i0): F[i%k].append(ix)
        return [np.array(f) for f in F]
    fs_all = []
    for seed in range(seeds):
        rng = np.random.RandomState(seed); fs = folds(y, rng); oof = np.zeros(len(y))
        for f in range(k):
            te = fs[f]; tr = np.concatenate([fs[j] for j in range(k) if j != f])
            v = build_vocab([S[i] for i in tr]); idf = idf_of([S[i] for i in tr], v)
            w, b = train_lr(feats([S[i] for i in tr], v, idf), y[tr])
            oof[te] = 1/(1+np.exp(-(feats([S[i] for i in te], v, idf)@w + b)))
        p = (oof >= 0.5).astype(int)
        tp = int(((p==1)&(y==1)).sum()); fp = int(((p==1)&(y==0)).sum()); fn = int(((p==0)&(y==1)).sum())
        P = tp/(tp+fp) if tp+fp else 0; Rec = tp/(tp+fn) if tp+fn else 0
        fs_all.append(2*P*Rec/(P+Rec) if P+Rec else 0)
    return float(np.mean(fs_all)), float(np.std(fs_all))

# ---------- cluster-robust WLS ----------
def betacf(a,b,x):
    FPMIN=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1.0;d=1-qab*x/qap
    d=1/(FPMIN if abs(d)<FPMIN else d);h=d
    for mm in range(1,300):
        m2=2*mm;aa=mm*(b-mm)*x/((qam+m2)*(a+m2))
        d=1+aa*d;d=FPMIN if abs(d)<FPMIN else d;c=1+aa/c;c=FPMIN if abs(c)<FPMIN else c
        d=1/d;h*=d*c;aa=-(a+mm)*(qab+mm)*x/((a+m2)*(qap+m2))
        d=1+aa*d;d=FPMIN if abs(d)<FPMIN else d;c=1+aa/c;c=FPMIN if abs(c)<FPMIN else c
        d=1/d;de=d*c;h*=de
        if abs(de-1)<3e-12: break
    return h
def ibeta(a,b,x):
    if x<=0: return 0.0
    if x>=1: return 1.0
    lb=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b); f=math.exp(math.log(x)*a+math.log(1-x)*b-lb)
    return f/a*betacf(a,b,x) if x<(a+1)/(a+b+2) else 1-f/b*betacf(b,a,1-x)

def main():
    g = list(csv.DictReader(open(D("human_gold_372.csv"), encoding="utf-8")))
    S = [r["sentence"] for r in g]; y = np.array([int(r["gold_lenient"]) for r in g])
    f1m, f1s = cv_f1(S, y)
    print(f"Supervised classifier cross-validated F1 = {f1m:.2f} ± {f1s:.2f} (n={len(y)}, positives={int(y.sum())})")

    # train on all labels, classify both corpora
    V = build_vocab(S); IDF = idf_of(S, V); w, b = train_lr(feats(S, V, IDF), y)
    counts = {}
    for path in [D("corpus_sentences.csv"), D("corpus_2024.csv")]:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        X = feats([str(r["sentence"]) for r in rows], V, IDF)
        pred = (1/(1+np.exp(-(X@w + b))) >= 0.5).astype(int)
        for r, p in zip(rows, pred):
            k = (r["date"], r["party"]); counts[k] = counts.get(k, 0) + int(p)
    with open(D("supervised_counts.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["date", "party", "supervised_discriminatory"])
        for (dt, pty), n in sorted(counts.items()): wr.writerow([dt, pty, n])

    # merge into panel + regress
    P = list(csv.DictReader(open(D("panel_2008_2024.csv"), encoding="utf-8")))
    def num(x):
        try: return float(x)
        except: return np.nan
    dd = np.array([counts.get((r["date"], "Dem"), np.nan) for r in P], float)
    dr = np.array([counts.get((r["date"], "Rep"), np.nan) for r in P], float)
    ctrl = ["pre_democrats","pre_republicans","Immigration","ForeignPolicy","AbortionRights"]
    C = np.array([[num(r[c]) for c in ctrl] for r in P])
    wts = np.array([num(r["sample1"]) for r in P]); years = sorted(set(r["year"] for r in P))
    yd = np.array([[1.0 if r["year"]==Y else 0.0 for Y in years[1:]] for r in P])
    clusters = [r["date"] for r in P]
    def reg(yv):
        X = np.column_stack([np.ones(len(yv)), dd, dr] + [C[:, j] for j in range(C.shape[1])] + [yd[:, j] for j in range(yd.shape[1])])
        names = ["const","Discriminatory_Dem","Discriminatory_Rep"] + ctrl + [f"yr_{Y}" for Y in years[1:]]
        m = ~np.isnan(yv) & ~np.isnan(X).any(1) & (wts > 0)
        X, yv, w_, cl = X[m], yv[m], wts[m]/wts[m].mean(), np.array(clusters)[m]
        XtW = X.T*w_; bread = np.linalg.inv(XtW@X); beta = bread@(XtW@yv); res = yv - X@beta
        meat = np.zeros((X.shape[1],)*2)
        for c in set(cl):
            i = cl==c; s = (X[i]*w_[i][:,None]).T@res[i]; meat += np.outer(s, s)
        G = len(set(cl)); n, kk = X.shape; adj = (G/(G-1))*((n-1)/(n-kk))
        V2 = bread@meat@bread*adj; se = np.sqrt(np.diag(V2)); t = beta/se; df = G-1
        p = [ibeta(df/2, 0.5, df/(df+tt*tt)) for tt in t]
        return names, beta, se, t, p, G, n
    out = ["=== Section 6.10: H3 on the validated supervised discriminatory measure ===",
           f"Supervised classifier cross-validated F1 = {f1m:.2f} +/- {f1s:.2f}",
           "Spec: WLS weighted by poll sample size; election-year FE; SE clustered by debate."]
    for dv, lab in [("change_democrats","DV = change in Democratic poll share"),
                    ("change_republicans","DV = change in Republican poll share")]:
        names, beta, se, t, p, G, n = reg(np.array([num(r[dv]) for r in P]))
        out.append(f"\n{lab}  (N={n} polls, clusters={G})")
        out.append(f"{'var':22s}{'coef':>9}{'se':>9}{'t':>7}{'p':>8}")
        for nm, b_, s_, t_, p_ in zip(names, beta, se, t, p):
            if nm.startswith("yr_") or nm == "const": continue
            star = "***" if p_<.01 else "**" if p_<.05 else "*" if p_<.1 else ""
            out.append(f"{nm:22s}{b_:9.3f}{s_:9.3f}{t_:7.2f}{p_:8.3f} {star}")
    txt = "\n".join(out); print("\n"+txt)
    open(R("supervised_regression_results.txt"), "w", encoding="utf-8").write(txt+"\n")
    print("\nWrote data/supervised_counts.csv and results/supervised_regression_results.txt")

if __name__ == "__main__":
    main()
