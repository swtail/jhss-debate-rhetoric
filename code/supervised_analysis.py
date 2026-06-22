#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Bryan Kyung and Katie Kim
"""
End-to-end reproduction of the supervised-measure analysis (the paper's primary measure).

Steps (numpy/pandas only; no GPU, no API, no internet):
  1. Train a supervised TF-IDF + logistic-regression classifier for EACH of the three
     rhetorical categories on the human-coded 372-sentence gold standards
     (aggressive/inflammatory: data/human_gold_aggr_infl_372.csv;
      discriminatory:        data/human_gold_372.csv).
  2. Report 5-fold cross-validated F1 per category (Table 11, supervised row).
  3. Classify the full 2008-2024 corpus (data/corpus_sentences.csv >= 2008 and
     data/corpus_2024.csv) by speaker party and aggregate to candidate-debate counts.
  4. Merge the counts into the poll panel (data/panel_2008_2024.csv) and estimate the
     baseline, source-specific, and sample-size-heterogeneity models with WLS (weighted by
     poll sample size), election-year fixed effects, and standard errors clustered by debate.

Run:  pip install numpy pandas
      python code/supervised_analysis.py
"""
import os, re, csv, math
import numpy as np, pandas as pd
from collections import Counter

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
D=lambda *p: os.path.join(ROOT,"data",*p)

# ---------- text features + logistic regression ----------
def tok(s):
    t=re.findall(r"[a-z']+",str(s).lower()); return t+[t[i]+'_'+t[i+1] for i in range(len(t)-1)]
def build_vocab(docs,minc=2):
    c=Counter()
    for d in docs: c.update(set(tok(d)))
    return {w:i for i,w in enumerate([w for w,n in c.items() if n>=minc])}
def idf(docs,v):
    df=np.zeros(len(v))
    for d in docs:
        for w in set(tok(d)):
            if w in v: df[v[w]]+=1
    return np.log((1+len(docs))/(1+df))+1
def feats(docs,v,id_):
    X=np.zeros((len(docs),len(v)))
    for r,d in enumerate(docs):
        for w in tok(d):
            if w in v: X[r,v[w]]+=1
    X=np.log1p(X)*id_; n=np.linalg.norm(X,axis=1,keepdims=True); n[n==0]=1; return X/n
def lr(X,y,l2=1.0,it=600,step=0.5):
    n,p=X.shape; w=np.zeros(p); b=0.0
    cw=np.where(y==1,n/(2*max(y.sum(),1)),n/(2*max((1-y).sum(),1)))
    for _ in range(it):
        pr=1/(1+np.exp(-(X@w+b))); g=(pr-y)*cw; w-=step*(X.T@g/n+l2*w/n); b-=step*g.mean()
    return w,b

def cv_f1(S,y,seeds=10,k=5):
    def folds(y,rng):
        i0,i1=np.where(y==0)[0],np.where(y==1)[0]; rng.shuffle(i0); rng.shuffle(i1)
        F=[[] for _ in range(k)]
        for i,ix in enumerate(i1):F[i%k].append(ix)
        for i,ix in enumerate(i0):F[i%k].append(ix)
        return [np.array(f) for f in F]
    out=[]
    for s in range(seeds):
        rng=np.random.RandomState(s); fs=folds(y,rng); oof=np.zeros(len(y))
        for f in range(k):
            te=fs[f]; tr=np.concatenate([fs[j] for j in range(k) if j!=f])
            v=build_vocab([S[i] for i in tr]); id_=idf([S[i] for i in tr],v)
            w,b=lr(feats([S[i] for i in tr],v,id_),y[tr])
            oof[te]=1/(1+np.exp(-(feats([S[i] for i in te],v,id_)@w+b)))
        p=(oof>=0.5).astype(int)
        tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
        P=tp/(tp+fp) if tp+fp else 0;R=tp/(tp+fn) if tp+fn else 0
        out.append(2*P*R/(P+R) if P+R else 0)
    return float(np.mean(out)),float(np.std(out))

# ---------- Student-t two-sided p-value (self-contained) ----------
def _betacf(a,b,x):
    F=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1.0;d=1-qab*x/qap;d=1/(F if abs(d)<F else d);h=d
    for m in range(1,300):
        m2=2*m;aa=m*(b-m)*x/((qam+m2)*(a+m2));d=1+aa*d;d=F if abs(d)<F else d;c=1+aa/c;c=F if abs(c)<F else c;d=1/d;h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));d=1+aa*d;d=F if abs(d)<F else d;c=1+aa/c;c=F if abs(c)<F else c;d=1/d;de=d*c;h*=de
        if abs(de-1)<3e-12:break
    return h
def tp_two(t,df):
    x=df/(df+t*t)
    def ib(a,b,x):
        if x<=0:return 0.0
        if x>=1:return 1.0
        lb=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b);f=math.exp(math.log(x)*a+math.log(1-x)*b-lb)
        return f/a*_betacf(a,b,x) if x<(a+1)/(a+b+2) else 1-f/b*_betacf(b,a,1-x)
    return ib(df/2,0.5,x)

def main():
    # ---- 1. train + cross-validate the three supervised classifiers ----
    g=list(csv.DictReader(open(D("human_gold_372.csv"),encoding="utf-8")))
    sent={r["code_id"]:r["sentence"] for r in g}; disc={r["code_id"]:int(r["gold_lenient"]) for r in g}
    ai={r["code_id"]:r for r in csv.DictReader(open(D("human_gold_aggr_infl_372.csv"),encoding="utf-8"))}
    ids=list(sent); S=[sent[c] for c in ids]
    Y={"aggressive":np.array([int(ai[c]["aggr_lenient"]) for c in ids]),
       "inflammatory":np.array([int(ai[c]["infl_lenient"]) for c in ids]),
       "discriminatory":np.array([disc[c] for c in ids])}
    print("=== Supervised cross-validated F1 (Table 11, supervised row) ===")
    for k in Y:
        f,sd=cv_f1(S,Y[k]); print(f"  {k:14s} F1 = {f:.2f} ± {sd:.2f}  (positives {int(Y[k].sum())}/{len(S)})")

    # ---- 2. train on all labels, classify the full 2008-2024 corpus ----
    V=build_vocab(S); ID=idf(S,V); Xtr=feats(S,V,ID); models={k:lr(Xtr,Y[k]) for k in Y}
    counts={}
    for path in [D("corpus_sentences.csv"),D("corpus_2024.csv")]:
        rows=[r for r in csv.DictReader(open(path,encoding="utf-8")) if int(str(r["date"])[:4])>=2008]
        X=feats([str(r["sentence"]) for r in rows],V,ID)
        pred={k:(1/(1+np.exp(-(X@w+b)))>=0.5).astype(int) for k,(w,b) in models.items()}
        for i,r in enumerate(rows):
            c=counts.setdefault((r["date"],r["party"]),{k:0 for k in Y})
            for k in Y: c[k]+=int(pred[k][i])

    # ---- 3. merge into panel + regress ----
    P=pd.read_csv(D("panel_2008_2024.csv"))
    def cnt(dt,pt,cat): return counts.get((dt,pt),{}).get(cat,np.nan)
    for cat,s in [("aggressive","aggr"),("inflammatory","infl"),("discriminatory","disc")]:
        P[f"{s}_d"]=[cnt(r,"Dem",cat) for r in P["date"]]; P[f"{s}_r"]=[cnt(r,"Rep",cat) for r in P["date"]]
        P[f"{s}_u"]=P[f"{s}_d"]+P[f"{s}_r"]
    P["high_sample"]=(P["sample1"]>P["sample1"].median()).astype(int)
    years=sorted(P["year"].unique())
    for y in years[1:]: P[f"yr{y}"]=(P["year"]==y).astype(int)
    yrc=[f"yr{y}" for y in years[1:]]; ctrl=["pre_democrats","pre_republicans","Immigration","ForeignPolicy","AbortionRights"]
    src=["aggr_d","aggr_r","infl_d","infl_r","disc_d","disc_r"]; base=["aggr_u","infl_u","disc_u"]
    for v in src: P[v+"_hs"]=P[v]*P["high_sample"]
    ss=src+[v+"_hs" for v in src]+["high_sample"]
    def wls(dv,X):
        df=P.dropna(subset=[dv]+X+["sample1"]).copy()
        M=np.column_stack([np.ones(len(df))]+[df[c].values.astype(float) for c in X]); y=df[dv].values.astype(float)
        w=df["sample1"].values.astype(float); w=w/w.mean(); cl=df["date"].values
        XtW=M.T*w; bread=np.linalg.inv(XtW@M); beta=bread@(XtW@y); res=y-M@beta
        meat=np.zeros((M.shape[1],)*2)
        for c in set(cl):
            i=cl==c; sX=(M[i]*w[i][:,None]).T@res[i]; meat+=np.outer(sX,sX)
        G=len(set(cl)); n,k=M.shape; adj=(G/(G-1))*((n-1)/(n-k)); Vc=bread@meat@bread*adj
        se=np.sqrt(np.diag(Vc)); t=beta/se; p=[tp_two(tt,G-1) for tt in t]
        return dict(zip(["const"]+X,zip(beta,p))),G,n
    def star(p): return "***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
    def show(title,dv,X,keep):
        d,G,n=wls(dv,X); print(f"\n## {title} — {dv}  (N={n}, clusters={G})")
        for k in keep:
            b,p=d[k]; print(f"   {k:16s} {b:+.3f} ({p:.3f}){star(p)}")
    print("\n=== SUPERVISED-MEASURE REGRESSIONS (2008-2024) ===")
    for dv in ["change_democrats","change_republicans"]:
        show("BASELINE",dv,base+ctrl+yrc,base)
    for dv in ["change_democrats","change_republicans"]:
        show("SOURCE-SPECIFIC",dv,src+ctrl+yrc,src)
    for dv in ["change_democrats","change_republicans"]:
        show("SAMPLE-SIZE (interactions)",dv,ss+ctrl+yrc,[v+"_hs" for v in src])

if __name__=="__main__":
    main()
