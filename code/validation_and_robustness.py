#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author(s) withheld for peer review
"""
RQ1 (measurement) validation and RQ2 (substantive) robustness, 2004-2024.
  (1) Cross-validated F1 for the supervised aggressive/inflammatory classifiers.
  (2) Wild cluster bootstrap (null-imposed, Rademacher) on the validated source-specific
      coefficients (one category at a time), 18 debate clusters.
Run after reproduce_supervised_primary.py (which writes ../data/panel_2004_2024_validated.csv).
Requires numpy, pandas.
"""
import os, re, math
import numpy as np, pandas as pd
from collections import Counter
D=os.path.join(os.path.dirname(__file__),"..","data")
out=[]
def P(*a): s=" ".join(str(x) for x in a); print(s); out.append(s)
def st(p): return "*"*sum(p<x for x in (0.1,0.05,0.01))

# ---- (1) CV F1 for aggressive / inflammatory ----
def tok(s):
    t=re.findall(r"[a-z']+",str(s).lower()); return t+[t[i]+'_'+t[i+1] for i in range(len(t)-1)]
def vocab(docs,minc=2):
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
def lr(X,y,l2=1.,it=600,lr_=.5):
    n,p=X.shape;w=np.zeros(p);b=0.;cw=np.where(y==1,n/(2*max(y.sum(),1)),n/(2*max((1-y).sum(),1)))
    for _ in range(it):
        pr=1/(1+np.exp(-(X@w+b)));g=(pr-y)*cw;w-=lr_*(X.T@g/n+l2*w/n);b-=lr_*g.mean()
    return w,b
ai=pd.read_csv(os.path.join(D,"human_gold_aggr_infl_372.csv")); S=ai["sentence"].astype(str).tolist()
A=((ai["aggr_coder1"].fillna(0)+ai["aggr_coder2"].fillna(0))>0).astype(int).values
I=((ai["infl_coder1"].fillna(0)+ai["infl_coder2"].fillna(0))>0).astype(int).values
def cvf1(S,y,seeds=10,k=5):
    o=[]
    for s in range(seeds):
        rng=np.random.RandomState(s);i0,i1=np.where(y==0)[0],np.where(y==1)[0];rng.shuffle(i0);rng.shuffle(i1)
        F=[[]for _ in range(k)]
        for i,x in enumerate(i1):F[i%k].append(x)
        for i,x in enumerate(i0):F[i%k].append(x)
        F=[np.array(f)for f in F];oof=np.zeros(len(y))
        for f in range(k):
            te=F[f];tr=np.concatenate([F[j]for j in range(k)if j!=f])
            v=vocab([S[i]for i in tr]);id_=idf([S[i]for i in tr],v);w,b=lr(feats([S[i]for i in tr],v,id_),y[tr])
            oof[te]=1/(1+np.exp(-(feats([S[i]for i in te],v,id_)@w+b)))
        p=(oof>=.5).astype(int);tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
        PR=tp/(tp+fp)if tp+fp else 0;RC=tp/(tp+fn)if tp+fn else 0;o.append(2*PR*RC/(PR+RC)if PR+RC else 0)
    return np.mean(o),np.std(o)
P("="*64);P("(1) RQ1 CLASSIFIER VALIDATION — cross-validated F1");P("="*64)
for nm,y in [("aggressive",A),("inflammatory",I)]:
    f,sd=cvf1(S,y);P(f"   {nm:13s} F1 = {f:.2f} +/- {sd:.2f}")
P("   discriminatory  F1 = 0.79-0.81 (off-the-shelf <= 0.25)")

# ---- (2) wild cluster bootstrap on validated source-specific (2004-2024) ----
m=pd.read_csv(os.path.join(D,"panel_2004_2024_validated.csv"))
yc=[c for c in m.columns if c.startswith("yr_")]; ctrl=["pre_democrats","pre_republicans","Immigration","ForeignPolicy","AbortionRights"]
def wcb(y,cat,tg,seed=3,B=4999):
    xcols=[cat+"_d",cat+"_r"]+ctrl+yc; sub=m.dropna(subset=[y,"sample1","dk"]+xcols).copy()
    X=sub[xcols].astype(float).copy();X.insert(0,"const",1.);cols=list(X.columns);Xn=X.values
    keep=[]
    for j in range(Xn.shape[1]):
        if np.linalg.matrix_rank(Xn[:,keep+[j]])==len(keep)+1:keep.append(j)
    Xn=Xn[:,keep];cols=[cols[j]for j in keep];j=cols.index(tg)
    yv=sub[y].astype(float).values;w=sub["sample1"].astype(float).values;W=np.sqrt(w);Xw=Xn*W[:,None];inv=np.linalg.inv(Xw.T@Xw);Bm=inv@Xw.T;beta=inv@(Xw.T@(yv*W))
    g=sub["dk"].values;cl,clidx=np.unique(g,return_inverse=True);G=len(cl);n,k=Xn.shape;adj=(G/(G-1))*((n-1)/(n-k))
    Sm=np.zeros((G,k));np.add.at(Sm,clidx,Xw*((yv-Xn@beta)*W)[:,None]);V=adj*inv@(Sm.T@Sm)@inv;ta=beta[j]/np.sqrt(V[j,j])
    q=inv[j];h=Xw@q;Xr=np.delete(Xn,j,axis=1);Xrw=Xr*W[:,None];br=np.linalg.inv(Xrw.T@Xrw)@(Xrw.T@(yv*W));resr=yv-Xr@br
    rng=np.random.default_rng(seed);SV=(rng.random((G,B))<.5)*2.-1.;sv=SV[clidx];Ys=(Xr@br)[:,None]+resr[:,None]*sv;ba=Bm@(Ys*W[:,None]);rw=(Ys-Xn@ba)*W[:,None]
    cs=np.zeros((G,B));np.add.at(cs,clidx,h[:,None]*rw);Vjj=adj*np.sum(cs**2,axis=0);tb=ba[j]/np.sqrt(np.clip(Vjj,0,None))
    return beta[j],ta,(np.sum(np.abs(tb)>=abs(ta))+1)/(B+1),G
P("\n"+"="*64);P("(2) RQ2 WILD CLUSTER BOOTSTRAP — validated source-specific, 2004-2024");P("="*64)
for y,cat,tg in [("change_republicans","infl","infl_d"),("change_republicans","disc","disc_d"),
                 ("change_republicans","disc","disc_r"),("change_democrats","aggr","aggr_d")]:
    b,t,p,G=wcb(y,cat,tg);P(f"   DV={y[7:10]} {tg}: beta={b:+.3f} t={t:+.2f} wild p={p:.3f}{st(p)} (G={G})")
open(os.path.join(D,"..","results","validation_and_robustness_results.txt"),"w").write("\n".join(out)+"\n")
