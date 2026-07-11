#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author(s) withheld for peer review
"""
Reproduces the supervised-primary (validated-measure) analysis over 2004-2024.

Pipeline:
  1. Train TF-IDF + logistic-regression classifiers for AGGRESSIVE, INFLAMMATORY, and
     DISCRIMINATORY on the 372-sentence double-coded human gold standards, with
     inverse-probability weights (samp_weight) correcting the enriched-sample design.
  2. Score the full sentence corpus for all eighteen debates (2004 + 2008-2020 + 2024)
     and aggregate to candidate-debate counts by speaker party.
  3. Stack the 2004-2024 candidate-poll panel (2004-2020 from the .dta + 2024 panel),
     median-impute missing poll sample sizes, and merge the validated counts.
  4. Estimate (a) baseline pooled and (b) source-specific WLS, one rhetorical category at
     a time (the full six-coefficient model is under-identified with eighteen debates),
     weighted by poll sample size, with election-year fixed effects and debate-clustered
     SEs; p-values use a self-contained Student-t.

Inputs (../data): human_gold_aggr_infl_372.csv, human_gold_discriminatory_372.csv,
  discriminatory_sampling_key.csv, corpus_2004.csv, corpus_sentences_2008_2020.csv,
  corpus_2024.csv, panel_2008_2024.csv, poll_final_data_with_change.dta
Run:  python reproduce_supervised_primary.py     (requires numpy, pandas)
"""
import os, re, math
import numpy as np, pandas as pd
from collections import Counter
D=os.path.join(os.path.dirname(__file__),"..","data")

# ---------- TF-IDF + IPW logistic regression ----------
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
def lr_w(X,y,sw,l2=1.0,it=800,lr_=0.5):
    n,p=X.shape; w=np.zeros(p); b=0.0
    cw0=n/(2*max((sw*(1-y)).sum(),1)); cw1=n/(2*max((sw*y).sum(),1)); cw=np.where(y==1,cw1,cw0)*sw
    for _ in range(it):
        pr=1/(1+np.exp(-(X@w+b))); g=(pr-y)*cw; w-=lr_*(X.T@g/n+l2*w/n); b-=lr_*g.mean()
    return w,b

# ---------- labels + weights ----------
key=pd.read_csv(os.path.join(D,"discriminatory_sampling_key.csv"))[["code_id","samp_weight"]]
wmap=dict(zip(key.code_id,key.samp_weight))
ai=pd.read_csv(os.path.join(D,"human_gold_aggr_infl_372.csv"))
S=ai["sentence"].astype(str).tolist()
A=((ai["aggr_coder1"].fillna(0)+ai["aggr_coder2"].fillna(0))>0).astype(int).values
I=((ai["infl_coder1"].fillna(0)+ai["infl_coder2"].fillna(0))>0).astype(int).values
sw=np.array([float(wmap.get(c,1.0)) for c in ai["code_id"]]); sw=sw/sw.mean()
dg=pd.read_csv(os.path.join(D,"human_gold_discriminatory_372.csv"))
Sd=dg["sentence"].astype(str).tolist(); Yd=dg["gold_lenient"].astype(int).values
dsw=np.array([float(wmap.get(c,1.0)) for c in dg["code_id"]]); dsw=dsw/dsw.mean()

# ---------- score full 2004-2024 corpus ----------
corp=pd.concat([pd.read_csv(os.path.join(D,"corpus_2004.csv")),
                pd.read_csv(os.path.join(D,"corpus_sentences_2008_2020.csv")),
                pd.read_csv(os.path.join(D,"corpus_2024.csv"))],ignore_index=True)
corp["sentence"]=corp["sentence"].astype(str)
def score(St,y,swt):
    v=vocab(St); id_=idf(St,v); w,b=lr_w(feats(St,v,id_),y,swt)
    return (1/(1+np.exp(-(feats(corp["sentence"].tolist(),v,id_)@w+b)))>=0.5).astype(int)
corp["aggr_sup"]=score(S,A,sw); corp["infl_sup"]=score(S,I,sw); corp["disc_sup"]=score(Sd,Yd,dsw)
cnt=corp.groupby(["date","party"])[["aggr_sup","infl_sup","disc_sup"]].sum().reset_index()
print("debates scored:",cnt["date"].nunique())

# ---------- build 2004-2024 panel ----------
dta=pd.read_stata(os.path.join(D,"poll_final_data_with_change.dta"))
keepc=["debate_date","change_democrats","change_republicans","pre_democrats","pre_republicans","sample1","Immigration","ForeignPolicy","AbortionRights","incumbent_democrat"]
p04=dta[dta["year"].round()==2004][keepc].copy()
pan=pd.read_csv(os.path.join(D,"panel_2008_2024.csv")); pan["debate_date"]=pan.get("debate_date",pan.get("date"))
full=pd.concat([p04,pan[keepc]],ignore_index=True)
full["dk"]=pd.to_datetime(full["debate_date"]).dt.strftime("%Y-%m-%d"); full["yr"]=pd.to_datetime(full["debate_date"]).dt.year
full["sample1"]=full["sample1"].fillna(full["sample1"].median())
def wide(df,val,pfx):
    df=df.copy(); df["dk"]=pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    w=df.pivot_table(index="dk",columns="party",values=val,aggfunc="max").reset_index(); w.columns.name=None
    return w.rename(columns={"Dem":pfx+"_d","Rep":pfx+"_r"})
m=full.merge(wide(cnt,"aggr_sup","aggr"),on="dk").merge(wide(cnt,"infl_sup","infl"),on="dk").merge(wide(cnt,"disc_sup","disc"),on="dk")
for b in ["aggr","infl","disc"]: m[b+"_tot"]=m[b+"_d"]+m[b+"_r"]
yr=pd.get_dummies(m["yr"],prefix="yr",drop_first=True).astype(float)
for c in yr.columns: m[c]=yr[c].values
yc=list(yr.columns); ctrl=["pre_democrats","pre_republicans","Immigration","ForeignPolicy","AbortionRights"]
print("panel rows:",len(m)," debates:",m["dk"].nunique())

# ---------- WLS + cluster SE + t(G-1) ----------
def betacf(a,b,x):
    EPS=3e-12;FP=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1.0;dd=1-qab*x/qap;dd=FP if abs(dd)<FP else dd;dd=1/dd;h=dd
    for mm in range(1,300):
        m2=2*mm;aa=mm*(b-mm)*x/((qam+m2)*(a+m2));dd=1+aa*dd;dd=FP if abs(dd)<FP else dd;c=1+aa/c;c=FP if abs(c)<FP else c;dd=1/dd;h*=dd*c
        aa=-(a+mm)*(qab+mm)*x/((a+m2)*(qap+m2));dd=1+aa*dd;dd=FP if abs(dd)<FP else dd;c=1+aa/c;c=FP if abs(c)<FP else c;dd=1/dd;de=dd*c;h*=de
        if abs(de-1)<EPS:break
    return h
def betai(a,b,x):
    if x<=0:return 0.0
    if x>=1:return 1.0
    lb=math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x);bt=math.exp(lb)
    return bt*betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-bt*betacf(b,a,1-x)/b
def t_p(t,df): df=max(int(df),1); return betai(df/2.0,0.5,df/(df+t*t))
def st(p): return "*"*sum(p<x for x in (0.1,0.05,0.01))
def wls(d,y,xcols):
    sub=d.dropna(subset=[y,"sample1","dk"]+xcols).copy(); X=sub[xcols].astype(float).copy(); X.insert(0,"const",1.0); cols=list(X.columns); Xn=X.values
    keep=[]
    for j in range(Xn.shape[1]):
        if np.linalg.matrix_rank(Xn[:,keep+[j]])==len(keep)+1: keep.append(j)
    Xn=Xn[:,keep]; cols=[cols[j] for j in keep]
    yv=sub[y].astype(float).values; w=sub["sample1"].astype(float).values; W=np.sqrt(w); Xw=Xn*W[:,None]; inv=np.linalg.inv(Xw.T@Xw); b=inv@(Xw.T@(yv*W)); res=yv-Xn@b
    g=sub["dk"].values; meat=np.zeros((Xn.shape[1],)*2)
    for c in np.unique(g):
        idx=g==c; s=Xw[idx].T@(res*W)[idx]; meat+=np.outer(s,s)
    G=len(np.unique(g)); n,k=Xn.shape; adj=(G/(G-1))*((n-1)/(n-k)); V=adj*inv@meat@inv; se=np.sqrt(np.clip(np.diag(V),0,None)); tv=b/se
    return {c:(b[i],t_p(tv[i],G-1)) for i,c in enumerate(cols)},G,n

print("\n=== BASELINE (pooled), 2004-2024 ===")
for y in ["change_democrats","change_republicans"]:
    r,G,n=wls(m,y,["aggr_tot","infl_tot","disc_tot"]+ctrl+yc); print(f" DV={y[7:10]} N={n} G={G}: "+", ".join(f"{v} {r[v][0]:+.4f}({r[v][1]:.3f}){st(r[v][1])}" for v in ["aggr_tot","infl_tot","disc_tot"]))
print("\n=== SOURCE-SPECIFIC by category, 2004-2024 (full 6-reg model is under-identified) ===")
for cat in ["aggr","infl","disc"]:
    for y in ["change_democrats","change_republicans"]:
        r,G,n=wls(m,y,[cat+"_d",cat+"_r"]+ctrl+yc); a=r[cat+"_d"]; b=r[cat+"_r"]
        print(f" {cat} DV={y[7:10]} (G={G},N={n}): {cat}_d {a[0]:+.3f}({a[1]:.3f}){st(a[1])}  {cat}_r {b[0]:+.3f}({b[1]:.3f}){st(b[1])}")
m.to_csv(os.path.join(D,"panel_2004_2024_validated.csv"),index=False)
print("\n(Wild-bootstrap survivors on Republican change: Inflammatory_Words_Dem p~0.009, Discriminatory_Words_Dem p~0.003, Discriminatory_Words_Rep p~0.059; see validation_and_robustness.py.)")
