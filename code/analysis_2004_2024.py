#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Bryan Kyung and Katie Kim
"""
Primary analysis: 2004-2024, original debate-level rhetoric counts.

Builds the 2004-2024 candidate-poll panel by combining the original 2004-2020 panel
(data/poll_final_data_with_change.dta) with the 2024 debates (data/panel_2008_2024.csv),
then estimates the baseline, source-specific, and sample-size-heterogeneity models with
WLS (weighted by poll sample size), election-year fixed effects, and standard errors
clustered by debate. Polls missing a reported sample size (including the 2004 polls) are
weighted at the median sample size; this is stated in the manuscript.

Run:  pip install numpy pandas
      python code/analysis_2004_2024.py
"""
import os, math
import numpy as np, pandas as pd

HERE=os.path.dirname(os.path.abspath(__file__)); D=lambda *p: os.path.join(os.path.dirname(HERE),"data",*p)
CTRL=["pre_democrats","pre_republicans","Immigration","ForeignPolicy","AbortionRights"]
SRC=["aggressive_words_d","aggressive_words_r","inflammatory_words_d","inflammatory_words_r",
     "discriminatory_words_d","discriminatory_words_r"]
BASE=["aggressive_words","inflammatory_words","discriminatory_words"]
COLS=["debate_date","year","change_democrats","change_republicans","pre_democrats","pre_republicans",
      "sample1","Immigration","ForeignPolicy","AbortionRights","incumbent_democrat"]+BASE+SRC

def build_panel():
    base=pd.read_stata(D("poll_final_data_with_change.dta"))
    base=base[(base["year"]>=2004)&(base["year"]<=2020)].copy()
    base["debate_date"]=base["debate_date"].astype(str).str[:10]
    p24=pd.read_csv(D("panel_2008_2024.csv")); p24=p24[p24["year"]==2024].copy()
    p24["debate_date"]=p24["date"].astype(str).str[:10]
    for c in COLS:
        if c not in base.columns: base[c]=np.nan
        if c not in p24.columns: p24[c]=np.nan
    P=pd.concat([base[COLS],p24[COLS]],ignore_index=True); P["year"]=P["year"].astype(int)
    P["sample1"]=P["sample1"].fillna(P["sample1"].median())   # median-impute missing poll sizes (incl. 2004)
    return P

# ---- cluster-robust WLS + Student-t p-values ----
def _bcf(a,b,x):
    F=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1.0;d=1-qab*x/qap;d=1/(F if abs(d)<F else d);h=d
    for m in range(1,300):
        m2=2*m;aa=m*(b-m)*x/((qam+m2)*(a+m2));d=1+aa*d;d=F if abs(d)<F else d;c=1+aa/c;c=F if abs(c)<F else c;d=1/d;h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));d=1+aa*d;d=F if abs(d)<F else d;c=1+aa/c;c=F if abs(c)<F else c;d=1/d;de=d*c;h*=de
        if abs(de-1)<3e-12:break
    return h
def tp(t,df):
    x=df/(df+t*t)
    def ib(a,b,x):
        if x<=0:return 0.0
        if x>=1:return 1.0
        lb=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b);f=math.exp(math.log(x)*a+math.log(1-x)*b-lb)
        return f/a*_bcf(a,b,x) if x<(a+1)/(a+b+2) else 1-f/b*_bcf(b,a,1-x)
    return ib(df/2,0.5,x)

def wls(P,dv,X):
    df=P.dropna(subset=[dv]+X+["sample1"]).copy()
    M=np.column_stack([np.ones(len(df))]+[df[c].values.astype(float) for c in X]); y=df[dv].values.astype(float)
    w=df["sample1"].values.astype(float); w=w/w.mean(); cl=df["debate_date"].values
    XtW=M.T*w; bread=np.linalg.inv(XtW@M); beta=bread@(XtW@y); res=y-M@beta
    meat=np.zeros((M.shape[1],)*2)
    for c in set(cl):
        i=cl==c; s=(M[i]*w[i][:,None]).T@res[i]; meat+=np.outer(s,s)
    G=len(set(cl)); n,k=M.shape; adj=(G/(G-1))*((n-1)/(n-k)); V=bread@meat@bread*adj
    se=np.sqrt(np.clip(np.diag(V),0,None)); t=np.divide(beta,se,out=np.zeros_like(beta),where=se>0)
    return {nm:(beta[i],tp(t[i],G-1)) for i,nm in enumerate(["const"]+X)},G,n

def main():
    P=build_panel()
    print(f"Panel: {len(P)} candidate-poll obs, {P['debate_date'].nunique()} debates, "
          f"years {sorted(P['year'].unique())}")
    years=sorted(P["year"].unique())
    for y in years[1:]: P[f"yr{y}"]=(P["year"]==y).astype(int)
    yrc=[f"yr{y}" for y in years[1:]]
    P["high_sample"]=(P["sample1"]>P["sample1"].median()).astype(int)
    for v in SRC: P[v+"_hs"]=P[v]*P["high_sample"]
    ss=SRC+[v+"_hs" for v in SRC]+["high_sample"]
    def star(p): return "***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
    def show(title,dv,X,keep):
        d,G,n=wls(P,dv,X); print(f"\n## {title} — {dv} (N={n}, clusters={G})")
        for k in keep:
            b,p=d[k]; print(f"   {k:24s} {b:+.3f} ({p:.3f}){star(p)}")
    for dv in ["change_democrats","change_republicans"]:
        show("BASELINE",dv,BASE+CTRL+yrc,BASE)
    for dv in ["change_democrats","change_republicans"]:
        show("SOURCE-SPECIFIC",dv,SRC+CTRL+yrc,SRC)
    for dv in ["change_democrats","change_republicans"]:
        show("SAMPLE-SIZE (interactions)",dv,ss+CTRL+yrc,[v+"_hs" for v in SRC])

if __name__=="__main__":
    main()
