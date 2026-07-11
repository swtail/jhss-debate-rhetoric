# Placebo-TREATMENT falsification: content-free rhetoric should not move polls.
# Author(s) withheld for peer review.
#
# The substantive claim is that HOSTILE rhetoric (aggressive, inflammatory, discriminatory)
# moves post-debate polling. As a falsification, the hostile counts are replaced by
# content-free measures that carry no hostile content: (a) a candidate's TOTAL sentences
# spoken in the debate (verbosity/salience) and (b) NON-HOSTILE sentences (total minus the
# three flagged categories). These are run through the identical source-specific WLS
# (poll-sample-size weights, election-year fixed effects, debate-clustered SEs) with wild
# cluster bootstrap p-values. If the content-free treatments also predict polling, the
# apparent hostile-rhetoric associations are generic debate salience or length, not content;
# if they are null, that supports content specificity.
#
# Requires: numpy, pandas.  Reads: data/corpus_*.csv, data/panel_2004_2024_validated.csv
import os, numpy as np, pandas as pd
D=os.path.join(os.path.dirname(__file__),"..","data")
ctrl=["pre_democrats","pre_republicans","Immigration","ForeignPolicy","AbortionRights"]

# total sentences per candidate-debate
files=[os.path.join(D,f) for f in ["corpus_2004.csv","corpus_sentences_2008_2020.csv","corpus_2024.csv"]]
corp=pd.concat([pd.read_csv(f).rename(columns=str.lower)[["date","party","sentence"]] for f in files],ignore_index=True)
corp["dk"]=pd.to_datetime(corp["date"]).dt.strftime("%Y-%m-%d")
tot=corp.groupby(["dk","party"]).size().reset_index(name="n")
w=tot.pivot(index="dk",columns="party",values="n").reset_index().rename(columns={"Dem":"total_d","Rep":"total_r"})

m=pd.read_csv(os.path.join(D,"panel_2004_2024_validated.csv")).merge(w,on="dk",how="left")
m["nonhostile_d"]=m["total_d"]-(m["aggr_d"]+m["infl_d"]+m["disc_d"])
m["nonhostile_r"]=m["total_r"]-(m["aggr_r"]+m["infl_r"]+m["disc_r"])
yc=[c for c in m.columns if c.startswith("yr_")]

def prep(y,xv):
    sub=m.dropna(subset=[y,"sample1","dk"]+xv+ctrl).copy().reset_index(drop=True)
    X=sub[xv+ctrl+yc].astype(float).copy(); X.insert(0,"const",1.0); cols=list(X.columns); Xn=X.values
    keep=[]
    for j in range(Xn.shape[1]):
        if np.linalg.matrix_rank(Xn[:,keep+[j]])==len(keep)+1: keep.append(j)
    Xn=Xn[:,keep]; cols=[cols[j] for j in keep]
    yv=sub[y].astype(float).values; wt=sub["sample1"].astype(float).values; g=sub["dk"].values
    return Xn,cols,yv,wt,g

def fit(Xn,cols,yv,wt,g,tgt):
    j=cols.index(tgt); W=np.sqrt(wt); Xw=Xn*W[:,None]; inv=np.linalg.inv(Xw.T@Xw)
    b=inv@(Xw.T@(yv*W)); res=yv-Xn@b
    cl,ci=np.unique(g,return_inverse=True); G=len(cl); n,k=Xn.shape; adj=(G/(G-1))*((n-1)/(n-k))
    meat=np.zeros((k,k))
    for c in range(G):
        idx=ci==c; s=Xw[idx].T@((res*W)[idx]); meat+=np.outer(s,s)
    V=adj*inv@meat@inv; se=np.sqrt(max(V[j,j],1e-300)); return b[j],se,b[j]/se,G,n,inv,Xw,cl,ci,adj,k

def wildp(Xn,cols,yv,wt,g,tgt,B=4999,seed=13):
    b,se,tobs,G,n,inv,Xw,cl,ci,adj,k=fit(Xn,cols,yv,wt,g,tgt)
    j=cols.index(tgt); W=np.sqrt(wt); Bm=inv@Xw.T
    Xr=np.delete(Xn,j,axis=1); Xrw=Xr*W[:,None]; br=np.linalg.inv(Xrw.T@Xrw)@(Xrw.T@(yv*W)); resr=yv-Xr@br
    h=Xw@inv[j]; rng=np.random.default_rng(seed); SV=(rng.random((G,B))<.5)*2.-1.; sv=SV[ci]
    Ys=resr[:,None]*sv+(Xr@br)[:,None]; ba=Bm@(Ys*W[:,None]); rw=(Ys-Xn@ba)*W[:,None]
    cs=np.zeros((G,B)); np.add.at(cs,ci,h[:,None]*rw); Vjj=adj*np.sum(cs**2,axis=0)
    tb=ba[j]/np.sqrt(np.clip(Vjj,1e-300,None)); return b,se,(np.sum(np.abs(tb)>=abs(tobs))+1)/(B+1),G

print("="*74); print("PLACEBO-TREATMENT FALSIFICATION (content-free rhetoric)"); print("="*74)
for name,pre in [("Total sentences (verbosity)","total"),("Non-hostile sentences","nonhostile")]:
    print("\n"+name+":")
    for y,lab in [("change_democrats","D Democratic"),("change_republicans","D Republican")]:
        Xn,cols,yv,wt,g=prep(y,[pre+"_d",pre+"_r"])
        for suf in ["_d","_r"]:
            b,se,p,G=wildp(Xn,cols,yv,wt,g,pre+suf)
            print(f"   {pre+suf:14s} DV={lab:13s} coef={b:+.5f}  SE={se:.5f}  wild p={p:.3f} (G={G})")
print("\nReference (hostile counts, same spec, Republican-share change): infl_d wild p≈0.009, disc_d≈0.003, disc_r≈0.059.")
