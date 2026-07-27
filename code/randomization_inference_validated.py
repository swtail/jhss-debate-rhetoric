# Randomization inference on the VALIDATED source-specific survivors (vectorized).
# Author(s) withheld for peer review.
#
# Few-cluster check complementary to the wild bootstrap: permute the debate-level
# rhetoric profiles (paired Dem/Rep counts for a category) across the eighteen debates,
# broadcast each permuted profile back to that debate's polls, and recompute the
# debate-clustered t for the target coefficient. The randomization p-value is the share
# of permutations whose |t| is at least the observed |t|. We report the three wild-
# bootstrap survivors on the Republican-share change and the matching PLACEBO
# (pre-debate change) for the headline Democratic-inflammatory term.
#
# Requires: numpy, pandas.  Reads: data/panel_2004_2024_validated.csv,
#   data/poll_final_data_with_change.dta, data/supervised_counts_2004_2024.csv
import os, numpy as np, pandas as pd
D = os.path.join(os.path.dirname(__file__), "..", "data")
ctrl = ["pre_democrats","pre_republicans","Immigration","ForeignPolicy","AbortionRights"]

def prep(panel, y, cat, yc):
    xcols=[cat+"_d",cat+"_r"]+ctrl+yc
    sub=panel.dropna(subset=[y,"sample1","dk"]+xcols).copy().reset_index(drop=True)
    F=sub[ctrl+yc].astype(float).copy(); F.insert(0,"const",1.0); Fn=F.values
    keep=[]
    for j in range(Fn.shape[1]):
        if np.linalg.matrix_rank(Fn[:,keep+[j]])==len(keep)+1: keep.append(j)
    Fn=Fn[:,keep]
    yv=sub[y].astype(float).values; w=sub["sample1"].astype(float).values; W=np.sqrt(w)
    g=sub["dk"].values; cl,ci=np.unique(g,return_inverse=True); G=len(cl)
    prof=sub.groupby("dk")[[cat+"_d",cat+"_r"]].first().reset_index()
    dkpos={dk:i for i,dk in enumerate(prof["dk"])}
    pe=np.array([dkpos[dk] for dk in sub["dk"]])
    return Fn,yv,W,ci,G,pe,prof[cat+"_d"].values.astype(float),prof[cat+"_r"].values.astype(float)

def t_target(Fn,Cd,Cr,yv,W,ci,G,which):
    Xn=np.column_stack([Cd,Cr,Fn]); k=Xn.shape[1]; n=Xn.shape[0]
    Xw=Xn*W[:,None]; inv=np.linalg.inv(Xw.T@Xw); b=inv@(Xw.T@(yv*W)); res=yv-Xn@b
    adj=(G/(G-1))*((n-1)/(n-k)); meat=np.zeros((k,k))
    for c in range(G):
        idx=ci==c; s=Xw[idx].T@((res*W)[idx]); meat+=np.outer(s,s)
    V=adj*inv@meat@inv; return b[which]/np.sqrt(max(V[which,which],1e-300))

def ri(panel,y,cat,which,yc,B=1999,seed=7):
    Fn,yv,W,ci,G,pe,Pd,Pr=prep(panel,y,cat,yc)
    t_obs=t_target(Fn,Pd[pe],Pr[pe],yv,W,ci,G,which)
    rng=np.random.default_rng(seed); cnt=0
    for _ in range(B):
        p=rng.permutation(G)
        if abs(t_target(Fn,Pd[p][pe],Pr[p][pe],yv,W,ci,G,which))>=abs(t_obs): cnt+=1
    return t_obs,(cnt+1)/(B+1)

if __name__=="__main__":
    m=pd.read_csv(os.path.join(D,"panel_2004_2024_validated.csv"))
    yc=[c for c in m.columns if c.startswith("yr_")]
    print("RANDOMIZATION INFERENCE — validated, Republican-share change (post-debate)")
    for cat,which,lab in [("infl",0,"Inflammatory_Words_Dem"),("disc",0,"Discriminatory_Words_Dem"),("disc",1,"Discriminatory_Words_Rep")]:
        t,p=ri(m,"change_republicans",cat,which,yc); print(f"   {lab:26s} t={t:+.2f}  RI p={p:.3f}  (G=18)")
    dta=pd.read_stata(os.path.join(D,"poll_final_data_with_change.dta"))
    sc=pd.read_csv(os.path.join(D,"supervised_counts_2004_2024.csv"))
    w=sc.pivot(index="date",columns="party",values=["aggr_sup","infl_sup","disc_sup"])
    w.columns=[f"{a}_{'d' if b=='Dem' else 'r'}" for a,b in w.columns]; w=w.reset_index()
    w=w.rename(columns={"infl_sup_d":"infl_d","infl_sup_r":"infl_r","disc_sup_d":"disc_d","disc_sup_r":"disc_r"})
    dta["date"]=dta["debate_date"].astype(str).str[:10]; pl=dta.merge(w,on="date",how="inner"); pl["dk"]=pl["date"]
    ycp=[]
    for yv2 in sorted(pl["date"].str[:4].unique())[1:]:
        col="yr_"+yv2; pl[col]=(pl["date"].str[:4]==yv2).astype(float); ycp.append(col)
    t,p=ri(pl,"d_republicans","infl",0,ycp); print(f"PLACEBO (pre-debate): Inflammatory_Words_Dem  t={t:+.2f}  RI p={p:.3f}  (G=12)")
