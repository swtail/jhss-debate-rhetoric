# Sample-size heterogeneity on the VALIDATED supervised measure (Table 7).
# Author(s) withheld for peer review.
#
# For each rhetorical category we estimate, one category at a time (the full
# six-coefficient model is under-identified with eighteen debates), the source-
# specific terms together with their interactions with High_Sample_Size, an
# indicator equal to one for polls above the median sample size. We judge every
# rhetoric/interaction term with the same null-imposed wild cluster bootstrap
# (Rademacher weights, clustered by debate) used throughout the paper, because
# the asymptotic cluster-robust SEs over-reject with eighteen clusters.
#
# Requires: numpy, pandas.  Reads: data/panel_2004_2024_validated.csv
# (written by reproduce_supervised_primary.py).
import os, math
import numpy as np, pandas as pd
D = os.path.join(os.path.dirname(__file__), "..", "data")

# ---- self-contained Student-t two-sided p (regularized incomplete beta) ----
def betacf(a, b, x):
    EPS=3e-12; FP=1e-300; qab=a+b; qap=a+1; qam=a-1; c=1.0; d=1-qab*x/qap
    d=FP if abs(d)<FP else d; d=1/d; h=d
    for m in range(1,300):
        m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; d=FP if abs(d)<FP else d
        c=1+aa/c; c=FP if abs(c)<FP else c; d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=FP if abs(d)<FP else d
        c=1+aa/c; c=FP if abs(c)<FP else c; d=1/d; de=d*c; h*=de
        if abs(de-1)<EPS: break
    return h
def betai(a,b,x):
    if x<=0: return 0.0
    if x>=1: return 1.0
    lb=math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x); bt=math.exp(lb)
    return bt*betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-bt*betacf(b,a,1-x)/b
def tp(t,df): df=max(int(df),1); return betai(df/2.0,0.5,df/(df+t*t))
def star(p): return '*'*sum(p<th for th in (0.1,0.05,0.01))

m = pd.read_csv(os.path.join(D, "panel_2004_2024_validated.csv"))
m["High"] = (m["sample1"] > m["sample1"].median()).astype(float)
yc = [c for c in m.columns if c.startswith("yr_")]
ctrl = ["pre_democrats","pre_republicans","Immigration","ForeignPolicy","AbortionRights"]

def design(frame,y,xv):
    sub=frame.dropna(subset=[y,"sample1","dk"]+xv+ctrl).copy()
    X=sub[xv+ctrl+yc].astype(float).copy(); X.insert(0,"const",1.0); cols=list(X.columns); Xn=X.values
    keep=[]
    for j in range(Xn.shape[1]):
        if np.linalg.matrix_rank(Xn[:,keep+[j]])==len(keep)+1: keep.append(j)
    return sub,Xn[:,keep],[cols[j] for j in keep],sub[y].astype(float).values,sub["sample1"].astype(float).values,sub["dk"].values

def fitj(Xn,yv,wt,g,j):
    W=np.sqrt(wt); Xw=Xn*W[:,None]; inv=np.linalg.inv(Xw.T@Xw); b=inv@(Xw.T@(yv*W)); res=yv-Xn@b
    G=len(np.unique(g)); n,k=Xn.shape; adj=(G/(G-1))*((n-1)/(n-k)); meat=np.zeros((k,k))
    for cl in np.unique(g):
        idx=g==cl; s=Xw[idx].T@((res*W)[idx]); meat+=np.outer(s,s)
    V=adj*inv@meat@inv; return b[j], b[j]/np.sqrt(max(V[j,j],1e-300)), G, n

def wild(frame,y,xv,target,B=4999,seed=11):
    sub,Xn,cols,yv,wt,g=design(frame,y,xv); j=cols.index(target)
    bpt,t_act,G,n=fitj(Xn,yv,wt,g,j)
    W=np.sqrt(wt); Xw=Xn*W[:,None]; inv=np.linalg.inv(Xw.T@Xw); Bmat=inv@Xw.T
    Xr=np.delete(Xn,j,axis=1); Xrw=Xr*W[:,None]
    br=np.linalg.inv(Xrw.T@Xrw)@(Xrw.T@(yv*W)); resr=yv-Xr@br; fitr=Xr@br
    cl,ci=np.unique(g,return_inverse=True); k=Xn.shape[1]; adj=(G/(G-1))*((n-1)/(n-k))
    h=Xw@inv[j]; rng=np.random.default_rng(seed); SV=(rng.random((G,B))<0.5)*2.-1.; sv=SV[ci]
    Ys=resr[:,None]*sv+fitr[:,None]; Yw=Ys*W[:,None]; ba=Bmat@Yw
    resa=Ys-Xn@ba; rw=resa*W[:,None]; HR=h[:,None]*rw; cs=np.zeros((G,B)); np.add.at(cs,ci,HR)
    Vjj=adj*np.sum(cs**2,axis=0); ta=ba[j]/np.sqrt(np.clip(Vjj,1e-300,None))
    pw=(np.sum(np.abs(ta)>=abs(t_act))+1)/(B+1)
    return bpt, tp(t_act,G-1), pw, G, n

print("="*78)
print("TABLE 7  Sample-size heterogeneity, VALIDATED measure (wild cluster bootstrap)")
print("  High = poll sample size above median; one category per regression.")
print("="*78)
for y,lab in [("change_democrats","(1) D Democratic"),("change_republicans","(2) D Republican")]:
    print(f"\nDV = {lab}")
    for cat,nm in [("aggr","aggressive"),("infl","inflammatory"),("disc","discriminatory")]:
        m["Hd"]=m[cat+"_d"]*m["High"]; m["Hr"]=m[cat+"_r"]*m["High"]
        xv=[cat+"_d",cat+"_r","Hd","Hr","High"]
        labels=[(cat+"_d",f"{nm}_Dem"),(cat+"_r",f"{nm}_Rep"),
                ("Hd",f"{nm}_Dem x High"),("Hr",f"{nm}_Rep x High")]
        for tgt,tlab in labels:
            b,pa,pw,G,n=wild(m,y,xv,tgt)
            print(f"   {tlab:26s} {b:+.4f}  (wild p={pw:.3f}){star(pw)}   [asy p={pa:.3f}]  N={n} G={G}")
