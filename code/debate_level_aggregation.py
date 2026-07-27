# Debate-level aggregation (Table 10) on the VALIDATED supervised measure.
# Author(s) withheld for peer review.
#
# Rhetoric varies only across debates, so the substantively honest unit is the
# debate, not the poll. We collapse the 695-poll panel to the 18 debates: the
# dependent variable is the sample-size-weighted mean post-debate change in each
# share within the debate, and the regressors are the pooled rhetorical counts and
# the debate-level pre-debate baselines and topic controls. We fit OLS with
# heteroskedasticity-robust (HC1) standard errors. With only eighteen observations
# the specification is severely underpowered, so wide p-values are expected and are
# NOT evidence against the poll-level associations; the appropriate few-cluster test
# is the wild cluster bootstrap in validation_and_robustness.py / placebo_validated.py.
#
# Requires: numpy, pandas.  Reads: data/panel_2004_2024_validated.csv
import os, math
import numpy as np, pandas as pd
D = os.path.join(os.path.dirname(__file__), "..", "data")

def betacf(a,b,x):
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

# weighted-mean collapse to the debate
def wmean(s, w):
    w=w.loc[s.index]; return np.average(s.astype(float), weights=w)
rows=[]
for dk,gdf in m.groupby("dk"):
    w=gdf["sample1"].astype(float)
    rows.append(dict(dk=dk,
        cd=wmean(gdf["change_democrats"],w), cr=wmean(gdf["change_republicans"],w),
        aggr=gdf["aggr_tot"].iloc[0], infl=gdf["infl_tot"].iloc[0], disc=gdf["disc_tot"].iloc[0],
        pred=wmean(gdf["pre_democrats"],w), prer=wmean(gdf["pre_republicans"],w),
        Imm=gdf["Immigration"].astype(float).mean(), FP=gdf["ForeignPolicy"].astype(float).mean(),
        Ab=gdf["AbortionRights"].astype(float).mean()))
g=pd.DataFrame(rows)
print("debate-level observations:", len(g))

def ols_hc1(y,Xcols):
    sub=g.dropna(subset=[y]+Xcols).copy()
    X=sub[Xcols].astype(float).copy(); X.insert(0,"const",1.0); cols=list(X.columns); Xn=X.values
    yv=sub[y].astype(float).values; inv=np.linalg.inv(Xn.T@Xn); b=inv@(Xn.T@yv); res=yv-Xn@b
    n,k=Xn.shape; meat=(Xn*res[:,None]).T@(Xn*res[:,None]); V=inv@meat@inv*(n/(n-k))
    se=np.sqrt(np.clip(np.diag(V),0,None)); t=b/se
    return {c:(b[i],tp(t[i],n-k)) for i,c in enumerate(cols)}, n, k

print("="*70); print("TABLE 10  Debate-level aggregation (18 debates, OLS HC1)"); print("="*70)
Xc=["aggr","infl","disc","pred","prer","Imm","FP","Ab"]
for y,lab in [("cd","D Democratic"),("cr","D Republican")]:
    r,n,k=ols_hc1(y,Xc)
    print(f"\nDV = {lab}  (N={n}, params={k})")
    for v,vlab in [("aggr","Aggressive_Words"),("infl","Inflammatory_Words"),("disc","Discriminatory_Words")]:
        print(f"   {vlab:20s} {r[v][0]:+.4f}  (p={r[v][1]:.3f}){star(r[v][1])}")
