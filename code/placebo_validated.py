# Placebo and post-minus-placebo difference tests on the VALIDATED supervised measure.
# Reproduces Table 8 (two-panel placebo) and Table 9 (formal difference test).
# Author(s) withheld for peer review.
#
# Placebo logic: regress the PRE-debate poll-to-poll change (d_democrats/d_republicans)
# on debate-day rhetoric. Debate-day rhetoric cannot have moved earlier polls, so a
# credible placebo yields nulls. Each rhetorical category is estimated separately
# (Panel A pooled, Panel B source-specific), exactly parallel to the substantive
# analysis, and every coefficient is judged by a wild cluster bootstrap because the
# placebo rests on only ~12 debate clusters (asymptotic SEs are unreliable there).
#
# Requires: numpy, pandas.   Reads: data/poll_final_data_with_change.dta,
#                                    data/supervised_counts_2004_2024.csv
import pandas as pd, numpy as np, math

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

# ---- load + merge validated counts onto polling data ----
dta=pd.read_stata('data/poll_final_data_with_change.dta')
sc=pd.read_csv('data/supervised_counts_2004_2024.csv')
w=sc.pivot(index='date',columns='party',values=['aggr_sup','infl_sup','disc_sup'])
w.columns=[f"{a}_{'d' if b=='Dem' else 'r'}" for a,b in w.columns]; w=w.reset_index()
w['aggr_tot']=w['aggr_sup_d']+w['aggr_sup_r']
w['infl_tot']=w['infl_sup_d']+w['infl_sup_r']
w['disc_tot']=w['disc_sup_d']+w['disc_sup_r']
w=w.rename(columns={'aggr_sup_d':'aggr_d','aggr_sup_r':'aggr_r','infl_sup_d':'infl_d',
                    'infl_sup_r':'infl_r','disc_sup_d':'disc_d','disc_sup_r':'disc_r'})
dta['date']=dta['debate_date'].astype(str).str[:10]
m=dta.merge(w,on='date',how='inner'); m['dk']=m['date']
yr=pd.get_dummies(m['date'].str[:4],prefix='yr',drop_first=True).astype(float)
for c in yr.columns: m[c]=yr[c].values
yc=list(yr.columns); ctrl=['pre_democrats','pre_republicans','Immigration','ForeignPolicy','AbortionRights']

def design(frame,y,xv):
    sub=frame.dropna(subset=[y,'sample1','dk']+xv+ctrl).copy()
    X=sub[xv+ctrl+yc].astype(float).copy(); X.insert(0,'const',1.0); cols=list(X.columns); Xn=X.values
    keep=[]
    for j in range(Xn.shape[1]):
        if np.linalg.matrix_rank(Xn[:,keep+[j]])==len(keep)+1: keep.append(j)
    return sub,Xn[:,keep],[cols[j] for j in keep],sub[y].astype(float).values,sub['sample1'].astype(float).values,sub['dk'].values

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

# ===================== TABLE 8: two-panel placebo =====================
print("="*74); print("TABLE 8  Placebo: pre-debate poll change on debate-day rhetoric (validated)")
print("="*74)
for y,lab in [('d_democrats','(1) D Pre Democratic'),('d_republicans','(2) D Pre Republican')]:
    _,_,_,_,_,g=design(m,y,['aggr_tot']); print(f"\nDV = {lab}   (N={len(_ )}, clusters={len(np.unique(g))})")
    print(" Panel A - pooled (each category separately):")
    for cat in ['aggr','infl','disc']:
        b,pa,pw,G,n=wild(m,y,[cat+'_tot'],cat+'_tot')
        print(f"   {cat}_tot  {b:+.4f}  (wild p={pw:.3f}){star(pw)}")
    print(" Panel B - source-specific (each category separately):")
    for cat in ['aggr','infl','disc']:
        for suf in ['_d','_r']:
            b,pa,pw,G,n=wild(m,y,[cat+'_d',cat+'_r'],cat+suf)
            print(f"   {cat}{suf}  {b:+.4f}  (wild p={pw:.3f}){star(pw)}")

# ===================== TABLE 9: post - placebo difference =====================
print("\n"+"="*74); print("TABLE 9  Post - placebo difference (D Republican), validated, wild bootstrap")
print("="*74)
base=m.dropna(subset=['change_republicans','d_republicans','sample1']+ctrl+
              ['aggr_d','aggr_r','infl_d','infl_r','disc_d','disc_r']).copy()
def difftest(cat,target,B=4999,seed=5):
    rows=[]
    for post,dv in [(0,'d_republicans'),(1,'change_republicans')]:
        s=base.copy(); s['POST']=post; s['DV']=s[dv]; rows.append(s)
    L=pd.concat(rows,ignore_index=True)
    yrs=pd.get_dummies(L['date'].str[:4],prefix='yr',drop_first=True).astype(float)
    for c in yrs.columns: L[c]=yrs[c].values
    L['cd_post']=L[cat+'_d']*L['POST']; L['cr_post']=L[cat+'_r']*L['POST']
    L['dk']=L['date']
    xv=[cat+'_d',cat+'_r','cd_post','cr_post','POST']
    sub,Xn,cols,yv,wt,g=design(L,'DV',xv)  # design adds ctrl+yc; POST/interactions in xv
    j=cols.index(target); return wild(L,'DV',xv,target,B=B,seed=seed)
for cat in ['aggr','infl','disc']:
    for suf,nm in [('cd_post',cat+'_d x POST'),('cr_post',cat+'_r x POST')]:
        b,pa,pw,G,n=difftest(cat,suf)
        print(f"   {nm:18s} diff={b:+.4f}  (wild p={pw:.3f}){star(pw)}")
