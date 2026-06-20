# Authors: Bryan Kyung and Katie Kim
"""
JHSS R2 master re-analysis. Runs on R1/Data_Files/poll_final_data_with_change.dta
(production BERT counts; variable names match the manuscript). All numbers reported
in the revised manuscript and response come from this script. No scipy dependency.
"""
import sys, os, math, json
import numpy as np, pandas as pd, numpy.linalg as la

# ---------- Student-t two-sided p-value (no scipy) ----------
def _betacf(a,b,x):
    FPMIN=1e-300
    qab=a+b; qap=a+1; qam=a-1; c=1.0; d=1-qab*x/qap
    d=FPMIN if abs(d)<FPMIN else d; d=1/d; h=d
    for m in range(1,300):
        m2=2*m
        aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; d=FPMIN if abs(d)<FPMIN else d
        c=1+aa/c; c=FPMIN if abs(c)<FPMIN else c; d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=FPMIN if abs(d)<FPMIN else d
        c=1+aa/c; c=FPMIN if abs(c)<FPMIN else c; d=1/d; de=d*c; h*=de
        if abs(de-1)<3e-12: break
    return h
def betainc(a,b,x):
    if x<=0: return 0.0
    if x>=1: return 1.0
    lb=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    f=math.exp(math.log(x)*a+math.log(1-x)*b-lb)
    return f/a*_betacf(a,b,x) if x<(a+1)/(a+b+2) else 1-f/b*_betacf(b,a,1-x)
def t_p(t,df):
    t=abs(t); return betainc(df/2,0.5,df/(df+t*t))
def star(p): return '***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''

DATA=os.path.join(os.path.dirname(__file__),'..','..','R1','Data_Files','poll_final_data_with_change.dta')
df=pd.read_stata(DATA)
df=df.dropna(subset=['change_democrats','change_republicans','pre_democrats','pre_republicans']).copy()
df['year']=df['year'].astype(int)
TOPIC=['Immigration','ForeignPolicy','AbortionRights']

def reg(d,y,xv,cl='debate_date',fe='year',w='sample1',extra_fe_ok=True):
    d=d.dropna(subset=[y]+xv+[w,cl]).copy()
    X=d[xv].astype(float).values
    if fe and fe in d:
        yrs=sorted(d[fe].unique())[1:]
        fed=np.column_stack([(d[fe]==yr).astype(float).values for yr in yrs]) if yrs else np.empty((len(d),0))
        fenames=[f'{fe}_{int(yr)}' for yr in yrs]
    else:
        fed=np.empty((len(d),0)); fenames=[]
    Xm=np.column_stack([np.ones(len(d)),X,fed]); names=['const']+xv+fenames
    wt=d[w].astype(float).values; W=wt/wt.mean(); sw=np.sqrt(W)
    Y=d[y].astype(float).values; Xw=Xm*sw[:,None]; Yw=Y*sw
    keep=np.ones(Xm.shape[1],bool)
    if la.matrix_rank(Xw)<Xm.shape[1]:
        cur=[]
        for j in range(Xm.shape[1]):
            if la.matrix_rank(Xw[:,cur+[j]])==len(cur)+1: cur.append(j)
        keep=np.zeros(Xm.shape[1],bool); keep[cur]=True
    Xw2=Xw[:,keep]; names2=[n for n,k in zip(names,keep) if k]
    beta=la.lstsq(Xw2,Yw,rcond=None)[0]; res=Yw-Xw2@beta
    bread=la.inv(Xw2.T@Xw2); clv=d[cl].values; meat=np.zeros((Xw2.shape[1],)*2)
    for c in np.unique(clv):
        idx=clv==c; sc=Xw2[idx].T@res[idx]; meat+=np.outer(sc,sc)
    G=len(np.unique(clv)); n=len(d); k=Xw2.shape[1]; adj=(G/(G-1))*((n-1)/(n-k))
    V=bread@meat@bread*adj; se=np.sqrt(np.diag(V)); t=beta/se
    p=np.array([t_p(tt,G-1) for tt in t])
    adjr2=1-(1-(1-np.sum(res**2)/np.sum((Yw-Yw.mean())**2)))*(n-1)/(n-k)
    return ({nm:(b,s,pp) for nm,b,s,pp in zip(names2,beta,se,p)},
            dict(N=n,G=G,adjr2=adjr2,drop=[nm for nm,kk in zip(names,keep) if not kk]))

def block(title): print("\n"+"="*90+"\n"+title+"\n"+"="*90)
def sh(tag,o,i,vs):
    print(f"\n  {tag}  N={i['N']} clusters={i['G']} adjR2={i['adjr2']:.3f}")
    if i['drop']: print("    dropped (collinear):",", ".join(i['drop']))
    for v in vs:
        if v in o: b,s,p=o[v]; print(f"    {v:26s} {b:+.4f}{star(p):3s} (p={p:.3f})")
        else: print(f"    {v:26s} [dropped — collinear]")

print("JHSS R2 RE-ANALYSIS  | dataset: poll_final_data_with_change.dta")
print(f"Estimation N={len(df)} | debate clusters={df.debate_date.nunique()} | years={sorted(df.year.unique())}")
print("NOTE: incumbency, GDP, per-capita GDP take one value per election year -> absorbed by year FE.")

block("PREVALENCE BY PARTY (Comment 15)")
cd=df.drop_duplicates(['debate_date','candidate'])
for lab,sub in [('Democratic',cd[cd.democratdum==1]),('Republican',cd[cd.republicandum==1])]:
    print(f"  {lab} candidate-debates n={len(sub)}")
    for c in ['aggressive_words','inflammatory_words','discriminatory_words']:
        print(f"     {c:22s} mean={sub[c].mean():.2f}  total={int(sub[c].sum())}  share>0={(sub[c]>0).mean()*100:.0f}%")

main=['aggressive_words','inflammatory_words','discriminatory_words','pre_democrats','pre_republicans']+TOPIC
src =['aggressive_words_d','aggressive_words_r','inflammatory_words_d','inflammatory_words_r',
      'discriminatory_words_d','discriminatory_words_r','pre_democrats','pre_republicans']+TOPIC

block("TABLE 2 (REAL) Baseline, full controls (year FE + pre-shares + topics), WLS, debate-clustered SE")
o,i=reg(df,'change_democrats',main);   sh("DV=Change_Democrats",o,i,main[:3])
o,i=reg(df,'change_republicans',main); sh("DV=Change_Republicans",o,i,main[:3])

block("TABLE 3 (REAL) Source-specific, full controls")
o,i=reg(df,'change_democrats',src);   sh("DV=Change_Democrats",o,i,src[:6])
o,i=reg(df,'change_republicans',src); sh("DV=Change_Republicans",o,i,src[:6])

block("TABLE 4 (REAL) Sample-size heterogeneity (rhetoric x High_Sample)")
d2=df.copy()
inter=[]
for c in ['aggressive_words_d','aggressive_words_r','inflammatory_words_d','inflammatory_words_r','discriminatory_words_d','discriminatory_words_r']:
    d2[c+'_xhs']=d2[c]*d2['high_sample1']; inter.append(c+'_xhs')
xv4=['aggressive_words_d','aggressive_words_r','inflammatory_words_d','inflammatory_words_r','discriminatory_words_d','discriminatory_words_r','high_sample1']+inter+['pre_democrats','pre_republicans']+TOPIC
o,i=reg(d2,'change_democrats',xv4);   sh("DV=Change_Democrats",o,i,['inflammatory_words_r_xhs','discriminatory_words_d_xhs','discriminatory_words_r_xhs'])
o,i=reg(d2,'change_republicans',xv4); sh("DV=Change_Republicans",o,i,['inflammatory_words_r_xhs','discriminatory_words_d_xhs','discriminatory_words_r_xhs'])

block("TABLE 5 (REAL) Polarization amplification (Comments 13,14) -- interactions vs year FE")
d3=df.copy(); d3['highpol']=d3.year.isin([2016,2020]).astype(float)
pol=[]
for c in ['inflammatory_words_d','inflammatory_words_r','discriminatory_words_d','discriminatory_words_r','aggressive_words_d','aggressive_words_r']:
    d3[c+'_xhp']=d3[c]*d3['highpol']; pol.append(c+'_xhp')
xv5=['inflammatory_words_d','inflammatory_words_r','discriminatory_words_d','discriminatory_words_r','aggressive_words_d','aggressive_words_r']+pol+['pre_democrats','pre_republicans']+TOPIC
o,i=reg(d3,'change_democrats',xv5); sh("DV=Change_Democrats",o,i,pol)

block("DEBATE-LEVEL AGGREGATION ~ (Comment 8): collapse to candidate-debate, OLS HC1")
agg=df.groupby(['debate_date','candidate']).agg({'change_democrats':'mean','change_republicans':'mean',
     'aggressive_words':'first','inflammatory_words':'first','discriminatory_words':'first',
     'inflammatory_words_d':'first','inflammatory_words_r':'first','year':'first'}).reset_index()
print(f"  N candidate-debate units = {len(agg)}")
def ols_hc1(d,y,xv):
    d=d.dropna(subset=[y]+xv); X=np.column_stack([np.ones(len(d))]+[d[v].astype(float).values for v in xv])
    Y=d[y].astype(float).values; b=la.lstsq(X,Y,rcond=None)[0]; res=Y-X@b
    n,k=X.shape; bread=la.inv(X.T@X); meat=(X*res[:,None]).T@(X*res[:,None])*(n/(n-k))
    V=bread@meat@bread; se=np.sqrt(np.diag(V)); t=b/se; p=np.array([t_p(tt,n-k) for tt in t])
    return {nm:(bb,ss,pp) for nm,bb,ss,pp in zip(['const']+xv,b,se,p)}
o=ols_hc1(agg,'change_republicans',['aggressive_words','inflammatory_words','discriminatory_words'])
print("  DV=Change_Republicans (unsigned):")
for v in ['aggressive_words','inflammatory_words','discriminatory_words']:
    b,s,p=o[v]; print(f"    {v:22s} {b:+.4f}{star(p):3s} (p={p:.3f})")

block("PLACEBO + FORMAL DIFFERENCE TEST (Comment 9): stack pre & post, interact x POST")
# post change vs pre-debate change (d_democrats/d_republicans are pre-debate poll-to-poll)
recs=[]
for _,r in df.iterrows():
    base=dict(debate_date=r.debate_date,sample1=r.sample1,pre_democrats=r.pre_democrats,pre_republicans=r.pre_republicans,
              inflammatory_words_r=r.inflammatory_words_r,discriminatory_words_r=r.discriminatory_words_r,
              Immigration=r.Immigration,ForeignPolicy=r.ForeignPolicy,AbortionRights=r.AbortionRights,year=r.year)
    recs.append({**base,'y':r.change_republicans,'POST':1})
    recs.append({**base,'y':r.d_republicans,'POST':0})
st=pd.DataFrame(recs).dropna(subset=['y'])
for c in ['inflammatory_words_r','discriminatory_words_r']:
    st[c+'_xpost']=st[c]*st['POST']
xv=['inflammatory_words_r','discriminatory_words_r','inflammatory_words_r_xpost','discriminatory_words_r_xpost','POST','pre_democrats','pre_republicans']+TOPIC
o,i=reg(st,'y',xv); 
print("  Difference = post-debate minus placebo coefficient:")
for v in ['inflammatory_words_r_xpost','discriminatory_words_r_xpost']:
    if v in o: b,s,p=o[v]; print(f"    {v:30s} {b:+.4f}{star(p):3s} (p={p:.3f})")

block("MEASUREMENT-ERROR SENSITIVITY (Comment 4): 15% random label noise, 400 MC reps")
rng=np.random.default_rng(7)
base_o,_=reg(df,'change_republicans',src); b0=base_o['inflammatory_words_r'][0]
ests=[]
for _ in range(400):
    dd=df.copy()
    noise=rng.binomial(1,0.15,len(dd))*rng.integers(-1,2,len(dd))
    dd['inflammatory_words_r']=np.clip(dd['inflammatory_words_r']+noise,0,None)
    try:
        oo,_=reg(dd,'change_republicans',src); ests.append(oo['inflammatory_words_r'][0])
    except Exception: pass
ests=np.array(ests)
print(f"  baseline inflammatory_words_r = {b0:+.3f}")
print(f"  under noise: mean={ests.mean():+.3f} sd={ests.std():.3f} 5-95pct=[{np.percentile(ests,5):+.3f},{np.percentile(ests,95):+.3f}] same-sign={(np.sign(ests)==np.sign(b0)).mean()*100:.1f}%")

block("SPECIFICATION SENSITIVITY (the honest robustness point): inflammatory_words_d on Change_Democrats")
for lab,ctrl,fe in [("year FE + pre-shares",['pre_democrats','pre_republicans'],'year'),
                    ("+ topic controls (main spec)",['pre_democrats','pre_republicans']+TOPIC,'year'),
                    ("no year FE, topics+macro",['pre_democrats','pre_republicans']+TOPIC+['gdpbillionsofus','percapitaus','incumbent_democrat'],None)]:
    dd=df.copy()
    o,i=reg(dd,'change_democrats',['aggressive_words_d','aggressive_words_r','inflammatory_words_d','inflammatory_words_r','discriminatory_words_d','discriminatory_words_r']+ctrl,fe=fe)
    b,s,p=o['inflammatory_words_d']; print(f"  {lab:32s} inflam_d = {b:+.3f}{star(p):3s} (p={p:.3f})")

block("VIF (Comment 13): year-level macro vars vs year structure")
def vif(d,xv):
    d=d.dropna(subset=xv); out={}
    for j,v in enumerate(xv):
        y=d[v].astype(float).values; X=np.column_stack([np.ones(len(d))]+[d[c].astype(float).values for c in xv if c!=v])
        b=la.lstsq(X,y,rcond=None)[0]; r=y-X@b; r2=1-np.sum(r**2)/np.sum((y-y.mean())**2)
        out[v]=1/(1-r2) if r2<1 else np.inf
    return out
vv=vif(df,['inflammatory_words_d','inflammatory_words_r','discriminatory_words_d','discriminatory_words_r','incumbent_democrat','gdpbillionsofus','percapitaus','pre_democrats','pre_republicans'])
for k,val in vv.items(): print(f"  VIF {k:24s} = {val:9.1f}")
print("\nDONE.")
