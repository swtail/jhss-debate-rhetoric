# RQ2 (reframed) sensitivity & identifiability analysis.
# Author(s) withheld for peer review.
#
# (A) MULTIVERSE / SPECIFICATION CURVE. The substantive "asymmetric civility penalty"
#     estimand is the own-party, own-category source-specific coefficient (e.g.,
#     Republican-spoken inflammatory rhetoric on the Republican-share outcome). We
#     re-estimate it across every combination of:
#       classifier        : keyword lexicon, context-aware lexicon, supervised (validated)
#       outcome            : post-debate CHANGE vs post-debate LEVEL
#       sample             : 2004-2024 (full) vs 2008-2020 (sentence-transcript core)
#       weighting          : poll-sample-size WLS vs unweighted OLS
#       topic controls     : included vs excluded
#     (election-year FE and debate clustering held fixed). We then report, per category,
#     how often the sign and the significance of the estimand flip across the multiverse.
#
# (B) MINIMUM DETECTABLE EFFECT. With 18 debate clusters the design's precision is fixed
#     by the cluster-robust SE. We report MDE = (t_{.975,17}+t_{.80,17}) * SE for each
#     source-specific coefficient (two-sided 5%, 80% power), in per-sentence units and as
#     a per-debate cumulative association (MDE * mean sentences/ debate, Table 4), and
#     compare to the observed coefficients.
#
# Requires: numpy, pandas.  Reads: data/corpus_*.csv, data/panel_2004_2024_validated.csv,
#   data/supervised_counts_2004_2024.csv
import os, math, itertools
import numpy as np, pandas as pd
D=os.path.join(os.path.dirname(__file__),"..","data")

# ---------- t p-values (regularized incomplete beta) ----------
def betacf(a,b,x):
    EPS=3e-12;FP=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1.;d=1-qab*x/qap;d=FP if abs(d)<FP else d;d=1/d;h=d
    for m in range(1,300):
        m2=2*m;aa=m*(b-m)*x/((qam+m2)*(a+m2));d=1+aa*d;d=FP if abs(d)<FP else d;c=1+aa/c;c=FP if abs(c)<FP else c;d=1/d;h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));d=1+aa*d;d=FP if abs(d)<FP else d;c=1+aa/c;c=FP if abs(c)<FP else c;d=1/d;de=d*c;h*=de
        if abs(de-1)<EPS:break
    return h
def betai(a,b,x):
    if x<=0:return 0.
    if x>=1:return 1.
    lb=math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x);bt=math.exp(lb)
    return bt*betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-bt*betacf(b,a,1-x)/b
def tp(t,df):df=max(int(df),1);return betai(df/2.,.5,df/(df+t*t))
def tcrit(p,df):  # two-sided critical t via bisection on tp
    lo,hi=0.,100.
    for _ in range(200):
        mid=(lo+hi)/2
        if tp(mid,df)>p: lo=mid
        else: hi=mid
    return mid
def tcrit_onesided(p,df):  # t with upper-tail prob = p  -> two-sided tp=2p
    return tcrit(2*p,df)

# ---------- lexicons (from classifier_robustness.py) ----------
AGG=['fight','fighting','attack','beat','destroy','defeat','wrong','lie','lying','liar','failed','fail','incompetent','voted against','you said',"doesn't mention","don't put words"]
INF=['disaster','disgrace','horrible','horribly','terrible','radical','dangerous','hatred','destroying','catastrophe','stupid','tragedy','heartbreak','enemy','corrupt','extreme','outrage','evil','crisis','disgraceful','heartbreaking']
GROUP=['immigrant','immigrants','mexican','mexicans','muslim','muslims','islam ','islamic','black people','african american','latino','latinos','hispanic','women','jewish','jews','gay ','lesbian','transgender']
DEROG=['animals','thugs','rapists','criminals','invasion','infest','vermin',"don't belong",'do not belong','go back to your country','take their place','these people are','those people are']
CONDEMN=['racist','xenophob','calling','he calls','she calls','called them','called us','said that','because it','denounce','condemn']
FOREIGN=['north korea','putin','xi ','jinping','kim ','jong','dictator','iran','tehran','china president','chinese president']
NEG=['not','no','never','hardly',"n't"]
def has(t,lex):return any(w in t for w in lex)
def keyword(s):
    t=' '+str(s).lower()+' ';return (1 if has(t,AGG) else 0,1 if has(t,INF) else 0,1 if (has(t,GROUP) or has(t,DEROG)) else 0)
def context(s):
    t=' '+str(s).lower()+' ';agg=1 if has(t,AGG) else 0;inf=0
    for w in INF:
        if w in t:
            i=t.find(w);pre=t[max(0,i-22):i]
            if any(n+' ' in pre for n in NEG):continue
            inf=1;break
    dis=1 if (has(t,GROUP) and has(t,DEROG)) and not has(t,CONDEMN) and not has(t,FOREIGN) else 0
    return agg,inf,dis

files=[os.path.join(D,f) for f in ["corpus_2004.csv","corpus_sentences_2008_2020.csv","corpus_2024.csv"]]
corp=pd.concat([pd.read_csv(f).rename(columns=str.lower)[['date','party','sentence']] for f in files],ignore_index=True)
corp['date']=corp['date'].astype(str).str[:10]
def counts_for(fn):
    lab=corp['sentence'].apply(fn)
    tmp=corp.copy()
    for k,suf in enumerate(['a','i','d']): tmp[suf]=[x[k] for x in lab]
    g=tmp.groupby(['date','party'])[['a','i','d']].sum().reset_index()
    w=g.pivot(index='date',columns='party',values=['a','i','d']).reindex(
        columns=pd.MultiIndex.from_product([['a','i','d'],['Dem','Rep']]),fill_value=0)
    w.columns=[f"{a}_{'d' if b=='Dem' else 'r'}" for a,b in w.columns];w=w.reset_index()
    return w.rename(columns={'a_d':'aggr_d','a_r':'aggr_r','i_d':'infl_d','i_r':'infl_r','d_d':'disc_d','d_r':'disc_r'})
KW=counts_for(keyword);CTX=counts_for(context)
sup=pd.read_csv(os.path.join(D,"supervised_counts_2004_2024.csv"))
sw=sup.pivot(index='date',columns='party',values=['aggr_sup','infl_sup','disc_sup'])
sw.columns=[f"{a.split('_')[0]}_{'d' if b=='Dem' else 'r'}" for a,b in sw.columns];SUP=sw.reset_index()

pan=pd.read_csv(os.path.join(D,"panel_2004_2024_validated.csv"))
pan['date']=pan['debate_date'].astype(str).str[:10]
pan['level_dem']=pan['pre_democrats']+pan['change_democrats']
pan['level_rep']=pan['pre_republicans']+pan['change_republicans']
pan['yr4']=pan['date'].str[:4]

def cluster_fit(sub,y,xv,weighted):
    sub=sub.dropna(subset=[y,'dk']+xv).copy()
    if weighted: sub=sub.dropna(subset=['sample1'])
    X=sub[xv].astype(float).copy();X.insert(0,'const',1.);cols=list(X.columns);Xn=X.values
    keep=[]
    for j in range(Xn.shape[1]):
        if np.linalg.matrix_rank(Xn[:,keep+[j]])==len(keep)+1:keep.append(j)
    Xn=Xn[:,keep];cols=[cols[j] for j in keep]
    yv=sub[y].astype(float).values
    w=sub['sample1'].astype(float).values if weighted else np.ones(len(sub))
    W=np.sqrt(w);Xw=Xn*W[:,None];inv=np.linalg.inv(Xw.T@Xw);b=inv@(Xw.T@(yv*W));res=yv-Xn@b
    g=sub['dk'].values;cl=np.unique(g);G=len(cl);n,k=Xn.shape;adj=(G/(G-1))*((n-1)/(n-k));meat=np.zeros((k,k))
    for c in cl:
        idx=g==c;s=Xw[idx].T@((res*W)[idx]);meat+=np.outer(s,s)
    V=adj*inv@meat@inv;se=np.sqrt(np.clip(np.diag(V),0,None))
    return cols,b,se,G

CLF={'keyword':KW,'context':CTX,'supervised':SUP}
ctrl_topic=['Immigration','ForeignPolicy','AbortionRights']
yrdum=lambda s: pd.get_dummies(s['yr4'],prefix='yr',drop_first=True).astype(float)

# target: own-party own-category coefficient
TARGETS=[('infl','infl_r','rep','Rep inflammatory -> Rep'),
         ('infl','infl_d','dem','Dem inflammatory -> Dem'),
         ('disc','disc_r','rep','Rep discriminatory -> Rep'),
         ('disc','disc_d','dem','Dem discriminatory -> Dem')]

rows=[]
for clf,cdf in CLF.items():
    base=pan.drop(columns=[c for c in ['aggr_d','aggr_r','infl_d','infl_r','disc_d','disc_r'] if c in pan.columns]).merge(cdf,on='date',how='left')
    for outcome in ['change','level']:
        for sample in ['2004-2024','2008-2020']:
            for weighted in [True,False]:
                for topic in [True,False]:
                    sub=base.copy()
                    if sample=='2008-2020': sub=sub[(sub['yr4'].astype(int)>=2008)&(sub['yr4'].astype(int)<=2020)]
                    yd=yrdum(sub)
                    for c in yd.columns: sub[c]=yd[c].values
                    yc=list(yd.columns)
                    for cat,tgt,side,lab in TARGETS:
                        dv=('change_'+('republicans' if side=='rep' else 'democrats')) if outcome=='change' else ('level_'+('rep' if side=='rep' else 'dem'))
                        pre=['pre_republicans','pre_democrats'] if outcome=='change' else (['pre_republicans'] if side=='rep' else ['pre_democrats'])
                        xv=[cat+'_d',cat+'_r']+pre+(ctrl_topic if topic else [])+yc
                        try:
                            cols,b,se,G=cluster_fit(sub,dv,xv,weighted)
                        except Exception: continue
                        if tgt not in cols: continue
                        j=cols.index(tgt);coef=b[j];p=tp(coef/se[j],G-1)
                        rows.append(dict(clf=clf,outcome=outcome,sample=sample,weighted=weighted,topic=topic,
                                         cat=cat,target=tgt,label=lab,coef=coef,p=p,G=G))
M=pd.DataFrame(rows)
print("="*78);print("(A) MULTIVERSE / SPECIFICATION CURVE");print("="*78)
print(f"Total specifications fit: {len(M)}  ({M['clf'].nunique()} classifiers x outcome x sample x weight x topic x {len(TARGETS)} targets)")
for lab in [t[3] for t in TARGETS]:
    d=M[M['label']==lab]
    if len(d)==0: continue
    pos=(d['coef']>0).mean();neg=(d['coef']<0).mean()
    sig=(d['p']<0.05).mean();sig10=(d['p']<0.10).mean()
    signflip = min(pos,neg)>0
    print(f"\n{lab}  (N spec={len(d)})")
    print(f"   coef sign: {pos*100:.0f}% positive / {neg*100:.0f}% negative   -> sign {'FLIPS' if signflip else 'stable'}")
    print(f"   significant: {sig*100:.0f}% at 5%, {sig10*100:.0f}% at 10%")
    print(f"   coef range: [{d['coef'].min():+.4f}, {d['coef'].max():+.4f}]  median {d['coef'].median():+.4f}")

print("\n"+"="*78);print("(B) MINIMUM DETECTABLE EFFECT @ 18 debate clusters (validated, change, WLS, topic)");print("="*78)
# preferred spec: supervised, change, 2004-2024, weighted, topic controls
base=pan.copy(); yd=yrdum(base)
for c in yd.columns: base[c]=yd[c].values
yc=list(yd.columns)
mult=tcrit_onesided(0.025,17)+tcrit_onesided(0.20,17)  # (t.975 + t.80) df=17
print(f"  MDE multiplier (t_.975,17 + t_.80,17) = {mult:.2f}")
prev=pan.drop_duplicates('dk')[['infl_d','infl_r','disc_d','disc_r']].mean()
for cat,tgt,side,lab in TARGETS:
    dv='change_'+('republicans' if side=='rep' else 'democrats')
    xv=[cat+'_d',cat+'_r','pre_republicans','pre_democrats']+ctrl_topic+yc
    cols,b,se,G=cluster_fit(base,dv,xv,True);j=cols.index(tgt)
    mde=mult*se[j]; meanN=prev[tgt]
    print(f"  {lab:26s} coef={b[j]:+.4f}  SE={se[j]:.4f}  MDE/sentence={mde:.4f}  "
          f"(per-debate MDE={mde*meanN:+.2f} pts at mean {meanN:.0f} sentences); |coef|<MDE: {abs(b[j])<mde}")
