# Classifier-robustness table (Table 2): own-party, own-category source-specific
# coefficient under three measurement instruments (keyword lexicon, context-aware
# lexicon, supervised/validated), on the full 2004-2024 panel.
# Author(s) withheld for peer review.
#
# Motivation for RQ1: the substantive estimate should not hinge on which classifier
# measures the rhetoric. Holding the regression fixed and swapping only the measure
# shows the sign and magnitude move sharply across instruments, and the context-aware
# lexicon detects almost no discriminatory content (its Republican discriminatory
# coefficient is not estimable). Only the supervised measure validates (Table 3) and
# is carried to all RQ2 tests.
#
# Requires: numpy, pandas.
# Reads: data/corpus_2004.csv, data/corpus_sentences_2008_2020.csv, data/corpus_2024.csv,
#        data/panel_2004_2024_validated.csv
import pandas as pd, numpy as np, math

B="data/"
files=[B+"corpus_2004.csv",B+"corpus_sentences_2008_2020.csv",B+"corpus_2024.csv"]
corp=pd.concat([pd.read_csv(f).rename(columns=str.lower)[['date','party','sentence']] for f in files],ignore_index=True)
corp['date']=corp['date'].astype(str).str[:10]

# ---- transparent lexicons (rules from the original context-aware pipeline) ----
AGG=['fight','fighting','attack','beat','destroy','defeat','wrong','lie','lying','liar','failed','fail','incompetent','voted against','you said',"doesn't mention","don't put words"]
INF=['disaster','disgrace','horrible','horribly','terrible','radical','dangerous','hatred','destroying','catastrophe','stupid','tragedy','heartbreak','enemy','corrupt','extreme','outrage','evil','crisis','disgraceful','heartbreaking']
GROUP=['immigrant','immigrants','mexican','mexicans','muslim','muslims','islam ','islamic','black people','african american','latino','latinos','hispanic','women','jewish','jews','gay ','lesbian','transgender']
DEROG=['animals','thugs','rapists','criminals','invasion','infest','vermin',"don't belong",'do not belong','go back to your country','take their place','these people are','those people are']
CONDEMN=['racist','xenophob','calling','he calls','she calls','called them','called us','said that','because it','denounce','condemn']
FOREIGN=['north korea','putin','xi ','jinping','kim ','jong','dictator','iran','tehran','china president','chinese president']
NEG=['not','no','never','hardly',"n't"]
def has(t,lex): return any(w in t for w in lex)
def keyword(s):
    t=' '+str(s).lower()+' '
    return (1 if has(t,AGG) else 0, 1 if has(t,INF) else 0, 1 if (has(t,GROUP) or has(t,DEROG)) else 0)
def context(s):
    t=' '+str(s).lower()+' '; agg=1 if has(t,AGG) else 0; inf=0
    for w in INF:
        if w in t:
            i=t.find(w); pre=t[max(0,i-22):i]
            if any(n+' ' in pre for n in NEG): continue
            inf=1; break
    dis=1 if (has(t,GROUP) and has(t,DEROG)) and not has(t,CONDEMN) and not has(t,FOREIGN) else 0
    return agg,inf,dis
for nm,fn in [('kw',keyword),('ctx',context)]:
    lab=corp['sentence'].apply(fn)
    for k,suf in enumerate(['a','i','d']): corp[f'{nm}_{suf}']=[x[k] for x in lab]
print("context-aware discriminatory flags in corpus:",int(corp['ctx_d'].sum()),
      "(Republican:",int(corp.loc[corp.party=='Rep','ctx_d'].sum()),")")

def counts(nm):
    g=corp.groupby(['date','party'])[[f'{nm}_a',f'{nm}_i',f'{nm}_d']].sum().reset_index()
    w=g.pivot(index='date',columns='party',values=[f'{nm}_a',f'{nm}_i',f'{nm}_d']).reindex(
        columns=pd.MultiIndex.from_product([[f'{nm}_a',f'{nm}_i',f'{nm}_d'],['Dem','Rep']]),fill_value=0)
    w.columns=[f"{a.split('_')[1]}_{'d' if b=='Dem' else 'r'}" for a,b in w.columns]; w=w.reset_index()
    w.columns=['date']+[{'a_d':'aggr_d','a_r':'aggr_r','i_d':'infl_d','i_r':'infl_r','d_d':'disc_d','d_r':'disc_r'}[c] for c in w.columns[1:]]
    return w
KW=counts('kw'); CTX=counts('ctx')

pan=pd.read_csv(B+"panel_2004_2024_validated.csv"); pan['date']=pan['debate_date'].astype(str).str[:10]
yc=[c for c in pan.columns if c.startswith('yr_')]; ctrl=['pre_democrats','pre_republicans','Immigration','ForeignPolicy','AbortionRights']

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
def tp(t,df): df=max(int(df),1); return betai(df/2.,.5,df/(df+t*t))
def star(p): return '*'*sum(p<th for th in (.1,.05,.01))
def wls(m,y,cat,target):
    xv=[cat+'_d',cat+'_r']; sub=m.dropna(subset=[y,'sample1','dk']+xv+ctrl).copy()
    X=sub[xv+ctrl+yc].astype(float).copy(); X.insert(0,'const',1.); cols=list(X.columns); Xn=X.values
    keep=[]
    for j in range(Xn.shape[1]):
        if np.linalg.matrix_rank(Xn[:,keep+[j]])==len(keep)+1: keep.append(j)
    Xn=Xn[:,keep]; cols=[cols[j] for j in keep]
    if target not in cols: return None
    yv=sub[y].astype(float).values; wt=sub['sample1'].astype(float).values; W=np.sqrt(wt)
    Xw=Xn*W[:,None]; inv=np.linalg.inv(Xw.T@Xw); b=inv@(Xw.T@(yv*W)); res=yv-Xn@b
    g=sub['dk'].values; G=len(np.unique(g)); n,k=Xn.shape; adj=(G/(G-1))*((n-1)/(n-k)); meat=np.zeros((k,k))
    for cl in np.unique(g):
        idx=g==cl; s=Xw[idx].T@((res*W)[idx]); meat+=np.outer(s,s)
    V=adj*inv@meat@inv; j=cols.index(target); se=math.sqrt(max(V[j,j],1e-300))
    return b[j], tp(b[j]/se,G-1)
def fmt(r): return "  n/e" if r is None else f"{r[0]:+.3f}{star(r[1])}"
def colvals(cdf):
    m=pan.drop(columns=['aggr_d','aggr_r','infl_d','infl_r','disc_d','disc_r']).merge(cdf,on='date',how='left')
    out={}
    for cat in ['aggr','infl','disc']:
        out[('Rep',cat)]=fmt(wls(m,'change_republicans',cat,cat+'_r'))
        out[('Dem',cat)]=fmt(wls(m,'change_democrats',cat,cat+'_d'))
    return out
sup=pan[['date','aggr_d','aggr_r','infl_d','infl_r','disc_d','disc_r']].drop_duplicates('date')
cols={'Keyword':colvals(KW),'Context-aware':colvals(CTX),'Supervised':colvals(sup)}
print(f"\n{'Relationship':38s}{'Keyword':>11s}{'Context':>11s}{'Supervised':>12s}")
for cat,nm in [('aggr','aggressive'),('infl','inflammatory'),('disc','discriminatory')]:
    for party in ['Rep','Dem']:
        lab=f"{party} {nm} -> {party} share"
        print(f"{lab:38s}{cols['Keyword'][(party,cat)]:>11s}{cols['Context-aware'][(party,cat)]:>11s}{cols['Supervised'][(party,cat)]:>12s}")
