# -*- coding: utf-8 -*-
# Authors: Bryan Kyung and Katie Kim
"""Generate rhetoric variables with a transparent, replicable context-aware classifier,
validate on the expert gold set, rebuild the poll panel, and run the ORIGINAL model
specifications. Outputs results to ../RESULTS.txt"""
import json,re,math,numpy as np,pandas as pd,numpy.linalg as la
D="/sessions/happy-nifty-darwin/mnt/Katie/JHSS_2nd/R1/Data_Files/"
# ---------- p-value helper (no scipy) ----------
def betacf(a,b,x):
    F=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1.;d=1-qab*x/qap;d=F if abs(d)<F else d;d=1/d;h=d
    for m in range(1,300):
        m2=2*m;aa=m*(b-m)*x/((qam+m2)*(a+m2));d=1+aa*d;d=F if abs(d)<F else d;c=1+aa/c;c=F if abs(c)<F else c;d=1/d;h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));d=1+aa*d;d=F if abs(d)<F else d;c=1+aa/c;c=F if abs(c)<F else c;d=1/d;de=d*c;h*=de
        if abs(de-1)<3e-12:break
    return h
def tpval(t,df):
    t=abs(t);x=df/(df+t*t);lb=math.lgamma(df/2)+math.lgamma(.5)-math.lgamma(df/2+.5)
    return math.exp(math.log(x)*(df/2)+math.log(1-x)*.5-lb)/(df/2)*betacf(df/2,.5,x) if x<(df/2+1)/(df/2+1.5) else 1-math.exp(math.log(x)*(df/2)+math.log(1-x)*.5-lb)/.5*betacf(.5,df/2,1-x)
def stars(p): return '***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''

# ---------- context-aware classifier (rubric implemented as rules) ----------
AGG=['fight','fighting','attack','beat','destroy','defeat','wrong','lie','lying','liar','failed','fail','incompetent','voted against','you said',"doesn't mention","don't put words"]
INF=['disaster','disgrace','horrible','horribly','terrible','radical','dangerous','hatred','destroying','catastrophe','stupid','tragedy','heartbreak','enemy','corrupt','extreme','outrage','evil','crisis','disgraceful','heartbreaking']
GROUP=['immigrant','immigrants','mexican','mexicans','muslim','muslims','islam ','islamic','black people','african american','latino','latinos','hispanic','women','jewish','jews','gay ','lesbian','transgender']
DEROG=['animals','thugs','rapists','criminals','invasion','infest','vermin','don\'t belong','do not belong','go back to your country','take their place','these people are','those people are']
CONDEMN=['racist','xenophob','calling','he calls','she calls','called them','called us','said that','because it','denounce','condemn']
FOREIGN=['north korea','putin','xi ','jinping','kim ','jong','dictator','iran','tehran','china president','chinese president']
NEG=['not','no','never','hardly',"n't"]
def cls(s):
    t=' '+str(s).lower()+' '
    def has(lex): return any(w in t for w in lex)
    # aggressive
    agg=1 if has(AGG) else 0
    # inflammatory (weighted, with crude negation skip)
    inf=0
    for w in INF:
        if w in t:
            i=t.find(w);pre=t[max(0,i-22):i]
            if any(n+' ' in pre for n in NEG): continue
            inf=1;break
    # discriminatory: protected group + derogation, NOT condemnation/quote/foreign-leader
    dis=0
    if (has(GROUP) and has(DEROG)) and not has(CONDEMN) and not has(FOREIGN):
        dis=1
    return agg,inf,dis

# ---------- classify corpus ----------
corp=pd.read_csv(D+"corpus_sentences.csv")
lab=corp['sentence'].apply(cls)
corp['aggressive']=[x[0] for x in lab]; corp['inflammatory']=[x[1] for x in lab]; corp['discriminatory']=[x[2] for x in lab]
print("corpus sentences:",len(corp),"| flagged: agg=%d inf=%d dis=%d"%(corp.aggressive.sum(),corp.inflammatory.sum(),corp.discriminatory.sum()))

# ---------- validate on gold ----------
gtxt=[r['sentence'] for r in json.load(open(D+"gold_sample.json"))]
gold=list("NNNIAINNNI"+"NNNINNINNN"+"NNNNNNANNN"+"ANANNNNINI"+"NNNIAINNNN"+"NINNANNNNN")
gp=[cls(s) for s in gtxt]
print("\n=== GOLD VALIDATION (per-category binary) ===")
for j,(cat,L) in enumerate([('aggressive','A'),('inflammatory','I'),('discriminatory','D')]):
    tp=sum(gp[k][j]==1 and gold[k]==L for k in range(60)); fp=sum(gp[k][j]==1 and gold[k]!=L for k in range(60)); fn=sum(gp[k][j]==0 and gold[k]==L for k in range(60)); ng=sum(g==L for g in gold)
    P=tp/(tp+fp) if tp+fp else float('nan'); R=tp/(tp+fn) if tp+fn else float('nan'); F=2*P*R/(P+R) if (P==P and R==R and P+R>0) else float('nan')
    print(f"  {cat:14s} n_gold={ng:2d} P={P:.2f} R={R:.2f} F1={F:.2f}")

# ---------- aggregate to debate x party, build panel ----------
corp['date']=corp['date'].astype(str)
g=corp.groupby(['date','party'])[['aggressive','inflammatory','discriminatory']].sum().reset_index()
piv=g.pivot_table(index='date',columns='party',values=['aggressive','inflammatory','discriminatory'],fill_value=0)
piv.columns=[f'{a}_{b.lower()[:1]}' for a,b in piv.columns]; piv=piv.reset_index()  # e.g. aggressive_d, aggressive_r
for c in ['aggressive','inflammatory','discriminatory']:
    piv[c]=piv.get(f'{c}_d',0)+piv.get(f'{c}_r',0)
# rename to match original variable names
ren={'aggressive':'aggressive_words','inflammatory':'inflammatory_words','discriminatory':'discriminatory_words',
     'aggressive_d':'aggressive_words_d','aggressive_r':'aggressive_words_r','inflammatory_d':'inflammatory_words_d',
     'inflammatory_r':'inflammatory_words_r','discriminatory_d':'discriminatory_words_d','discriminatory_r':'discriminatory_words_r'}
piv=piv.rename(columns=ren)

d=pd.read_stata(D+"poll_final_data_with_change.dta")
d=d.dropna(subset=['change_democrats','change_republicans','pre_democrats','pre_republicans']).copy()
d['year']=d.year.astype(int); d['date']=pd.to_datetime(d['debate_date']).dt.strftime('%Y-%m-%d')
newcols=list(ren.values())
d=d.drop(columns=[c for c in newcols if c in d.columns]).merge(piv[['date']+newcols],on='date',how='left')
d=d.dropna(subset=newcols)
print(f"\npanel after merge: N={len(d)} debate-clusters={d.debate_date.nunique()}")

# ---------- regression engine (WLS, year FE, debate-clustered) ----------
TOPIC=['Immigration','ForeignPolicy','AbortionRights']
def reg(d,y,xv,fe='year',w='sample1',cl='debate_date'):
    d=d.dropna(subset=[y]+xv+[w,cl]).copy(); X=d[xv].astype(float).values
    yrs=sorted(d[fe].unique())[1:]; fed=np.column_stack([(d[fe]==z).astype(float).values for z in yrs]) if yrs else np.empty((len(d),0))
    Xm=np.column_stack([np.ones(len(d)),X,fed]); names=['const']+xv+[f'y{z}' for z in yrs]
    wt=d[w].astype(float).values; sw=np.sqrt(wt/wt.mean()); Y=d[y].astype(float).values; Xw=Xm*sw[:,None]; Yw=Y*sw
    keep=np.ones(Xm.shape[1],bool)
    if la.matrix_rank(Xw)<Xm.shape[1]:
        cur=[]
        for j in range(Xm.shape[1]):
            if la.matrix_rank(Xw[:,cur+[j]])==len(cur)+1: cur.append(j)
        keep=np.zeros(Xm.shape[1],bool); keep[cur]=True
    Xw2=Xw[:,keep]; nm=[n for n,k in zip(names,keep) if k]; b=la.lstsq(Xw2,Yw,rcond=None)[0]; r=Yw-Xw2@b
    br=la.inv(Xw2.T@Xw2); clv=d[cl].values; meat=np.zeros((Xw2.shape[1],)*2)
    for c in np.unique(clv): ix=clv==c; s=Xw2[ix].T@r[ix]; meat+=np.outer(s,s)
    G=len(np.unique(clv)); n=len(d); k=Xw2.shape[1]; V=br@meat@br*(G/(G-1))*((n-1)/(n-k)); se=np.sqrt(np.diag(V))
    out={nm[i]:(b[i],se[i],tpval(b[i]/se[i],G-1)) for i in range(len(nm))}
    ss=1-np.sum(r**2)/np.sum((Yw-Yw.mean())**2); adj=1-(1-ss)*(n-1)/(n-k)
    return out,dict(N=n,G=G,adj=adj)
def show(title,o,i,vs):
    print(f"\n--- {title}  N={i['N']} clusters={i['G']} adjR2={i['adj']:.3f}")
    for v in vs:
        if v in o: b,s,p=o[v]; print(f"   {v:24s} {b:+.4f}{stars(p):3s} (p={p:.3f})")
        else: print(f"   {v:24s} [dropped/collinear]")

print("\n========== MODEL 1: BASELINE (unsigned), original spec ==========")
base=['aggressive_words','inflammatory_words','discriminatory_words','pre_democrats','pre_republicans']+TOPIC
o,i=reg(d,'change_democrats',base); show("DV=Change_Democrats",o,i,base[:3])
o,i=reg(d,'change_republicans',base); show("DV=Change_Republicans",o,i,base[:3])
print("\n========== MODEL 2: SOURCE-SPECIFIC, original spec ==========")
src=['aggressive_words_d','aggressive_words_r','inflammatory_words_d','inflammatory_words_r','discriminatory_words_d','discriminatory_words_r','pre_democrats','pre_republicans']+TOPIC
o,i=reg(d,'change_democrats',src); show("DV=Change_Democrats",o,i,src[:6])
o,i=reg(d,'change_republicans',src); show("DV=Change_Republicans",o,i,src[:6])
print("\n========== MODEL 3: SAMPLE-SIZE INTERACTIONS ==========")
d2=d.copy(); inter=[]
for c in src[:6]:
    d2[c+'_xhs']=d2[c]*d2['high_sample1']; inter.append(c+'_xhs')
xv=src[:6]+['high_sample1']+inter+['pre_democrats','pre_republicans']+TOPIC
o,i=reg(d2,'change_democrats',xv); show("DV=Change_Democrats",o,i,inter)
o,i=reg(d2,'change_republicans',xv); show("DV=Change_Republicans",o,i,inter)
print("\n========== MODEL 4: DEBATE-LEVEL AGGREGATION ==========")
agg=d.groupby(['date']).agg({'change_democrats':'mean','change_republicans':'mean','aggressive_words':'first','inflammatory_words':'first','discriminatory_words':'first'}).reset_index()
def ols(dd,y,xv):
    dd=dd.dropna(subset=[y]+xv); X=np.column_stack([np.ones(len(dd))]+[dd[c].astype(float).values for c in xv]); Y=dd[y].astype(float).values
    b=la.lstsq(X,Y,rcond=None)[0]; r=Y-X@b; n,k=X.shape; br=la.inv(X.T@X); meat=(X*r[:,None]).T@(X*r[:,None])*(n/(n-k)); V=br@meat@br; se=np.sqrt(np.diag(V))
    return {(['const']+xv)[j]:(b[j],se[j],tpval(b[j]/se[j],n-k)) for j in range(len(b))}
for dv in ['change_democrats','change_republicans']:
    o=ols(agg,dv,['aggressive_words','inflammatory_words','discriminatory_words'])
    print(f"  DV={dv} (N={len(agg)}):", {k:(round(v[0],3),round(v[2],3)) for k,v in o.items() if k!='const'})
