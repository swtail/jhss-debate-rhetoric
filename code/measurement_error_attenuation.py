# Measurement-error attenuation of a debate-level rhetoric effect.
# Author(s) withheld for peer review.
#
# A classifier turns each sentence's TRUE label y into a predicted label yhat with
# category-specific sensitivity s = P(yhat=1|y=1) and false-positive rate f =
# P(yhat=1|y=0), estimated against the 372-sentence human gold standard (supervised
# scored by 5-fold CV; lexicons applied directly). The debate-level regressor is the
# SUM of yhat over a debate's sentences. Under classical errors-in-variables the OLS
# coefficient on the observed count is attenuated by the reliability ratio
#   lambda = Cov(Nhat, Ntrue) / Var(Nhat),
# so the estimated effect is lambda * (true effect). We simulate debates (true positives
# ~ Binomial(L, base rate); observed = Binomial(Ntrue,s)+Binomial(L-Ntrue,f)) and report
# lambda per classifier x category. Lower lambda = more of any real effect erased by the
# measure; this is why an unvalidated classifier can shrink or flip a substantive estimate.
#
# Requires: numpy, pandas.  Reads: data/human_gold_*.csv, data/corpus_*.csv
import os, re
import numpy as np, pandas as pd
from collections import Counter
D=os.path.join(os.path.dirname(__file__),"..","data")
rng=np.random.default_rng(0)

# ---- gold labels ----
ai=pd.read_csv(os.path.join(D,"human_gold_aggr_infl_372.csv")); S=ai["sentence"].astype(str).tolist()
A=((ai["aggr_coder1"].fillna(0)+ai["aggr_coder2"].fillna(0))>0).astype(int).values
I=((ai["infl_coder1"].fillna(0)+ai["infl_coder2"].fillna(0))>0).astype(int).values
dg=pd.read_csv(os.path.join(D,"human_gold_discriminatory_372.csv")); Sd=dg["sentence"].astype(str).tolist()
Dd=dg["gold_lenient"].astype(int).values

# ---- lexicons (subset for aggr/infl/disc) ----
AGG=['fight','fighting','attack','beat','destroy','defeat','wrong','lie','lying','liar','failed','fail','incompetent','voted against','you said',"doesn't mention","don't put words"]
INF=['disaster','disgrace','horrible','horribly','terrible','radical','dangerous','hatred','destroying','catastrophe','stupid','tragedy','heartbreak','enemy','corrupt','extreme','outrage','evil','crisis','disgraceful','heartbreaking']
GROUP=['immigrant','immigrants','mexican','mexicans','muslim','muslims','islam ','islamic','black people','african american','latino','latinos','hispanic','women','jewish','jews','gay ','lesbian','transgender']
DEROG=['animals','thugs','rapists','criminals','invasion','infest','vermin',"don't belong",'do not belong','go back to your country','take their place','these people are','those people are']
def has(t,lex):return any(w in t for w in lex)
def lex_pred(sent_list,cat):
    out=[]
    for s in sent_list:
        t=' '+str(s).lower()+' '
        if cat=='aggr': out.append(1 if has(t,AGG) else 0)
        elif cat=='infl': out.append(1 if has(t,INF) else 0)
        else: out.append(1 if (has(t,GROUP) or has(t,DEROG)) else 0)
    return np.array(out)

# ---- supervised CV oof predictions (TF-IDF + IPW-free logistic, balanced) ----
def tok(s):
    t=re.findall(r"[a-z']+",str(s).lower());return t+[t[i]+'_'+t[i+1] for i in range(len(t)-1)]
def vocab(docs,minc=2):
    c=Counter()
    for d in docs:c.update(set(tok(d)))
    return {w:i for i,w in enumerate([w for w,n in c.items() if n>=minc])}
def idf(docs,v):
    df=np.zeros(len(v))
    for d in docs:
        for w in set(tok(d)):
            if w in v: df[v[w]]+=1
    return np.log((1+len(docs))/(1+df))+1
def feats(docs,v,id_):
    X=np.zeros((len(docs),len(v)))
    for r,d in enumerate(docs):
        for w in tok(d):
            if w in v:X[r,v[w]]+=1
    X=np.log1p(X)*id_;n=np.linalg.norm(X,axis=1,keepdims=True);n[n==0]=1;return X/n
def lr(X,y,l2=1.,it=600,lr_=.5):
    n,p=X.shape;w=np.zeros(p);b=0.;cw=np.where(y==1,n/(2*max(y.sum(),1)),n/(2*max((1-y).sum(),1)))
    for _ in range(it):
        pr=1/(1+np.exp(-(X@w+b)));g=(pr-y)*cw;w-=lr_*(X.T@g/n+l2*w/n);b-=lr_*g.mean()
    return w,b
def cv_oof(Sl,y,k=5,seed=0):
    rg=np.random.RandomState(seed);i0,i1=np.where(y==0)[0],np.where(y==1)[0];rg.shuffle(i0);rg.shuffle(i1)
    F=[[] for _ in range(k)]
    for i,x in enumerate(i1):F[i%k].append(x)
    for i,x in enumerate(i0):F[i%k].append(x)
    F=[np.array(f) for f in F];oof=np.zeros(len(y))
    for f in range(k):
        te=F[f];tr=np.concatenate([F[j] for j in range(k) if j!=f])
        v=vocab([Sl[i] for i in tr]);id_=idf([Sl[i] for i in tr],v);w,b=lr(feats([Sl[i] for i in tr],v,id_),y[tr])
        oof[te]=(1/(1+np.exp(-(feats([Sl[i] for i in te],v,id_)@w+b)))>=.5).astype(int)
    return oof.astype(int)

def sf(true,pred):
    s=pred[true==1].mean() if (true==1).any() else 0.0       # sensitivity
    f=pred[true==0].mean() if (true==0).any() else 0.0       # false-positive rate
    base=true.mean()
    return s,f,base

# sentences per debate (mean) for realistic L
import glob
corp=pd.concat([pd.read_csv(os.path.join(D,f)).rename(columns=str.lower) for f in
                ["corpus_2004.csv","corpus_sentences_2008_2020.csv","corpus_2024.csv"]],ignore_index=True)
L=int(round(len(corp)/corp['date'].astype(str).str[:10].nunique()/2))  # per candidate-debate (2 speakers)
print(f"mean sentences per candidate-debate L ≈ {L}")

def reliability(s,f,base,L,reps=8000):
    # base = CORPUS prevalence of the category (not the enriched-gold rate)
    Ntrue=rng.binomial(L,base,reps)
    Nhat=rng.binomial(Ntrue,s)+rng.binomial(L-Ntrue,f)
    return np.cov(Nhat,Ntrue)[0,1]/np.var(Nhat)

CORP_PREV={'aggr':0.171,'infl':0.053,'disc':0.025}   # supervised corpus prevalence (true-rate proxy)
CATS=[('aggr',S,A),('infl',S,I),('disc',Sd,Dd)]
print("="*82);print("MEASUREMENT-ERROR ATTENUATION  (lambda = retained fraction of a true effect)")
print("  s,f estimated on gold; base = corpus prevalence; L sentences per candidate-debate")
print("="*82)
print(f"{'category':12s}{'classifier':12s}{'sens':>6s}{'FPR':>6s}{'corpPrev':>9s}{'lambda':>9s}{'attenuated':>12s}")
for cat,Sl,y in CATS:
    sup=cv_oof(Sl,y)
    preds={'supervised':sup,'lexicon':lex_pred(Sl,cat)}
    base=CORP_PREV[cat]
    for nm,pred in preds.items():
        s,f,_=sf(y,pred); lam=reliability(s,f,base,L)
        print(f"{cat:12s}{nm:12s}{s:6.2f}{f:6.2f}{base:9.3f}{lam:9.2f}{(1-lam)*100:10.0f}%")
print("\nNote: lambda is the OLS attenuation factor for the debate-level count regressor")
print("(estimated effect = lambda * true effect). Lower lambda = more of a real effect erased.")
