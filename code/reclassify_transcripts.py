# -*- coding: utf-8 -*-
# Authors: Bryan Kyung and Katie Kim
"""Upstream re-analysis: re-classify debate transcripts FROM SOURCE.
Parse speaker turns from debate_stats.debate_text, attribute to candidate party,
sentence-segment, and re-classify each sentence with a transparent, documented
lexicon method (independent binary categories, matching the original design).
Regenerate candidate-debate rhetoric counts and compare to the production counts.
"""
import sqlite3, re, pandas as pd, numpy as np

DB="/sessions/youthful-tender-cannon/mnt/Katie/Python_Program/debate_analysis.db"
OUT="/sessions/youthful-tender-cannon/mnt/JHSS_2nd/R1/analysis/reclass_results.txt"

DEM={'OBAMA','BIDEN','CLINTON','KAINE','HARRIS','GORE','KERRY','EDWARDS','LIEBERMAN'}
REP={'MCCAIN','PALIN','ROMNEY','RYAN','TRUMP','PENCE','BUSH','CHENEY'}
CAND=DEM|REP

# --- transparent lexicons (documented in Appendix B) ---
AGG=[r'\bfight', r'\battack', r'\bbeat\b', r'\bdestroy', r'\bcrush', r'\bweak\b', r'\bfail',
     r'\bwrong\b', r'\blie\b', r'\blying\b', r'\bstupid', r'\bfoolish', r'\bdisaster', r'\bincompeten',
     r'\bridiculous', r'\bpathetic', r'\bdefeat']
INF=[r'\bterrible', r'\bhorrible', r'\bdisgrace', r'\bcatastroph', r'\bradical', r'\bdangerous',
     r'\bhatred', r'\boutrage', r'\benemy', r'\bcorrupt', r'\bcriminal', r'\bdestroy(ing)? our',
     r'\bsocialis', r'\bextremist', r'\bthreat to', r'\bevil', r'\bshame']
# discriminatory operationalized strictly: demeaning/targeting a group (not merely mentioning race)
DIS=[r'\bthose people\b', r'\billegals?\b', r'\binvasion\b', r'\banimals?\b', r'\bthugs?\b',
     r'\brapists?\b', r'\bgo back to\b', r'\bthey don.?t belong', r'\bnot like us\b',
     r'\bthese people don', r'\bsome of them.{0,20}assume']
def flags(s):
    s=' '+s.lower()+' '
    return dict(aggr=int(any(re.search(p,s) for p in AGG)),
                infl=int(any(re.search(p,s) for p in INF)),
                disc=int(any(re.search(p,s) for p in DIS)))

con=sqlite3.connect(DB)
ds=pd.read_sql("SELECT election_year, debate_date, debate_text FROM debate_stats", con)
# normalize debate_date
def norm_date(x):
    x=str(x)
    m=re.match(r'(\d{4})-(\d{2})-(\d{2})',x)
    return m.group(0) if m else None
ds['date']=ds['debate_date'].apply(norm_date)

panel_dates=['2008-10-07','2008-10-15','2012-10-03','2012-10-11','2012-10-16','2012-10-22',
             '2016-09-26','2016-10-04','2016-10-09','2016-10-19','2020-09-29','2020-10-22']

def split_turns(text):
    # find speaker tags: uppercase token(s) >=3 letters ending with ':'
    parts=re.split(r'(?<![A-Za-z])([A-Z][A-Z\.\']{2,})\s*:', text)
    # parts: [pre, SPK1, txt1, SPK2, txt2, ...]
    turns=[]
    for i in range(1,len(parts)-1,2):
        spk=parts[i].strip(" .'"); txt=parts[i+1]
        turns.append((spk,txt))
    return turns

def sent_split(t):
    t=re.sub(r'\s+',' ',t)
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if len(s.strip())>=12]

rows=[]
for _,r in ds.iterrows():
    if r['date'] not in panel_dates: continue
    for spk,txt in split_turns(r['debate_text']):
        if spk not in CAND: continue
        party='Dem' if spk in DEM else 'Rep'
        for s in sent_split(txt):
            f=flags(s)
            rows.append((r['date'],spk,party,f['aggr'],f['infl'],f['disc']))
S=pd.DataFrame(rows,columns=['date','speaker','party','aggr','infl','disc'])

out=[]
def P(*a): s=" ".join(str(x) for x in a); print(s); out.append(s)
P("="*80); P("TRANSCRIPT RE-ANALYSIS — re-classified from source (transparent lexicon method)"); P("="*80)
P(f"Debates parsed: {S['date'].nunique()} ; candidate sentences classified: {len(S)}")
P(f"Sentences per party: {S.groupby('party').size().to_dict()}\n")

# counts per debate x party
agg=S.groupby(['date','party']).agg(n_sent=('aggr','size'),
        aggressive=('aggr','sum'), inflammatory=('infl','sum'), discriminatory=('disc','sum')).reset_index()
P("Re-derived candidate-debate rhetoric counts (NEW):")
P(agg.to_string(index=False)); P("")

# Compare to production counts
df=pd.read_stata('/sessions/youthful-tender-cannon/mnt/Katie/poll_final_data_with_change.dta')
d=df.dropna(subset=['change_democrats','sample1','debate_date']).copy()
dd=d.drop_duplicates(subset=['debate_date','candidate']).copy()
dd['date']=pd.to_datetime(dd['debate_date']).dt.strftime('%Y-%m-%d')
dd['party']=np.where(dd['democratdum']==1,'Dem','Rep')
old=dd.groupby(['date','party']).agg(aggressive_old=('aggressive_words','sum'),
        inflammatory_old=('inflammatory_words','sum'), discriminatory_old=('discriminatory_words','sum')).reset_index()
cmp=agg.merge(old,on=['date','party'],how='outer').fillna(0).sort_values(['date','party'])
P("NEW vs PRODUCTION (old) counts by debate x party:")
P(cmp[['date','party','aggressive','aggressive_old','inflammatory','inflammatory_old',
       'discriminatory','discriminatory_old']].to_string(index=False)); P("")

# correlations
for cat,old_c in [('aggressive','aggressive_old'),('inflammatory','inflammatory_old'),('discriminatory','discriminatory_old')]:
    a=cmp[cat].astype(float); b=cmp[old_c].astype(float)
    if a.std()>0 and b.std()>0:
        rcorr=np.corrcoef(a,b)[0,1]
    else: rcorr=float('nan')
    P(f"  corr(new,old) {cat:14s} = {rcorr:.3f}   totals new={a.sum():.0f} old={b.sum():.0f}")

S.to_csv('/sessions/youthful-tender-cannon/mnt/JHSS_2nd/R1/analysis/reclassified_sentences.csv',index=False)
agg.to_csv('/sessions/youthful-tender-cannon/mnt/JHSS_2nd/R1/analysis/reclassified_counts.csv',index=False)
open(OUT,'w').write("\n".join(out))
print("\nwrote",OUT)
