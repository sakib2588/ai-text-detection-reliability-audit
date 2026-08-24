import pandas as pd, numpy as np, hashlib, re, json, collections, pathlib
P = '/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project /hc3.jsonl'
OUT = pathlib.Path(__file__).resolve().parent

raw = pd.read_json(P, lines=True)
rows = []
for col, lab in (('human_answers', 0), ('chatgpt_answers', 1)):
    s = raw[[col, 'source']].explode(col).rename(columns={col: 'text'})
    s['label'] = lab
    rows.append(s)
df = pd.concat(rows, ignore_index=True).dropna()
df['text'] = df['text'].astype(str)
print(f'FULL HC3 CORPUS: {len(df):,} answers  ({(df.label==0).sum():,} human / {(df.label==1).sum():,} chatgpt)')

norm = df['text'].map(lambda t: re.sub(r'\s+', ' ', t).strip().lower())
df['h'] = [hashlib.md5(t.encode()).hexdigest() for t in norm]
df['nchar'] = norm.str.len()

print('\n' + '='*70 + '\nA. EXACT DUPLICATES\n' + '='*70)
vc = df['h'].value_counts()
dupg = vc[vc > 1]
duprows = int(dupg.sum() - len(dupg))
print(f'  duplicate groups          {len(dupg):,}')
print(f'  redundant rows            {duprows:,}  ({duprows/len(df)*100:.2f}% of corpus)')
for lab, nm in ((0,'human'), (1,'chatgpt')):
    sub = df[df.label==lab]; v = sub['h'].value_counts(); dg = v[v>1]
    print(f'  {nm:8s}: {int(dg.sum()-len(dg)):,} redundant rows of {len(sub):,} ({(dg.sum()-len(dg))/len(sub)*100:.2f}%)')

print('\n  --- CROSS-LABEL: same text labelled BOTH human and chatgpt (label noise) ---')
g = df.groupby('h')['label'].nunique()
cross = g[g > 1]
print(f'  texts appearing under both labels: {len(cross):,}')
if len(cross):
    ex = df[df.h.isin(cross.index[:3])].sort_values('h')
    for _, r in ex.head(6).iterrows():
        print(f'    label={r.label} src={r.source:12s} {r.text[:70]!r}')

print('\n' + '='*70 + '\nB. DEGENERATE AND ARTEFACT ANSWERS\n' + '='*70)
print(f'  empty (0 chars)           {int((df.nchar==0).sum()):,}')
print(f'  under 20 chars            {int((df.nchar<20).sum()):,}')
print(f'  under 50 chars            {int((df.nchar<50).sum()):,}')

PATTERNS = {
 'rate-limit / API error': r'too many requests|error generating|network error|something went wrong|try again later',
 'refusal / disclaimer'  : r"i'm sorry,? but|i am sorry,? but|as an ai language model|i cannot provide|i do not have (the )?ability",
 'empty-ish placeholder' : r'^\s*(n/?a|none|nothing|\.|-)\s*$',
}
print('\n  pattern matches by class:')
art = pd.Series(False, index=df.index)
for name, pat in PATTERNS.items():
    m = norm.str.contains(pat, regex=True, na=False)
    art |= m
    h_, c_ = int((m & (df.label==0)).sum()), int((m & (df.label==1)).sum())
    print(f'    {name:24s} human {h_:6,}   chatgpt {c_:6,}')
print(f'\n  ANY artefact pattern       human {int((art&(df.label==0)).sum()):,}   chatgpt {int((art&(df.label==1)).sum()):,}')
print(f'  share of chatgpt class that is artefact-flagged: {(art&(df.label==1)).sum()/(df.label==1).sum()*100:.2f}%')

print('\n  --- most frequently repeated CHATGPT answers ---')
cg = df[df.label==1]
top = cg['h'].value_counts().head(8)
for h_, n in top.items():
    t = cg[cg.h==h_]['text'].iloc[0]
    print(f'    x{n:<5d} {t[:88]!r}')

print('\n' + '='*70 + '\nC. BY SOURCE\n' + '='*70)
for src, sub in df.groupby('source'):
    v = sub['h'].value_counts(); dg = v[v>1]
    a = art[sub.index].sum()
    print(f'  {src:12s} n={len(sub):7,}  dup rows {int(dg.sum()-len(dg)):6,} ({(dg.sum()-len(dg))/len(sub)*100:5.2f}%)  artefact {int(a):5,} ({a/len(sub)*100:5.2f}%)')

json.dump({'n_total': int(len(df)), 'dup_groups': int(len(dupg)), 'dup_rows': duprows,
           'cross_label_texts': int(len(cross)),
           'artefact_chatgpt': int((art&(df.label==1)).sum()),
           'artefact_human': int((art&(df.label==0)).sum()),
           'empty': int((df.nchar==0).sum()), 'under20': int((df.nchar<20).sum())},
          open(OUT/'hc3_full_audit.json','w'), indent=2)
df[['label','source','nchar','h']].assign(artefact=art).to_parquet(OUT/'hc3_audit_rows.parquet')
print(f'\nwritten: {OUT}/hc3_full_audit.json and hc3_audit_rows.parquet')
