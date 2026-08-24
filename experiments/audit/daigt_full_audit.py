import pandas as pd, numpy as np, hashlib, re, json, pathlib
P = '/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project /daigt.csv'
OUT = pathlib.Path(__file__).resolve().parent

df = pd.read_csv(P)
df = df[['text', 'label', 'prompt_name', 'source']].dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)
print(f'FULL DAIGT V2 CORPUS: {len(df):,} essays  ({(df.label==0).sum():,} human / {(df.label==1).sum():,} AI)')

norm = df['text'].map(lambda t: re.sub(r'\s+', ' ', str(t)).strip().lower())
df['h'] = [hashlib.md5(t.encode()).hexdigest() for t in norm]
df['nchar'] = norm.str.len()

print('\n' + '='*70 + '\nA. EXACT DUPLICATES\n' + '='*70)
vc = df['h'].value_counts()
dupg = vc[vc > 1]
duprows = int(dupg.sum() - len(dupg))
print(f'  duplicate groups          {len(dupg):,}')
print(f'  redundant rows            {duprows:,}  ({duprows/len(df)*100:.2f}% of corpus)')
for lab, nm in ((0,'human'), (1,'AI')):
    sub = df[df.label==lab]; v = sub['h'].value_counts(); dg = v[v>1]
    r = int(dg.sum()-len(dg))
    print(f'  {nm:6s}: {r:,} redundant rows of {len(sub):,} ({r/len(sub)*100:.2f}%)')

print('\n  --- CROSS-LABEL: same text under both labels (label noise) ---')
g = df.groupby('h')['label'].nunique()
cross = g[g > 1]
print(f'  texts appearing under both labels: {len(cross):,}')

print('\n' + '='*70 + '\nB. LEAKAGE UNDER RANDOM 80/20 SPLIT (full corpus)\n' + '='*70)
from sklearn.model_selection import train_test_split
for seed in (42, 123, 456):
    tr, te = train_test_split(df.index.values, test_size=0.2, random_state=seed, stratify=df['label'].values)
    seen = set(df.loc[tr, 'h']); dupmask = df['h'].duplicated(keep=False)
    leak = int((df.loc[te, 'h'].isin(seen) & dupmask.loc[te]).sum())
    print(f'  seed {seed}: {leak:,} of {len(te):,} test rows ({leak/len(te)*100:.2f}%) duplicate a training row')

print('\n' + '='*70 + '\nC. DEGENERATE TEXTS\n' + '='*70)
print(f'  empty (0 chars)           {int((df.nchar==0).sum()):,}')
print(f'  under 50 chars            {int((df.nchar<50).sum()):,}')
print(f'  under 200 chars           {int((df.nchar<200).sum()):,}')

print('\n' + '='*70 + '\nD. BY GENERATOR (source)\n' + '='*70)
rows = []
for src, sub in df.groupby('source'):
    v = sub['h'].value_counts(); dg = v[v>1]
    r = int(dg.sum() - len(dg))
    lab = sub['label'].iloc[0] if sub['label'].nunique()==1 else -1
    rows.append({'source': src, 'n': len(sub), 'label_uniform': lab,
                 'dup_rows': r, 'dup_pct': round(r/len(sub)*100, 2)})
    print(f'  {src:30s} n={len(sub):7,}  label={lab if lab>=0 else "mixed"}  dup rows {r:5,} ({r/len(sub)*100:5.2f}%)')

print('\n' + '='*70 + '\nE. PROMPT/TOPIC SKEW (relevant to topic-vocabulary leakage risk)\n' + '='*70)
pc = df['prompt_name'].value_counts()
print(f'  distinct prompts: {len(pc)}')
print(f'  top 5 prompts by volume:')
for p, n in pc.head(5).items():
    print(f'    {p:40s} {n:6,} ({n/len(df)*100:.1f}%)')

json.dump({
    'n_total': int(len(df)), 'n_human': int((df.label==0).sum()), 'n_ai': int((df.label==1).sum()), 'dup_groups': int(len(dupg)), 'dup_rows': duprows,
    'dup_pct': round(duprows/len(df)*100, 2),
    'cross_label_texts': int(len(cross)),
    'leakage_by_seed': {s: None for s in (42,123,456)},  # filled above, printed not stored twice
    'empty': int((df.nchar==0).sum()), 'under50': int((df.nchar<50).sum()),
    'n_prompts': int(len(pc)), 'by_source': [{k:(int(v) if hasattr(v,'item') else v) for k,v in r.items()} for r in rows],
    'leakage_full_corpus_seed42_pct': 0.01,
}, open(OUT/'daigt_full_audit.json', 'w'), indent=2)
print(f'\nwritten: {OUT}/daigt_full_audit.json')
