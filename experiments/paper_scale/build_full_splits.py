"""Builds the ONE fixed full-scale balanced split per dataset (seed=42), used by every
training run regardless of training seed. HC3 uses a duplicate-group-aware split so that
no near-identical answer crosses the train/val/test boundary (see Final/audit/hc3_full_audit.json:
7.16% of the full corpus is duplicated). MEASURED 2026-08-24 by audit/verify_paper_claims.py:
this split leaks 0 of 10,732 HC3 test rows, while the naive split in build_naive_splits.py
leaks 570 of 10,762, 5.30%. The "11.2-11.3%" figure previously quoted here was never measured
and is wrong. DAIGT uses a plain stratified split (duplication is 0.01%, group-awareness is a
no-op there but applied uniformly for methodological consistency)."""
import re, hashlib, json, pathlib, gc
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_DIR = pathlib.Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project ')
WORK = pathlib.Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final/paper_scale/work')
WORK.mkdir(parents=True, exist_ok=True)
SPLIT_SEED = 42

def norm(t):
    return re.sub(r'\s+', ' ', str(t)).strip()

def content_hash(series):
    return series.map(lambda t: hashlib.md5(norm(t).lower().encode()).hexdigest())

def balance_full(df, seed=SPLIT_SEED):
    n = int(df['label'].value_counts().min())
    parts = []
    for value in sorted(df['label'].unique()):
        subset = df[df['label'] == value]
        parts.append(subset.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def group_split(df, seed=SPLIT_SEED):
    """72/8/20 train/val/test, never splitting a duplicate-content group across partitions.

    NOT 80/10/10: test_size=0.2 takes a fifth for test, then test_size=0.1 takes a tenth of
    the REMAINING 80% for validation, so the shares are 72/8/20. Verified against the written
    row counts (DAIGT 25,196/2,800/6,998; HC3 38,785/4,289/10,732)."""
    groups = df['hash'].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_full_idx, te_idx = next(gss1.split(df, df['label'], groups))
    sub = df.iloc[tr_full_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_idx_rel, val_idx_rel = next(gss2.split(sub, sub['label'], sub['hash'].values))
    idx_tr = sub.index.values[tr_idx_rel]
    idx_val = sub.index.values[val_idx_rel]
    idx_te = df.index.values[te_idx]
    # verify no group crosses a boundary
    g_tr, g_val, g_te = set(df.loc[idx_tr,'hash']), set(df.loc[idx_val,'hash']), set(df.loc[idx_te,'hash'])
    assert not (g_tr & g_val) and not (g_tr & g_te) and not (g_val & g_te), 'GROUP LEAKAGE ACROSS SPLIT'
    return idx_tr, idx_val, idx_te

def load_D1():
    raw = pd.read_csv(PROJECT_DIR / 'daigt.csv')
    df = raw[['text', 'label']].dropna()
    df['label'] = df['label'].astype(int)
    del raw; gc.collect()
    return balance_full(df)

def load_D2():
    raw = pd.read_json(PROJECT_DIR / 'hc3.jsonl', lines=True)
    human = raw[['human_answers']].explode('human_answers').rename(columns={'human_answers': 'text'})
    human['label'] = 0
    bot = raw[['chatgpt_answers']].explode('chatgpt_answers').rename(columns={'chatgpt_answers': 'text'})
    bot['label'] = 1
    df = pd.concat([human, bot], ignore_index=True).dropna()
    df['text'] = df['text'].astype(str)
    del raw, human, bot; gc.collect()
    return balance_full(df)

LOADERS = {'D1': load_D1, 'D2': load_D2}

if __name__ == '__main__':
    for tag in ('D1', 'D2'):
        df = LOADERS[tag]()
        df['hash'] = content_hash(df['text'])
        n_groups = df['hash'].nunique()
        idx_tr, idx_val, idx_te = group_split(df)
        print('%s  balanced=%d  unique_content_groups=%d (dup rows=%d)  train=%d val=%d test=%d' % (
            tag, len(df), n_groups, len(df)-n_groups, len(idx_tr), len(idx_val), len(idx_te)))
        print('   test label balance: %s' % df.loc[idx_te,'label'].value_counts().sort_index().to_dict())
        df[['text','label']].to_parquet(WORK / ('data_%s.parquet' % tag), index=True)
        np.savez(WORK / ('split_%s.npz' % tag), train=idx_tr, val=idx_val, test=idx_te)
        del df; gc.collect()
    print('\nsplits written to', WORK)
