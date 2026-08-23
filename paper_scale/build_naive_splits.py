"""Builds a PLAIN random 80/10/10 split (no duplicate-group awareness) on the SAME
balanced sample already used for the duplicate-aware split, so the only difference
between the two conditions is the split method itself -- not which rows were sampled.
Written to distinct filenames (split_D1_naive.npz / split_D2_naive.npz) so this can
never collide with or overwrite the group-aware split_D1.npz / split_D2.npz that the
main sweep depends on."""
import pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

WORK = pathlib.Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final/paper_scale/work')

for tag in ('D1', 'D2'):
    pq = WORK / ('data_%s.parquet' % tag)
    out = WORK / ('split_%s_naive.npz' % tag)
    if out.exists():
        print('[skip] %s naive split already built' % tag); continue
    df = pd.read_parquet(pq)  # the SAME balanced sample used by the group-aware split
    idx_tr, idx_te = train_test_split(df.index.values, test_size=0.2, random_state=42,
                                      stratify=df['label'].values)
    idx_tr, idx_val = train_test_split(idx_tr, test_size=0.1, random_state=42,
                                       stratify=df.loc[idx_tr, 'label'].values)
    np.savez(out, train=idx_tr, val=idx_val, test=idx_te)
    print('%s naive split: train=%d val=%d test=%d' % (tag, len(idx_tr), len(idx_val), len(idx_te)))
    # quantify the leak this split actually contains, for the paper's before/after table
    dupmask = df['text'].duplicated(keep=False)
    seen_hash = set(df.loc[idx_tr, 'text']) | set(df.loc[idx_val, 'text'])
    leaked = df.loc[idx_te, 'text'].isin(seen_hash).sum()
    print('   exact-text leak in this naive test split: %d / %d (%.2f%%)' % (
        leaked, len(idx_te), 100*leaked/len(idx_te)))
