"""Surface-content decomposition per sub-corpus, not per corpus.

The paper measures two benchmarks. Both carry subgroup labels it never uses, and
those labels rejoin the existing splits with no loss, so the same measurement runs
over roughly twenty sub-corpora at CPU cost.

The two corpora need different subgroup logic.

  HC3        splits by source domain. Each domain carries both classes, so a domain
             is a self-contained human-versus-machine task.
  DAIGT V2   splits by machine generator. A generator supplies only the machine side,
             so each is paired against the shared human essay pool. The question
             becomes whether THAT generator's output is separable by form.

Comparability requires care the corpus-level analysis did not. Class balance varies
sharply by subgroup, from 0.41 machine on reddit_eli5 to 0.88 on open_qa, so each
subgroup is rebalanced to parity within train and within test before fitting.
Without that, error rates are not comparable across subgroups.

The global group-aware split is preserved by restriction rather than resampled, so a
subgroup inherits the corpus's leakage guarantee instead of needing its own.

Subgroups below MIN_TEST balanced test rows are computed but flagged, because a paired
test there fails to separate for want of power rather than for want of an effect, and
the two must not be reported as the same thing.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

FINAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FINAL / 'experiments' / 'audit'))
from surface_content_decomposition import content_normalise, surface_features  # noqa: E402

PROJ = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project ')
WORK = FINAL / 'experiments' / 'paper_scale' / 'work'
OUT = FINAL / 'experiments' / 'audit' / 'subgroup_decomposition.json'
SEED, BOOT, MIN_TEST = 42, 10000, 200
rng = np.random.default_rng(SEED)


def hc3_sources():
    raw = pd.read_json(PROJ / 'hc3.jsonl', lines=True)
    frames = []
    for col in ('human_answers', 'chatgpt_answers'):
        sub = raw[[col, 'source']].explode(col).dropna()
        frames.append(pd.DataFrame({'text': sub[col].astype(str), 'source': sub['source']}))
    return pd.concat(frames, ignore_index=True).drop_duplicates('text').set_index('text')['source']


def daigt_sources():
    d = pd.read_csv(PROJ / 'daigt.csv', usecols=['text', 'source'])
    return d.drop_duplicates('text').set_index('text')['source']


def balance(idx, y, r):
    """Downsample to parity within an index set."""
    a = idx[y[idx] == 0]
    b = idx[y[idx] == 1]
    n = min(len(a), len(b))
    if n == 0:
        return np.array([], dtype=int)
    return np.concatenate([r.choice(a, n, replace=False), r.choice(b, n, replace=False)])


def fit_predict(Xtr, ytr, Xte):
    clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr, ytr)
    return clf.predict(Xte)


def err(y, p):
    return float(1 - accuracy_score(y, p))


def f1(y, p):
    _, _, v, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return float(v)


def paired(y, pa, pb, r):
    wa, wb = pa != y, pb != y
    b = int((~wa & wb).sum())
    c = int((wa & ~wb).sum())
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    idx = r.integers(0, len(y), size=(BOOT, len(y)))
    boot = wa[idx].mean(1) - wb[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'mcnemar_p': round(float(p), 6),
            'err_diff_pp': round(float((wa.mean() - wb.mean()) * 100), 3),
            'ci95_pp': [round(float(lo * 100), 3), round(float(hi * 100), 3)],
            'excludes_zero': bool(lo > 0 or hi < 0)}


def run_corpus(tag, name, subgroup_of, pair_against_human_pool):
    df = pd.read_parquet(WORK / f'data_{tag}.parquet')
    sp = np.load(WORK / f'split_{tag}.npz')
    y = df['label'].values
    itr = df.index.get_indexer(sp['train'])
    ite = df.index.get_indexer(sp['test'])
    src = df['text'].map(subgroup_of).values

    t0 = time.time()
    X = np.nan_to_num(np.array([surface_features(t) for t in df['text'].values], dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.array([content_normalise(t) for t in df['text'].values], dtype=object)
    print(f'[{tag}] featurised {len(df)} docs in {time.time()-t0:.1f}s', flush=True)

    if pair_against_human_pool:
        groups = sorted({s for s, lab in zip(src, y) if lab == 1 and isinstance(s, str)})
    else:
        groups = sorted({s for s in src if isinstance(s, str)})

    out = {}
    for g in groups:
        if pair_against_human_pool:
            sel = ((src == g) & (y == 1)) | (y == 0)
        else:
            sel = src == g
        tr = np.array([i for i in itr if sel[i]])
        te = np.array([i for i in ite if sel[i]])
        tr, te = balance(tr, y, rng), balance(te, y, rng)
        if len(tr) < 40 or len(te) < 20:
            print(f'  {g:26s} skipped, too few rows ({len(tr)} train / {len(te)} test)')
            continue

        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        Xz = (X - mu) / sd
        ps = fit_predict(Xz[tr], y[tr], Xz[te])

        vec = CountVectorizer()
        Ctr, Cte = vec.fit_transform(norm[tr]), vec.transform(norm[te])
        pc = fit_predict(Ctr, y[tr], Cte)

        yte = y[te]
        rec = {'n_train': int(len(tr)), 'n_test': int(len(te)),
               'surface_err': round(err(yte, ps), 4), 'surface_f1': round(f1(yte, ps), 4),
               'content_err': round(err(yte, pc), 4), 'content_f1': round(f1(yte, pc), 4),
               'paired_surface_minus_content': paired(yte, ps, pc, rng),
               'underpowered': bool(len(te) < MIN_TEST)}
        rec['surface_share'] = round(
            1 - rec['surface_err'] / max(rec['surface_err'] + rec['content_err'], 1e-9), 4)
        out[g] = rec
        flag = '  [UNDERPOWERED]' if rec['underpowered'] else ''
        pr = rec['paired_surface_minus_content']
        print(f'  {g:26s} n_te {len(te):5d}  surface {rec["surface_err"]*100:6.2f}%  '
              f'content {rec["content_err"]*100:6.2f}%  diff {pr["err_diff_pp"]:+7.2f} pp  '
              f'p={pr["mcnemar_p"]:.3g}{flag}', flush=True)
    return out


def main():
    report = {'seed': SEED, 'bootstrap': BOOT, 'min_test_for_claim': MIN_TEST,
              'balancing': 'each subgroup downsampled to parity within train and within test',
              'split': 'global group-aware split restricted to the subgroup, never resampled',
              'corpora': {}}
    print('=== HC3, by source domain ===', flush=True)
    report['corpora']['HC3'] = run_corpus('D2', 'HC3', hc3_sources(), False)
    print('\n=== DAIGT V2, by machine generator, each against the shared human pool ===', flush=True)
    report['corpora']['DAIGT V2'] = run_corpus('D1', 'DAIGT V2', daigt_sources(), True)
    json.dump(report, open(OUT, 'w'), indent=1)
    print('\nwritten to', OUT.relative_to(FINAL))


if __name__ == '__main__':
    main()
