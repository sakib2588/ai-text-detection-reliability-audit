"""The one-cue baseline, made traceable.

The paper compares a single boolean rule against Tian et al.'s single-token rule.
That comparison was quoted from a working note with no artefact behind it, and the
two sides were not the same metric, so this script recomputes ours on the same
group-aware HC3 test partition every other experiment uses and reports BOTH
accuracy and weighted F1, since Tian et al. quote F1.

Rule: a document is human when a space precedes a sentence-level punctuation mark.
The cue regex is imported from the surface arm so the rule reads exactly the
feature the decomposition credits.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

FINAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FINAL / 'experiments' / 'audit'))
from surface_content_decomposition import WS_PUNCT  # noqa: E402

PROJ = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project ')
WORK = FINAL / 'experiments' / 'paper_scale' / 'work'
OUT = FINAL / 'experiments' / 'audit' / 'single_rule_baseline.json'


def hc3_sources():
    raw = pd.read_json(PROJ / 'hc3.jsonl', lines=True)
    frames = []
    for col in ('human_answers', 'chatgpt_answers'):
        sub = raw[[col, 'source']].explode(col).dropna()
        frames.append(pd.DataFrame({'text': sub[col].astype(str), 'source': sub['source']}))
    return pd.concat(frames, ignore_index=True).drop_duplicates('text').set_index('text')['source']


def score(y, pred):
    _, _, f1, _ = precision_recall_fscore_support(y, pred, average='weighted', zero_division=0)
    return {'n': int(len(y)),
            'accuracy': round(float(accuracy_score(y, pred)), 4),
            'weighted_f1': round(float(f1), 4),
            'error_rate': round(float(1 - accuracy_score(y, pred)), 4)}


def population(df):
    """Score the rule over one row set, whole and by source domain."""
    # label 1 is machine-generated, so the rule predicts 0 when the cue fires
    cue = df['text'].map(lambda t: len(WS_PUNCT.findall(str(t))) > 0)
    pred = np.where(cue, 0, 1)
    y = df['label'].to_numpy()
    rec = {'all': score(y, pred), 'by_source': {}}
    src = df['source'].to_numpy()
    for name in sorted(set(s for s in src if isinstance(s, str))):
        m = src == name
        rec['by_source'][name] = score(y[m], pred[m])
    rec['unmapped_source_rows'] = int(df['source'].isna().sum())
    return rec


def main():
    df = pd.read_parquet(WORK / 'data_D2.parquet')
    df['source'] = df['text'].map(hc3_sources())
    test = np.load(WORK / 'split_D2.npz')['test']

    out = {'seed': 42,
           'rule': "human when re.compile(r' +[.,;:!?]') matches at least once",
           'note': 'the paper quotes the balanced-corpus accuracy, so both populations are recorded',
           'balanced_corpus': population(df),
           'test_split': population(df.iloc[test])}

    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
