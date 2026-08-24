"""Measures DAIGT V2 essay length against the bert-base-uncased tokenizer,
and whether 128-token truncation (used throughout this project's transformer
training) affects the human and AI classes equally. Companion to
hc3_full_audit.py / daigt_full_audit.py -- same audit/ directory, same
"measure against real data, don't just cite the literature" convention.

Sampled (not full-corpus) for tokenizer-speed reasons, seeded for
reproducibility per this project's protocol.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

FINAL_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
DATA_PATH = FINAL_DIR / 'paper_scale/work/data_D1.parquet'
OUT_PATH = FINAL_DIR / 'audit/daigt_token_length_audit.json'
SEED = 42
SAMPLE_N = 2000
MAX_LEN = 128


def main():
    df = pd.read_parquet(DATA_PATH)
    tok = AutoTokenizer.from_pretrained('bert-base-uncased')

    sample = df.sample(SAMPLE_N, random_state=SEED)
    lens = np.array([len(tok.encode(str(t), add_special_tokens=True)) for t in sample['text']])
    sample = sample.assign(tok_len=lens)

    by_label = {}
    for label, name in ((0, 'human'), (1, 'ai')):
        sub = sample[sample.label == label]['tok_len']
        by_label[name] = {
            'n': int(len(sub)),
            'median_len': float(sub.median()),
            'mean_len': float(sub.mean()),
            'pct_exceeding_128': float((sub > MAX_LEN).mean() * 100),
        }

    out = {
        'source_file': str(DATA_PATH),
        'tokenizer': 'bert-base-uncased',
        'sample_n': SAMPLE_N,
        'sample_seed': SEED,
        'overall': {
            'median_len': float(np.median(lens)),
            'mean_len': float(lens.mean()),
            'max_len': int(lens.max()),
            'pct_exceeding_128': float((lens > MAX_LEN).mean() * 100),
            'median_pct_kept_at_128': float(np.median(np.minimum(lens, MAX_LEN) / lens) * 100),
        },
        'by_label': by_label,
        'length_asymmetry_pct': float(
            (by_label['human']['median_len'] - by_label['ai']['median_len'])
            / by_label['ai']['median_len'] * 100
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print('\nwritten to:', OUT_PATH)


if __name__ == '__main__':
    main()
