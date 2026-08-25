"""Evaluates a published detector on our splits, closing the no-baseline objection.

Hello-SimpleAI/chatgpt-detector-roberta is the detector released with HC3 by the
corpus's own authors. Running it changes what the paper can say, but the two corpora
have to be read differently and the difference matters more than the numbers.

  HC3        CONTAMINATED. The detector was trained on HC3, so our test rows were
             almost certainly in its training set. The number is an upper bound on a
             memorised task and is reported for completeness, not as evidence.
  DAIGT V2   CLEAN. The detector never saw this corpus, so this is a genuine
             out-of-domain evaluation and the one that carries weight.

Inference only, no fine-tuning.
"""
import json
import os
import time

os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FINAL = Path(__file__).resolve().parents[2]
WORK = FINAL / 'experiments' / 'paper_scale' / 'work'
OUT = FINAL / 'experiments' / 'audit' / 'published_detector_baseline.json'
NAME = 'Hello-SimpleAI/chatgpt-detector-roberta'
MAXLEN, BS = 512, 64


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NAME).to(dev).eval()
    print(f'{NAME} on {dev}, labels {model.config.id2label}', flush=True)

    rep = {'detector': NAME, 'max_length': MAXLEN, 'inference_only': True,
           'label_map': 'model label 1 = ChatGPT = our label 1', 'datasets': {}}

    for tag, name, contaminated in (('D1', 'DAIGT V2', False), ('D2', 'HC3', True)):
        df = pd.read_parquet(WORK / f'data_{tag}.parquet')
        sp = np.load(WORK / f'split_{tag}.npz')
        ite = df.index.get_indexer(sp['test'])
        texts = [str(t) for t in df['text'].values[ite]]
        y = df['label'].values[ite]

        t0 = time.time()
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), BS):
                b = tok(texts[i:i + BS], truncation=True, max_length=MAXLEN,
                        padding=True, return_tensors='pt').to(dev)
                preds.append(model(**b).logits.argmax(-1).cpu().numpy())
        p = np.concatenate(preds)
        secs = time.time() - t0

        acc = accuracy_score(y, p)
        pre, rec, f1, _ = precision_recall_fscore_support(y, p, average='weighted',
                                                          zero_division=0)
        rep['datasets'][tag] = {
            'name': name, 'n_test': int(len(y)), 'contaminated': contaminated,
            'accuracy': round(float(acc), 4), 'precision': round(float(pre), 4),
            'recall': round(float(rec), 4), 'weighted_f1': round(float(f1), 4),
            'error_rate': round(float(1 - acc), 4),
            'seconds': round(secs, 1),
            'note': ('detector was trained on this corpus, so this is an upper bound on a '
                     'memorised task' if contaminated else
                     'detector never saw this corpus, genuine out-of-domain evaluation'),
        }
        r = rep['datasets'][tag]
        flag = '  [CONTAMINATED, trained on this corpus]' if contaminated else '  [clean, out of domain]'
        print(f'{name:9s} n={len(y):6d}  F1 {f1:.4f}  err {(1-acc)*100:6.2f}%  '
              f'{secs:5.1f}s{flag}', flush=True)

    json.dump(rep, open(OUT, 'w'), indent=1)
    print('\nwritten to', OUT.relative_to(FINAL))


if __name__ == '__main__':
    main()
