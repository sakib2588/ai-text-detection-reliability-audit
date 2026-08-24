"""Measure real training throughput and peak VRAM at longer sequence lengths.

Extrapolating transformer cost from 128 tokens is unreliable, because attention
is quadratic in length while the rest is linear, and because an out-of-memory
fallback to a smaller batch changes the constant. This times a fixed number of
real training steps at each candidate length and reports seconds per step and
peak memory, so the full-run estimate is measured rather than guessed.

Runs no evaluation and saves nothing. It exists only to produce an estimate.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)
from datasets import Dataset

FINAL = Path(__file__).resolve().parents[2]
WORK = FINAL / 'experiments' / 'paper_scale' / 'work'
OUT = FINAL / 'experiments' / 'audit' / 'seqlen_calibration.json'

MODELS = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
# the deployed configurations, so the calibration matches what would be re-run
CFG = {('D1', 'BERT'): 32, ('D1', 'DeBERTa'): 16,
       ('D2', 'BERT'): 16, ('D2', 'DeBERTa'): 16}
N_STEPS = 30
WARMUP = 5


def time_config(tag, mk, seq_len, bs):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    df = pd.read_parquet(WORK / f'data_{tag}.parquet')
    sp = np.load(WORK / f'split_{tag}.npz')
    sub = df.loc[sp['train']].head(bs * (N_STEPS + WARMUP + 2))
    tok = AutoTokenizer.from_pretrained(MODELS[mk])
    ds = Dataset.from_dict({'text': [str(t) for t in sub['text']],
                            'labels': [int(v) for v in sub['label']]})
    ds = ds.map(lambda b: tok(b['text'], truncation=True, max_length=seq_len),
                batched=True, remove_columns=['text'])
    dl = DataLoader(ds, batch_size=bs, shuffle=False,
                    collate_fn=DataCollatorWithPadding(tok))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[mk], num_labels=2).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    times = []
    for i, batch in enumerate(dl):
        if i >= N_STEPS + WARMUP:
            break
        batch = {k: v.cuda() for k, v in batch.items()}
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= WARMUP:
            times.append(time.time() - t0)

    peak = torch.cuda.max_memory_allocated() / 1024**3
    del model, opt; torch.cuda.empty_cache()
    return float(np.median(times)), float(peak)


def main():
    report = {'n_timed_steps': N_STEPS, 'gpu': torch.cuda.get_device_name(0),
              'results': {}}
    for (tag, mk), bs in CFG.items():
        n_train = len(np.load(WORK / f'split_{tag}.npz')['train'])
        for seq_len in (128, 256, 512):
            key = f'{tag}_{mk}_len{seq_len}'
            try:
                sps, peak = time_config(tag, mk, seq_len, bs)
                steps_per_epoch = int(np.ceil(n_train / bs))
                report['results'][key] = {
                    'batch_size': bs, 'seq_len': seq_len,
                    'sec_per_step': round(sps, 4),
                    'peak_vram_gib': round(peak, 2),
                    'steps_per_epoch': steps_per_epoch,
                    'est_sec_per_epoch': round(sps * steps_per_epoch, 1),
                    'oom': False,
                }
                print(f'  {key:24s} bs={bs:2d} {sps*1000:7.1f} ms/step '
                      f'peak={peak:5.2f}GiB  epoch~{sps*steps_per_epoch/60:5.1f} min', flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                report['results'][key] = {'batch_size': bs, 'seq_len': seq_len, 'oom': True}
                print(f'  {key:24s} bs={bs:2d} OOM', flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, 'w'), indent=2)
    print('\nwritten to:', OUT)


if __name__ == '__main__':
    main()
