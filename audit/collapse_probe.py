"""Is the one-directional collapse under attack a LEARNED asymmetry, or just a
saturated decision function?

The attacked HC3 models emit class 0 for ~98.6% of inputs, with the same single
false-positive row recurring across three different corruption mechanisms. That
pattern is equally consistent with two very different stories:

  learned cue      the perturbation genuinely moves text toward the
                   human region of the model's decision space
  saturated head   the model answers "human" to anything it cannot read, so
                   the asymmetry is a property of the head, not of the attack

They are separated by feeding inputs that carry no label information at all.
A model with a learned cue should be uncertain on content-free noise. A
saturated head answers "human" confidently regardless.

Reports argmax distribution and mean max-softmax for each control input, so the
paper can state which story the data supports instead of assuming one.
"""
import json
from pathlib import Path
import random
import string

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FINAL = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
WORK = FINAL / 'paper_scale' / 'work'
MODELS = FINAL / 'paper_scale' / 'models'
OUT = FINAL / 'audit' / 'collapse_probe.json'
N = 400
SEED = 42
MAXLEN = 128


def build_controls(real_texts, rng):
    """Inputs carrying no human/machine label information."""
    ctl = {}
    ctl['random_chars'] = [
        ''.join(rng.choice(string.ascii_lowercase + ' ') for _ in range(800))
        for _ in range(N)]
    ctl['token_shuffled'] = []
    for t in real_texts:
        w = str(t).split()
        rng.shuffle(w)
        ctl['token_shuffled'].append(' '.join(w))
    ctl['repeated_token'] = ['the ' * 200 for _ in range(N)]
    ctl['empty'] = ['' for _ in range(N)]
    ctl['punctuation_only'] = [
        ''.join(rng.choice('.,;:!? ') for _ in range(400)) for _ in range(N)]
    return ctl


@torch.no_grad()
def score(mdir, texts, device):
    tok = AutoTokenizer.from_pretrained(str(mdir))
    model = AutoModelForSequenceClassification.from_pretrained(str(mdir)).to(device)
    model.eval()
    probs = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors='pt').to(device)
        probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    del model, tok
    return np.concatenate(probs, 0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = random.Random(SEED)
    report = {'n_per_control': N, 'device': device, 'max_len': MAXLEN,
              'label_meaning': {'0': 'human', '1': 'machine'}, 'models': {}}

    for tag in ('D2', 'D1'):
        df = pd.read_parquet(WORK / f'data_{tag}.parquet')
        sp = np.load(WORK / f'split_{tag}.npz')
        te = df.loc[sp['test']]
        real = te.sample(N, random_state=SEED)['text'].tolist()
        controls = build_controls(real, rng)

        for mk in ('BERT', 'DeBERTa'):
            mdir = MODELS / f'{tag}_{mk}'
            if not (mdir / 'model.safetensors').exists():
                continue
            key = f'{tag}_{mk}'
            report['models'][key] = {}
            print(f'\n=== {key} ===', flush=True)
            for cname, ctexts in controls.items():
                p = score(mdir, ctexts, device)
                pred = p.argmax(1)
                frac_human = float(np.mean(pred == 0))
                report['models'][key][cname] = {
                    'frac_predicted_human': round(frac_human, 4),
                    'frac_predicted_machine': round(float(np.mean(pred == 1)), 4),
                    'mean_max_softmax': round(float(np.mean(p.max(1))), 4),
                    'mean_p_human': round(float(np.mean(p[:, 0])), 4),
                }
                print(f'  {cname:18s} human={frac_human:6.1%}  '
                      f'conf={np.mean(p.max(1)):.4f}', flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(report, fh, indent=2)
    print('\nwritten to:', OUT)


if __name__ == '__main__':
    main()
