"""Adversarial robustness evaluation (Gap 2): apply typo injection, homoglyph
substitution, and back-translation paraphrase to TEST TEXT ONLY (never
training data), and measure how much the existing in-domain, raw-trained
checkpoints degrade under each attack. This is the piece that turns "our
detector scores 99%" into "here's what survives contact with a simple
adversarial attack" -- see Final/paper_review/LITERATURE_REVIEW_CANONICAL.md
Section 3.4 for the design this follows.

Inference only -- no training, no gradients -- so this runs on CPU by
default, same convention as cross_dataset_eval.py, specifically so it does
not contend with a concurrent GPU training job.
"""
import os, gc, json, time
from pathlib import Path
import random
import numpy as np, pandas as pd

os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from text_perturbations import inject_typos, homoglyph_substitute, backtranslate

FINAL_DIR   = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final/paper_scale')
WORK_DIR    = FINAL_DIR / 'work'
MODELS_DIR  = FINAL_DIR / 'models'
RESULTS_DIR = FINAL_DIR / 'results'
PROBS_DIR   = FINAL_DIR / 'probs'
MAX_LEN = 128
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}
STRENGTHS = (0.01, 0.05, 0.10)


def load_test_split(tag):
    df = pd.read_parquet(WORK_DIR / f'data_{tag}.parquet')
    sp = np.load(WORK_DIR / f'split_{tag}.npz')
    return df.loc[sp['test']]


def weighted_metrics(y, p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    a = accuracy_score(y, p)
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return round(a, 4), round(pr, 4), round(rc, 4), round(f, 4)


def atomic_write_json(path, obj):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_npz(path, **arrays):
    path = Path(path); tmp = path.with_suffix('.npz.tmp')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def perturb_texts(texts, attack, strength, device='cpu'):
    """attack in {'typo', 'homoglyph', 'backtranslation'}. strength is a
    float rate for typo/homoglyph, ignored for backtranslation."""
    if attack == 'backtranslation':
        return backtranslate(texts, device=device)
    rng = random.Random(42)  # single seeded RNG, applied in row order -> reproducible
    fn = inject_typos if attack == 'typo' else homoglyph_substitute
    return [fn(t, strength, rng) for t in texts]


def score_checkpoint(mdir, texts, y, device='cpu'):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import confusion_matrix

    tok = AutoTokenizer.from_pretrained(str(mdir))
    model = AutoModelForSequenceClassification.from_pretrained(str(mdir)).to(device)
    model.eval()

    all_probs = []
    batch = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt').to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    p = np.concatenate(all_probs, axis=0)
    acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
    out = {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}
    conf = confusion_matrix(y, p.argmax(1)).tolist()

    del model, tok; gc.collect()
    return out, conf, p


def strength_tag(strength):
    return 'bt' if strength is None else ('%02d' % round(strength * 100))


def run_cell(tag, model_key, attack, strength, device='cpu'):
    key = 'adv_%s_%s_%s_%s' % (tag, model_key, attack, strength_tag(strength))
    jpath, ppath = RESULTS_DIR / (key + '.json'), PROBS_DIR / (key + '.npz')
    if jpath.exists() and ppath.exists():
        rec = json.load(open(jpath))
        print('[skip] %-40s adv_f1=%.4f' % (key, rec['test']['f1']))
        return rec

    mdir = MODELS_DIR / ('%s_%s' % (tag, model_key))
    if not (mdir / 'model.safetensors').exists():
        print('[missing] no checkpoint at %s, skipping' % mdir)
        return None

    test_df = load_test_split(tag)
    raw_texts = list(test_df['text'])
    y = test_df['label'].values

    print('perturbing %d test texts: attack=%s strength=%s ...' % (len(raw_texts), attack, strength))
    t0 = time.time()
    texts = perturb_texts(raw_texts, attack, strength, device=device)
    perturb_secs = time.time() - t0

    print('scoring %s (trained on %s) under %s @ %s ...' % (model_key, DATASET_NAMES[tag], attack, strength))
    t0 = time.time()
    test_metrics, conf, p = score_checkpoint(mdir, texts, y, device=device)
    eval_secs = time.time() - t0

    out = {'key': key, 'dataset': tag, 'dataset_name': DATASET_NAMES[tag], 'model': model_key,
           'attack': attack, 'strength': strength, 'n_test': len(y),
           'perturb_seconds': round(perturb_secs, 1), 'eval_seconds': round(eval_secs, 1),
           'test': test_metrics, 'test_confusion': conf}
    atomic_write_npz(ppath, test_probs=p, test_labels=y)
    atomic_write_json(jpath, out)
    print('[done] %-40s adv_f1=%.4f  (%.1f min)' % (key, test_metrics['f1'], (perturb_secs + eval_secs) / 60))
    return out


def build_table():
    rows = []
    for tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            in_domain_files = sorted(RESULTS_DIR.glob(f'full_{tag}_{mk}_*_s42.json'))
            in_domain_f1, best_val = None, -1
            for f in in_domain_files:
                r = json.load(open(f))
                if r.get('val', {}).get('f1', -1) > best_val:
                    best_val = r['val']['f1']; in_domain_f1 = r['test']['f1']
            for attack, strengths in (('typo', STRENGTHS), ('homoglyph', STRENGTHS), ('backtranslation', (None,))):
                for strength in strengths:
                    key = 'adv_%s_%s_%s_%s' % (tag, mk, attack, strength_tag(strength))
                    p = RESULTS_DIR / (key + '.json')
                    adv_f1 = json.load(open(p))['test']['f1'] if p.exists() else None
                    rows.append({'dataset': DATASET_NAMES[tag], 'model': mk, 'attack': attack,
                                 'strength': strength, 'in_domain_f1': in_domain_f1, 'adv_f1': adv_f1,
                                 'drop': round(in_domain_f1 - adv_f1, 4) if (in_domain_f1 and adv_f1) else None})
    t = pd.DataFrame(rows)
    out_csv = FINAL_DIR.parent / 'table_adversarial_robustness.csv'
    t.to_csv(out_csv, index=False)
    print('\n' + t.to_string(index=False))
    print('\nwritten to:', out_csv)


if __name__ == '__main__':
    import torch
    device = os.environ.get('NLP_CROSS_DEVICE', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print('WARNING: cuda requested but not available, falling back to cpu')
        device = 'cpu'
    print('running adversarial robustness eval on device:', device,
          '(set NLP_CROSS_DEVICE=cuda to use the GPU once it is free)')

    # typo + homoglyph first (fast), backtranslation last per combo (slow: MT
    # model download + translation pass) -- so cheap results land first.
    for tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            for attack in ('typo', 'homoglyph'):
                for strength in STRENGTHS:
                    run_cell(tag, mk, attack, strength, device=device)
    for tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            run_cell(tag, mk, 'backtranslation', None, device=device)

    build_table()
