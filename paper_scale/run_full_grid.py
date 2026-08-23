"""Fills the remaining Table 1 cells: all 8 hyperparameter configurations per model per
dataset, at seed 42, full dataset scale. The 4 'winning' configs (already run by
run_full_scale.py, possibly at 3 seeds) are skipped automatically via the same
cache-check in run_one -- no wasted GPU time."""
import time
from run_full_scale import run_one, MODELS, DATASET_NAMES

GRID = [(2e-05,16,0.01), (3e-05,16,0.01), (2e-05,32,0.01), (3e-05,32,0.01),
        (2e-05,16,0.1),  (3e-05,16,0.1),  (2e-05,32,0.1),  (3e-05,32,0.1)]

# Work is split with a friend's machine (RTX 4060). The 4 already-completed/queued
# winner configs are excluded from both halves. The remaining 28 are interleaved by
# index across all 32 grid cells (not grouped by dataset/model) so each half gets a
# balanced mix of fast (BERT) and slow (DeBERTa) runs rather than one side getting
# stuck with all the expensive configurations.
_WINNER_CFGS = {('D1','BERT',3e-05,32,0.1), ('D1','DeBERTa',3e-05,16,0.01),
                ('D2','BERT',2e-05,16,0.1), ('D2','DeBERTa',3e-05,16,0.1)}
_ALL32 = [(tag, mk, lr, bs, wd) for tag in ('D1','D2') for mk in ('BERT','DeBERTa')
          for lr, bs, wd in GRID]
_REMAINING = [c for c in _ALL32 if c not in _WINNER_CFGS]
MY_HALF = _REMAINING[0::2]   # this machine's 14 assigned configs

def run_one_config(tag, model_key, lr, bs, wd, seed=42, verbose=True):
    """Same as run_one but takes lr/bs/wd directly rather than reading WINNERS."""
    import run_full_scale as m
    orig_winners = m.WINNERS
    m.WINNERS = {**orig_winners, (tag, model_key): dict(lr=lr, bs=bs, wd=wd)}
    try:
        return run_one(tag, model_key, seed, save_model=False, verbose=verbose)
    finally:
        m.WINNERS = orig_winners

if __name__ == '__main__':
    total = len(MY_HALF)
    print('running this machine\'s assigned half: %d of the 28 remaining configs' % total)
    done = 0
    t0 = time.time()
    for tag, mk, lr, bs, wd in MY_HALF:
        done += 1
        print('--- grid %d/%d  %s %s lr=%g bs=%d wd=%g ---' % (done, total, tag, mk, lr, bs, wd))
        run_one_config(tag, mk, lr, bs, wd, seed=42)
        e = time.time() - t0
        print('    elapsed %.1f min\n' % (e/60))
    print('full grid finished (this machine\'s half) in %.2f hours' % ((time.time()-t0)/3600))
