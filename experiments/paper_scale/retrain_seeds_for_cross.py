"""Retrains seeds 123 and 456 for the 4 winning configs WITH model-saving on.

Why this exists: run_full_scale.py's original 3-seed sweep (seeds 42/123/456) already
computed and saved metrics for all 3 seeds, but only saved model WEIGHTS for seed 42
(save_model=(seed==42)). cross_dataset_eval.py can therefore only ever evaluate seed
42's checkpoints -- the cross-dataset generalization gap has been reported from a
single seed, while every other headline result in this project is 3-seed. This script
closes that gap: it retrains the same 4 (dataset, model) configs at seeds 123 and 456,
this time saving the weights, so cross_dataset_eval_multiseed.py has 3 seeds' worth of
checkpoints to evaluate.

Metrics for these 8 runs will very likely come out numerically identical to the
already-saved seed 123/456 results (same seed, same fixed split, same everything) --
minor float differences are possible from cuDNN non-determinism and are not a concern.
The point of rerunning is the weights, not new numbers.

8 runs total (4 configs x 2 seeds). Resumable like every other script in this
project: if interrupted, just run again -- run_one's normal per-run checkpointing
inside a single run's training still applies even under force=True, since force only
bypasses the "whole run already done" skip, not the mid-training checkpoint resume.
"""
import time
from pathlib import Path
from run_full_scale import run_one, WINNERS, MODELS_DIR

RESEED = [123, 456]


def model_already_saved(tag, mk, seed):
    """force=True (needed because the metrics-only json from the original
    3-seed sweep already exists and would otherwise trigger run_one's skip
    before weights are saved) means this script is NOT idempotent against
    run_one's own skip check -- it would retrain from scratch every time it's
    relaunched, even for configs whose weights already landed on disk. Guard
    against that explicitly here instead."""
    mdir = MODELS_DIR / ('%s_%s_seed%d' % (tag, mk, seed))
    return (mdir / 'model.safetensors').exists()


if __name__ == '__main__':
    total = len(WINNERS) * len(RESEED)
    done = 0
    t_start = time.time()
    for (tag, mk) in WINNERS:
        for seed in RESEED:
            done += 1
            if model_already_saved(tag, mk, seed):
                print('[skip] retrain %d/%d  %s %s seed=%d -- weights already on disk' % (done, total, tag, mk, seed))
                continue
            print('--- retrain %d/%d  %s %s seed=%d (saving model this time) ---' % (done, total, tag, mk, seed))
            run_one(tag, mk, seed, save_model=True, force=True)
            e = time.time() - t_start
            print('    elapsed %.1f min, projected remaining %.1f min\n' % (e / 60, e / done * (total - done) / 60))
    print('RESEED-FOR-CROSS COMPLETE in %.2f hours' % ((time.time() - t_start) / 3600))
