#!/usr/bin/env bash
# Crash/teardown/power-cut recovery guardian for the NLP paper_scale jobs.
#
# Exists because all three jobs died twice on 2026-08-22 with no warning: once
# to a Claude Code session restart, once to load shedding, and both times every
# job had to be relaunched by hand. Cron is the only launch path on this box
# that survives both session teardown and a power cycle. Proven twice the same
# day by the sibling ML-paper guardian -- once relaunching a dead run 97 s after
# boot, once correctly declining to relaunch into a job it should not disturb.
#
# Acts ONLY when a job is not alive and not already complete. Never starts a
# second copy of anything.
set -uo pipefail

PS_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project /.venv/bin/python"
cd "$PS_" 2>/dev/null || exit 0          # drive not mounted yet -> next tick

GLOG="$PS_/logs/guardian_nlp.log"
mkdir -p "$PS_/logs"
log(){ echo "[$(date '+%F %H:%M:%S')] NLPGUARD: $*" >> "$GLOG"; }

# manual stop: `touch .nlp.stop` halts recovery; `rm .nlp.stop` resumes it
[ -f "$PS_/.nlp.stop" ] && exit 0

[ -x "$PY" ] || { log "venv python missing at $PY -- cannot recover"; exit 0; }

alive(){ pgrep -f "$1" >/dev/null 2>&1; }
count(){ ls -1 "$PS_/results"/$1 2>/dev/null | wc -l; }

# ------------------------------------------------------------ GPU, serialized --
# Exactly ONE GPU launch per tick, ever. Two trainers on this 8 GB card is the
# documented OOM. The chain has priority; Phase 3b fills the gap afterwards.
#
# Phase 3b (run_artifact_cleaning_full.py) needs only the card, not the chain's
# results, so it launches as soon as stage 1 (retrain, GPU) exits -- it does NOT
# wait for stage 2 (cross_dataset_eval_multiseed.py), which is CPU-only. That
# parallelism is what gets Phase 3b finished before the sibling ML-paper job's
# 04:00 resume floor instead of colliding with it.
chain_done(){
  local newest
  newest=$(ls -1t "$PS_"/logs/seed_robust_cross_*.log 2>/dev/null | head -1)
  [ -n "$newest" ] && grep -q 'CHAIN COMPLETE' "$newest"
}
# 4 cells: 2 datasets (D1c/D2c) x 2 models, seed 42 only
p3b_done(){ [ "$(count 'full_D?c_*_s42.json')" -ge 4 ]; }

gpu_busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
           | tr -d ',' | awk '$2>500{print $1}' | wc -l)
gpu_busy=${gpu_busy:-1}

if alive 'retrain_seeds_for_cross.py'; then
  :                                       # stage 1 training -- card is in use
elif alive 'run_artifact_cleaning_full.py'; then
  :                                       # Phase 3b running -- card is in use
elif [ "$gpu_busy" -ne 0 ]; then
  log "GPU busy (${gpu_busy} proc(s) >500MiB) -- no GPU launch this tick"
elif ! chain_done && ! alive 'run_seed_robust_cross_dataset_overnight.sh' \
     && ! alive 'cross_dataset_eval_multiseed.py'; then
  log "RECOVERY: GPU chain not alive, not complete, card free -- relaunching"
  setsid nohup bash "$PS_/run_seed_robust_cross_dataset_overnight.sh" \
    >/dev/null 2>&1 </dev/null &
  sleep 15
  alive 'run_seed_robust_cross_dataset_overnight.sh' \
    && log "recovery OK: GPU chain alive" \
    || log "recovery FAILED: GPU chain not alive 15 s after launch"
elif ! p3b_done; then
  log "LAUNCH: stage 1 done, card free, Phase 3b at $(count 'full_D?c_*_s42.json')/4 -- starting"
  setsid nohup "$PY" run_artifact_cleaning_full.py \
    >> "$PS_/logs/artclean_full_guardian.log" 2>&1 </dev/null &
  sleep 15
  alive 'run_artifact_cleaning_full.py' \
    && log "launch OK: Phase 3b alive" \
    || log "launch FAILED: Phase 3b not alive 15 s after launch"
fi

# ------------------------------------------------------------- CPU job: adv --
# 28 cells. CPU-only by design (device='cpu' unless NLP_CROSS_DEVICE is set,
# MarianMT backtranslation included), so it never needs a GPU check.
nAdv=$(count 'adv_*.json')
if alive 'run_adversarial_robustness.py'; then
  :
elif [ "$nAdv" -ge 28 ]; then
  :
else
  log "RECOVERY: adversarial down at ${nAdv}/28 -- relaunching"
  setsid nohup "$PY" run_adversarial_robustness.py \
    >> "$PS_/logs/adv_robustness_guardian.log" 2>&1 </dev/null &
  sleep 10
  alive 'run_adversarial_robustness.py' \
    && log "recovery OK: adversarial alive" \
    || log "recovery FAILED: adversarial not alive 10 s after launch"
fi

# -------------------------------------------------- CPU job: zero-shot clean --
nClean=$(count 'artclean_zeroshot_*.json')
if alive 'run_artifact_cleaning_zeroshot.py'; then
  :
elif [ "$nClean" -ge 8 ]; then
  :
else
  log "RECOVERY: zeroshot cleaning down at ${nClean}/8 -- relaunching"
  setsid nohup "$PY" run_artifact_cleaning_zeroshot.py \
    >> "$PS_/logs/artclean_zeroshot_guardian.log" 2>&1 </dev/null &
  sleep 10
  alive 'run_artifact_cleaning_zeroshot.py' \
    && log "recovery OK: zeroshot alive" \
    || log "recovery FAILED: zeroshot not alive 10 s after launch"
fi
