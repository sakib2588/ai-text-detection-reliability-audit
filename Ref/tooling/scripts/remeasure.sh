#!/usr/bin/env bash
# One iteration of the in-band loop: rebuild the paper, re-convert it, re-measure,
# rebuild the supervisor report, print the rows that have to land in band.
set -u
cd "$(dirname "$0")/.."           # Ref/tooling
ROOT="$(cd ../.. && pwd)"
PAPER="$ROOT/paper/${PAPER_DIR:-iccit6}"   # PAPER_DIR=iccit6_profstyle to measure the prof-style cut
MARKER=/media/filwel/MLProject1/research_project/ids-compression-benchmark/.venv/bin/marker_single
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "== build =="
"$PAPER/build.sh" || { echo "BUILD FAILED"; exit 2; }
PAGES=$(pdfinfo "$PAPER/main.pdf" | awk '/^Pages/{print $2}')
[ "$PAGES" = "6" ] || echo "  WARNING pages = $PAGES, the ICCIT cap is 6"

if [ "${1:-}" != "--no-marker" ]; then
  echo "== convert =="
  cp "$PAPER/main.pdf" /tmp/00_OURS_iccit6.pdf
  "$MARKER" /tmp/00_OURS_iccit6.pdf --output_dir ../md --output_format markdown \
    --layout_batch_size 2 --detection_batch_size 2 --recognition_batch_size 8 \
    --disable_multiprocessing >/tmp/iccit_marker.log 2>&1 || { echo "MARKER FAILED"; tail -3 /tmp/iccit_marker.log; exit 2; }
fi

echo "== measure =="
python3 scripts/corpus_style_study.py --md-dir ../md --write-manifest >/tmp/iccit_study.log 2>&1
python3 scripts/build_report.py >/dev/null 2>&1
python3 - <<'PY'
import json, pathlib
d = json.load(open("../report/corpus_style_study.json"))
b = d["bands"]
rows = [("body.words", "body words"), ("body.sentences", "sentences"),
        ("body.mean_words_per_sentence", "mean w/sentence"),
        ("body.median_words_per_sentence", "median w/sentence"),
        ("structure.n_figures_total", "figures"), ("structure.n_tables_total", "tables"),
        ("body.mean_words_per_paragraph", "mean w/paragraph"),
        ("body.pct_over_35w", "% over 35w"), ("body.max_words_per_sentence", "longest sentence"),
        ("body.passive_per_1k_words", "passive per 1k"),
        ("body.citations_per_1k", "citations per 1k")]
bad = 0
print(f"  {'row':<20}{'ours':>9}{'min':>9}{'median':>9}{'max':>9}   verdict")
for k, label in rows:
    v = b[k]
    if v["in_band"]:
        verdict = "in band"
    else:
        verdict = "OUT-LOW" if v["ours"] < v["min"] else "OUT-HIGH"
        bad += 1
    print(f"  {label:<20}{v['ours']:>9}{v['min']:>9}{v['median']:>9}{v['max']:>9}   {verdict}")
print(f"\n  OUT OF BAND: {bad}")
PY
