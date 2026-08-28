#!/usr/bin/env bash
# ICCIT submission gate. Runs every hard constraint the paper must satisfy.
#   ./check.sh          -- check current artefacts
#   ./check.sh --build  -- rebuild first, then check
# Exit 0 only if every MUST gate passes. INFO lines are reported, not gated.
cd "$(dirname "$0")"
FAIL=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=1; }
info() { printf '  info  %s\n' "$1"; }

TEX="main.tex sections/00_abstract.tex sections/01_introduction.tex sections/02_methods.tex sections/03_results.tex sections/04_related_work.tex sections/05_discussion_conclusion.tex"

MANIFEST=".build-manifest"
build_inputs() {
  ls -1 $TEX bib/refs.bib 2>/dev/null
  ls -1 figures/*.png figures/*.pdf 2>/dev/null
}
write_manifest() { { build_inputs | sort | xargs -r sha256sum; sha256sum main.pdf; } > "$MANIFEST"; }

if [ "$1" = "--build" ]; then
  echo "== rebuild =="
  ./build.sh && write_manifest
fi

echo "== page budget =="
# RELAXED 2026-08-25 by user decision: this is the full-length cut, which carries
# every figure and table rather than the six that fit a conference budget. The
# page count is reported, not gated. Re-arm this gate before submitting to a
# venue with a hard cap by restoring the -le 6 test below.
PAGES=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}')
if [ -n "$PAGES" ]; then
  info "pages = $PAGES (not gated in the full-length cut; a 6-page venue cap would need a cut-down)"
else
  fail "pages = none (main.pdf did not build)"
fi
PAPER=$(pdfinfo main.pdf 2>/dev/null | awk -F'[()]' '/^Page size/{print $2}')
[ "$PAPER" = "A4" ] && pass "paper size = A4" || fail "paper size = ${PAPER:-unknown} (ICCIT requires the A4 template)"

if [ ! -f "$MANIFEST" ]; then
  fail "no build manifest -- run ./check.sh --build once to record source hashes"
else
  CHANGED=$(sha256sum -c --quiet "$MANIFEST" 2>/dev/null | sed 's/:.*//' | sort -u)
  LIST_NOW=$( { build_inputs; echo main.pdf; } | sort -u)
  LIST_WAS=$(awk '{print $2}' "$MANIFEST" | sort -u)
  SETDIFF=$(comm -3 <(echo "$LIST_NOW") <(echo "$LIST_WAS") | tr -d '\t' | sort -u | grep -c . || true)
  N=$(echo "$CHANGED" | grep -c . || true)
  if [ "${N:-0}" -eq 0 ] && [ "${SETDIFF:-0}" -eq 0 ]; then
    pass "main.pdf matches its sources (content hash)"
  elif [ "${SETDIFF:-0}" -ne 0 ]; then
    fail "build set changed, ${SETDIFF} file(s) added or removed -- run ./check.sh --build"
  elif [ "$CHANGED" = "main.pdf" ]; then
    fail "main.pdf was modified after the build, sources are unchanged -- rebuild or restore it"
  else
    fail "main.pdf is STALE, sources changed since the build: $(echo "$CHANGED" | grep -v '^main.pdf$' | tr '\n' ' ')-- run ./check.sh --build"
  fi
fi

echo "== references =="
BIB=$(grep -c '\\bibitem' main.bbl 2>/dev/null || true); BIB=${BIB:-0}
[ "$BIB" -ge 15 ] && pass "bibitems = $BIB (floor 15)" || fail "bibitems = $BIB (floor is 15)"
CITED=$(grep -ho '\\cite{[^}]*}' $TEX | sed 's/\\cite{//;s/}//' | tr ',' '\n' | sort -u | grep -c .)
info "unique \\cite keys in source = $CITED"
info "entries in bib/refs.bib = $(grep -c '^@' bib/refs.bib)"

echo "== build health =="
UNDEF=$(grep -icE 'undefined (reference|citation)|there were undefined' main.log 2>/dev/null || true); UNDEF=${UNDEF:-0}
[ "$UNDEF" -eq 0 ] && pass "undefined refs/cites = 0" || fail "undefined refs/cites = $UNDEF"
OVER=$(grep -c 'Overfull .hbox' main.log 2>/dev/null || true); OVER=${OVER:-0}
[ "$OVER" -eq 0 ] && pass "overfull hbox = 0" || fail "overfull hbox = $OVER"
BIBWARN=$(grep -c 'Warning' main.blg 2>/dev/null || true); BIBWARN=${BIBWARN:-0}
[ "$BIBWARN" -eq 0 ] && pass "bibtex warnings = 0" || fail "bibtex warnings = $BIBWARN"

echo "== banned glyphs (source) =="
for spec in "section-mark:$(printf '\xc2\xa7')" "pilcrow:$(printf '\xc2\xb6')" "dagger:$(printf '\xe2\x80\xa0')" "double-dagger:$(printf '\xe2\x80\xa1')"; do
  name=${spec%%:*}; glyph=${spec#*:}
  n=$(grep -oF "$glyph" $TEX 2>/dev/null | grep -c . || true)
  [ "${n:-0}" -eq 0 ] && pass "$name = 0" || fail "$name = $n"
done
LATEXDAG=$(grep -c 'dagger' $TEX 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
[ "$LATEXDAG" -eq 0 ] && pass "LaTeX \\dagger macros = 0" || fail "LaTeX \\dagger macros = $LATEXDAG"

echo "== banned glyphs (rendered PDF, catches figures) =="
if command -v pdftotext >/dev/null 2>&1; then
  TXT=$(pdftotext main.pdf - 2>/dev/null)
  for spec in "section-mark:$(printf '\xc2\xa7')" "pilcrow:$(printf '\xc2\xb6')" "dagger:$(printf '\xe2\x80\xa0')" "double-dagger:$(printf '\xe2\x80\xa1')"; do
    name=${spec%%:*}; glyph=${spec#*:}
    n=$(printf '%s' "$TXT" | grep -oF "$glyph" | grep -c . || true)
    [ "${n:-0}" -eq 0 ] && pass "rendered $name = 0" || fail "rendered $name = $n"
  done
else
  info "pdftotext unavailable, rendered-glyph check skipped"
fi

echo "== punctuation discipline =="
COLON=0
for f in sections/*.tex; do
  hits=$(sed -e 's/%.*$//' -e 's/\\label{[^}]*}//g' -e 's/\\\(auto\|c\|C\)\?ref{[^}]*}//g' \
             -e 's/\\cite[tp]\?{[^}]*}//g' -e 's|https\?://[^ }]*||g' "$f" \
         | grep -n ':' || true)
  if [ -n "$hits" ]; then
    while IFS= read -r line; do
      echo "        $f:$line"
      COLON=$((COLON+1))
    done <<< "$hits"
  fi
done
[ "$COLON" -eq 0 ] && pass "prose colons = 0" || fail "prose colons = $COLON (use a comma or rewrite)"
SEMI=$(grep -ho ';' sections/*.tex | grep -c . || true)
[ "${SEMI:-0}" -eq 0 ] && pass "prose semicolons = 0" || fail "prose semicolons = $SEMI"

echo "== double-blind =="
if grep -q '\\blindtrue' main.tex; then pass "\\blindtrue set"; else fail "\\blindtrue NOT set, ICCIT rejects non-anonymous submissions"; fi

echo
if [ "$FAIL" -eq 0 ]; then printf '\033[32mALL GATES PASS\033[0m\n'; else printf '\033[31mGATES FAILED\033[0m\n'; fi
exit $FAIL
