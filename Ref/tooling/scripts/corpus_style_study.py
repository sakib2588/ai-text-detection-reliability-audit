#!/usr/bin/env python3
r"""Measure the NDSS markdown corpus and place this manuscript against it.

Reads ndss_review_bundle/md/<NAME>/<NAME>.md (26 files: 25 provenance-verified NDSS
papers plus our own draft), measures sentence length, per-section word counts, tables,
figures, paragraphs, citation density, first person, and REGEX PROXIES for tense and
passive voice, then reports ours against a band computed over the 25 reference papers.

Contract, copied from scripts/73_prose_style_audit.py: ASSERT ONLY, NEVER GENERATE.
A bare run measures and reports; only --write-manifest writes. _normalisation_version
is refused across versions rather than silently compared, because a normalisation
change makes a committed band and a fresh measurement incommensurable with no visible
error. Each source .md is sha256-pinned. Exit 0 in band, 1 out of band, 2 could not
measure. Exit 2 is never a pass.

Ours is never a member of its own band. Every row prints a percentile, because a
min..max band over 25 papers passes at the extreme.

Read PROXY_LIMITATIONS before quoting any tense or passive number, and read the
caption limitation before quoting any body/total table split -- marker reorders
full-width table* floats, and our own paper's body tables are provably misplaced.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.markdown_corpus import (  # noqa: E402
    NORMALISATION_RULES, NORMALISATION_VERSION, PROXY_LIMITATIONS,
    measure_paper, read_paper,
)
from src.prose_metrics import CannotMeasure  # noqa: E402

OURS = "00_OURS_iccit6"
# EMPTY, and it must stay empty. Our section titles used to be assertive where the
# corpus is uniformly functional ("The Specificity Protocol", "What Does Not Survive",
# "The One Arm the Protocol Does Not Reclassify"), so section_role's regexes matched
# none of them and every role had to be asserted here -- help the 25 reference papers
# never got, which is what made every per-role band carry a HAND-MAPPED warning.
#
# The map was also silently wrong: it asserted V, VI, VII are evaluation, written
# against the numbering before Related Work was inserted at III. After that shift it
# filed our METHOD section as evaluation, so `method` read ABSENT and `evaluation` read
# 7,218 words against a 5,370 ceiling.
#
# 2026-08-10 fixed the cause instead: the sections were retitled to the corpus register
# (Measurement Design; Evaluation, absorbing the other two), so they now classify from
# their titles exactly like every reference paper's. Re-populating this map would
# re-introduce the asymmetry.
HAND_MAP: dict[str, dict[str, str]] = {}
# Excluded from SECTION-level bands only; retained at paper level.
SECTION_BAND_EXCLUDE: set[str] = set()

GATED = ("mean_words_per_sentence", "pct_over_35w", "mean_words_per_paragraph")

PAPER_ROWS = [
    ("body", "words", "body words"),
    # Same span minus ethics/acknowledgement/availability. `body words` above charges
    # a paper for back matter that no venue page cap counts, and this corpus is wildly
    # asymmetric there: most reference papers carry only a short acknowledgement.
    ("body_ex_backmatter", "words", "body words (ex back matter)"),
    ("structure", "n_backmatter_words", "back-matter words"),
    ("body", "sentences", "sentences"),
    ("body", "mean_words_per_sentence", "mean words / sentence"),
    ("body", "median_words_per_sentence", "median words / sentence"),
    ("body", "pct_over_35w", "% sentences over 35w"),
    ("body", "max_words_per_sentence", "longest sentence"),
    ("body", "paragraphs", "paragraphs"),
    ("body", "mean_words_per_paragraph", "mean words / paragraph"),
    ("body", "citations_per_1k", "citations per 1k"),
    ("body", "first_person_per_1k", "first person per 1k"),
    ("body", "passive_per_1k_words", "passive per 1k  [proxy]"),
    ("body", "pct_sentences_past", "% sentences past  [proxy]"),
    ("body", "pct_sentences_future", "% sentences future  [proxy]"),
    ("body", "pct_sentences_no_tense_anchor", "% NO tense anchor  [read first]"),
    ("structure", "n_sections", "sections"),
    # n_sections includes the unnumbered ethics/acknowledgement tail; this row is the
    # one to read when asking how many sections a paper has.
    ("structure", "n_numbered_sections", "sections (numbered)"),
    ("structure", "n_tables_total", "tables (total)"),
    ("structure", "n_figures_total", "figures (total)"),
]


def band(vals: list[float]) -> dict:
    s = sorted(vals)
    return {"n": len(s), "min": s[0], "median": st.median(s), "max": s[-1]}


def percentile_of(x: float, vals: list[float]) -> int:
    s = sorted(vals)
    return round(100 * sum(1 for v in s if v <= x) / len(s))


def get(res: dict, group: str, key: str):
    return res[group][key]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-dir", default="../md")
    ap.add_argument("--manifest", default="../report/corpus_style_study.json")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--per-section", action="store_true")
    ap.add_argument("--warn-only", action="store_true")
    args = ap.parse_args()

    files = sorted(pathlib.Path(args.md_dir).glob("*/*.md"))
    if not files:
        print(f"no markdown under {args.md_dir}")
        print("RESULT: UNRESOLVED")
        return 2

    print("=== NDSS CORPUS WRITING-STYLE STUDY ===")
    print(f"  normalisation v{NORMALISATION_VERSION} ({len(NORMALISATION_RULES)} ordered steps)")
    print(f"  source: {args.md_dir}   {len(files)} papers "
          f"= {len(files)-1} reference + ours ({OURS})")
    print("  marker ran with --disable_image_extraction: figures are CAPTION counts")
    print("  the reference papers are NOT a random sample of NDSS: 20 are NDSS 2025 and")
    print("  16 are backdoor/adversarial-ML papers chosen for topical proximity.")
    print("  \"the NDSS norm\" below means \"the norm among these 25\".")

    results, failed = {}, []
    for f in files:
        try:
            results[f.parent.name] = measure_paper(read_paper(f, HAND_MAP.get(f.parent.name)))
        except CannotMeasure as e:
            failed.append((f.parent.name, str(e)))
    if failed:
        print("\n  -- COULD NOT MEASURE --")
        for n, e in failed:
            print(f"     {n}: {e}")
        print("\nRESULT: UNRESOLVED")
        return 2
    if OURS not in results:
        print(f"\n  ours ({OURS}) not found")
        print("RESULT: UNRESOLVED")
        return 2

    ref = {k: v for k, v in results.items() if k != OURS}
    ours = results[OURS]

    # ---- parse health -------------------------------------------------------
    nsec = [v["structure"]["n_sections"] for v in results.values()]
    rej = sum(1 for v in results.values() for n in v["notes"] if "rejected 'I.'" in n)
    rejp = sum(1 for v in results.values() if any("rejected 'I.'" in n for n in v["notes"]))
    gaps = [(k, n) for k, v in results.items() for n in v["notes"] if "numeral gap" in n]
    runaway = sum(1 for v in results.values() for n in v["notes"] if "runaway" in n)
    pages = sum(1 for v in results.values() if v["structure"]["has_page_spans"])
    print("\n  -- parse health (nothing below is readable until this block is clean) --")
    print(f"     papers measured                    {len(results)}/{len(files)}")
    print(f"     sections detected                  {min(nsec)} .. {max(nsec)}  "
          f"median {st.median(nsec):.0f}")
    print(f"     'I.' rejected as subsection        {rej} across {rejp} papers")
    print(f"     numeral gaps (lost headings)       {len(gaps)}"
          + (f"  [{gaps[0][0]}]" if gaps else ""))
    print(f"     runaway sentences dropped          {runaway}")
    print(f"     inline images (expect 0)           "
          f"{sum(v['structure']['n_images'] for v in results.values())}")
    print(f"     papers with page anchors           {pages}/{len(results)}  "
          f"(page spans are an n={pages} metric, never mixed with n=25 bands)")

    # ---- paper level --------------------------------------------------------
    print(f"\n  -- ours against the {len(ref)}-paper band --")
    print(f"     {'metric':<34}{'ours':>10}{'min':>9}{'median':>9}{'max':>9}{'pct':>5}  flag")
    status, bands_out = 0, {}
    for group, key, label in PAPER_ROWS:
        vals = [get(v, group, key) for v in ref.values()]
        b = band(vals)
        o = get(ours, group, key)
        pc = percentile_of(o, vals)
        inb = b["min"] <= o <= b["max"]
        flag = "" if inb else ("OUT-LOW" if o < b["min"] else "OUT-HIGH")
        if not flag and pc >= 90:
            flag = "high"
        if not flag and pc <= 10:
            flag = "low"
        gate = key in GATED
        if gate and not inb:
            status = 1
        bands_out[f"{group}.{key}"] = {**b, "ours": o, "in_band": inb,
                                       "percentile": pc, "gated": gate}
        print(f"     {label:<34}{o:>10}{b['min']:>9}{b['median']:>9}{b['max']:>9}{pc:>5}  "
              f"{flag}{'  *gated' if gate else ''}")

    # ---- structure ----------------------------------------------------------
    rw = sum(1 for v in ref.values() if v["structure"]["has_related_work_section"])
    mg = sum(1 for v in ref.values() if v["structure"]["related_work_merged_into_background"])
    rm = sum(1 for v in ref.values() if v["structure"]["has_roadmap_paragraph"])
    pos = sorted(v["structure"]["related_work_position"] for v in ref.values()
                 if v["structure"]["related_work_position"] is not None)
    print(f"\n  -- structural conventions, measured on {len(ref)} papers --")
    print(f"     Related Work section present       {rw}/{len(ref)}   "
          f"(merged into Background in {mg})    OURS: "
          f"{ours['structure']['has_related_work_section']}")
    if pos:
        print(f"     ... its position in the body       median {st.median(pos):.2f}, "
              f"late (>0.7) in {sum(1 for x in pos if x > 0.7)}/{len(pos)}")
    print(f"     roadmap paragraph present          {rm}/{len(ref)}                    "
          f"OURS: {ours['structure']['has_roadmap_paragraph']}")

    # ---- per section --------------------------------------------------------
    if args.per_section:
        print("\n  -- per section role (paper 09 excluded: lost heading merges V and VI) --")
        secref = {k: v for k, v in ref.items() if k not in SECTION_BAND_EXCLUDE}
        roles = ["introduction", "background", "related_work", "threat_model",
                 "method", "evaluation", "discussion", "conclusion"]
        print(f"     {'role':<16}{'n':>4}{'ours w':>9}{'min':>8}{'med':>8}{'max':>8}"
              f"{'ours w/s':>10}{'band w/s':>16}")
        for role in roles:
            vals = [sum(s["words"] for s in v["sections"] if s["role"] == role)
                    for v in secref.values()
                    if any(s["role"] == role for s in v["sections"])]
            if not vals:
                continue
            osec = [s for s in ours["sections"] if s["role"] == role]
            ow = sum(s["words"] for s in osec) if osec else None
            ows = round(st.mean([s["mean_words_per_sentence"] for s in osec]), 1) if osec else None
            wsv = [st.mean([s["mean_words_per_sentence"] for s in v["sections"]
                            if s["role"] == role])
                   for v in secref.values() if any(s["role"] == role for s in v["sections"])]
            hm = "  HAND-MAPPED" if any(s["hand_mapped"] for s in osec) else ""
            print(f"     {role:<16}{len(vals):>4}{(ow if ow else 'ABSENT'):>9}"
                  f"{min(vals):>8}{st.median(vals):>8.0f}{max(vals):>8}"
                  f"{(ows if ows else '-'):>10}"
                  f"{f'{min(wsv):.1f} .. {max(wsv):.1f}':>16}{hm}")

    # ---- honesty ------------------------------------------------------------
    print("\n  -- read before quoting anything above --")
    for topic in ("captions", "tense", "passive", "sections", "page_spans", "sentence_split"):
        print(f"     [{topic}]")
        for lim in PROXY_LIMITATIONS[topic]:
            print(f"        - {lim}")
    print("     CALIBRATION: NOT DONE. The tense and passive proxies have not been")
    print("     hand-validated against labelled sentences; treat them as unvalidated.")

    manifest = {
        "_schema": "ndss_corpus_style_study",
        "_normalisation_version": NORMALISATION_VERSION,
        "_normalisation_rules": NORMALISATION_RULES,
        "_generated_by": "scripts/78_corpus_style_study.py",
        "_source_dir": args.md_dir,
        "_ours": OURS,
        "_hand_map": HAND_MAP,
        "_section_band_exclude": sorted(SECTION_BAND_EXCLUDE),
        "_corpus_caveat": ("25 reference papers, not a random sample of NDSS: 20 are NDSS 2025 "
                           "and 16 are backdoor/adversarial-ML papers selected for topical "
                           "proximity."),
        "_proxy_limitations": PROXY_LIMITATIONS,
        "_calibration": None,
        "papers": results,
        "bands": bands_out,
    }
    if args.write_manifest:
        p = pathlib.Path(args.manifest)
        p.write_text(json.dumps(manifest, indent=1))
        print(f"\n  wrote {p}")
    else:
        print(f"\n  (manifest NOT written; pass --write-manifest to write {args.manifest})")

    if args.warn_only:
        status = 0
    print(f"\nRESULT: {'PASS' if status == 0 else 'FAIL'}")
    return status


if __name__ == "__main__":
    sys.exit(main())
