#!/usr/bin/env python3
"""Build the tables-only corpus report (markdown + PDF) from the study manifest.

Shape copied from ML_Paper/notes/20260810-report-corpus-tables-only.pdf: headings,
tables, and bold-led notes only. Ours is never inside its own band.
"""
from __future__ import annotations
import json, pathlib, statistics as st, subprocess, sys

REF = pathlib.Path(__file__).resolve().parents[2]
REPORT = REF / "report"
MANIFEST = REPORT / "corpus_style_study.json"
OUT_MD = REPORT / "corpus-tables-only.md"
OUT_PDF = REPORT / "corpus-tables-only.pdf"
OURS = "00_OURS_iccit6"

ROWS = [
    ("body.words", "body words"),
    ("body.sentences", "sentences"),
    ("body.mean_words_per_sentence", "mean words / sentence"),
    ("body.median_words_per_sentence", "median words / sentence"),
    ("body.pct_over_35w", "% sentences over 35w"),
    ("body.max_words_per_sentence", "longest sentence"),
    ("body.paragraphs", "paragraphs"),
    ("body.mean_words_per_paragraph", "mean words / paragraph"),
    ("body.citations_per_1k", "citations per 1k"),
    ("body.first_person_per_1k", "first person per 1k"),
    ("body.passive_per_1k_words", "passive per 1k [proxy]"),
    ("body.pct_sentences_past", "% sentences past [proxy]"),
    ("body.pct_sentences_no_tense_anchor", "% no tense anchor"),
    ("structure.n_sections", "sections"),
    ("structure.n_numbered_sections", "sections (numbered)"),
    ("structure.n_tables_total", "tables (total)"),
    ("structure.n_figures_total", "figures (total)"),
]
ROLES = ["introduction", "background", "related_work", "threat_model", "method",
         "evaluation", "discussion", "conclusion"]


def pages(pdf: pathlib.Path) -> str:
    try:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return line.split()[1]
    except Exception:
        pass
    return "?"


def table(head: list[str], rows: list[list[str]]) -> str:
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
         for i, h in enumerate(head)]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(cells)) + " |"
    sep = "|" + "|".join("-" * (x + 2) for x in w) + "|"
    return "\n".join([line(head), sep] + [line(r) for r in rows]) + "\n"


def main() -> int:
    d = json.loads(MANIFEST.read_text())
    papers, bands = d["papers"], d["bands"]
    ref = {k: v for k, v in papers.items() if k != OURS}
    ours = papers[OURS]
    md: list[str] = []
    A = md.append

    A("# Conference-paper corpus study, 11 six-page papers against our ICCIT draft\n")
    A("**How to read the columns.** *ours* is `paper/iccit6/main.pdf`, our six-page ICCIT "
      "submission. *min*, *median*, *max* are the band over the eleven reference papers; our "
      "paper is never a member of its own band, so every band here is n=11. *pct* is our "
      "percentile inside that band, printed on every row because a min..max band over eleven "
      "papers passes at the extreme. A flag of OUT-HIGH or OUT-LOW means outside the band "
      "entirely.\n")
    A("**What the corpus is, and what it is not.** Eleven papers of six pages or fewer, "
      "pulled from four of our other project reference folders (SAR drone, ML paper, XAI NIDS). "
      "They are not a sample of ICCIT and not a sample of any single venue: ten of the eleven "
      "were read from arXiv-hosted copies, and only one (Okabe, Interspeech 2018) carries a "
      "printed conference venue line. Topic spread is wide. Read a band as \"the norm among "
      "these eleven short papers\", never as \"the ICCIT norm\".\n")
    A("**What was changed in the instrument.** The measurement code is "
      "`scripts/78_corpus_style_study.py` and `src/markdown_corpus.py`, copied from the "
      "ML_Paper repository, where it was written for twelve-to-twenty page NDSS papers. Four "
      "changes were made and no measurement rule beyond them. Three are recalibrations for "
      "six-page papers, namely the body-word "
      "sanity window from 5,000..20,000 down to 1,500..20,000, the minimum detected sections "
      "from five to four, and the references-heading pattern now also accepts arabic numbering "
      "(`6. References`, the Interspeech style). Image markdown is stripped before measurement "
      "because this corpus was converted with figure extraction on, unlike the NDSS bundle. "
      "The fourth is a parsing fix found by reading the output rather than the page count, since the caption "
      "pattern allowed one page-anchor span before the label where marker sometimes emits two, "
      "which silently cost a figure. It now accepts any number of anchors, and every count in "
      "this report, corpus and ours alike, was recomputed under that rule.\n")

    # 1. roster
    A("\n## 1. The corpus roster\n")
    rows = []
    for name in [OURS] + sorted(ref):
        v = papers[name]
        pdf = (REF.parent / "paper" / "iccit6" / "main.pdf") if name == OURS else REF / f"{name}.pdf"
        rows.append([("OURS " if name == OURS else "") + name[:46], pages(pdf),
                     v["body"]["words"], v["body"]["sentences"],
                     v["structure"]["n_sections"], v["structure"]["n_tables_total"],
                     v["structure"]["n_figures_total"],
                     v["body"]["median_words_per_sentence"]])
    A(table(["paper", "pp", "body words", "sent", "sec", "tab", "fig", "med w/sent"], rows))
    A("**Body words means first numbered heading to REFERENCES.** Front matter and the "
      "bibliography are outside it, so a body-word figure is not a page count in words.\n")

    # 2. paper level
    A("\n## 2. Paper-level metrics, ours against the eleven-paper band\n")
    rows = []
    for key, label in ROWS:
        b = bands.get(key)
        if not b:
            continue
        flag = "" if b["in_band"] else ("OUT-LOW" if b["ours"] < b["min"] else "OUT-HIGH")
        if not flag and b["percentile"] >= 90:
            flag = "high"
        if not flag and b["percentile"] <= 10:
            flag = "low"
        rows.append([label, b["ours"], b["min"], b["median"], b["max"], b["percentile"], flag or "-"])
    A(table(["metric", "ours", "min", "median", "max", "pct", "flag"], rows))

    # 3. per-section budget
    A("\n## 3. Per-section word budget\n")
    rows = []
    for role in ROLES:
        vals = []
        for v in ref.values():
            w = sum(s["words"] for s in v["sections"] if s["role"] == role)
            if w:
                vals.append(w)
        o = sum(s["words"] for s in ours["sections"] if s["role"] == role)
        if not vals:
            rows.append([role, 0, o or "ABSENT", "-", "-", "-", "no reference paper has this role"])
            continue
        lo, hi, med = min(vals), max(vals), int(st.median(vals))
        if len(vals) < 3:
            verdict = f"no band, only {len(vals)} of 11 papers"
            if o:
                verdict += ", ours " + ("inside" if lo <= o <= hi else "outside") + " the pair"
            rows.append([role, len(vals), o if o else "ABSENT", lo, med, hi, verdict])
            continue
        if o == 0:
            verdict = "ABSENT in ours"
        elif o < lo:
            verdict = f"below floor by {lo - o}"
        elif o > hi:
            verdict = f"over ceiling by {o - hi}"
        else:
            verdict = f"in band, median {med}"
        rows.append([role, len(vals), o if o else "ABSENT", lo, med, hi, verdict])
    A(table(["role", "n papers", "ours", "min", "median", "max", "verdict"], rows))
    A("**A role carried by fewer than three of the eleven papers has no band.** Background "
      "appears in one reference paper and a threat model in two, so neither is a convention "
      "this corpus can be said to have, and their absence from our draft is not a violation. "
      "Every role that can be banded, and every role our draft carries, is inside its band.\n")
    A("**The conclusion is now its own section.** The draft previously ran one Discussion and "
      "Conclusion section, so every word of it was filed under discussion and the conclusion "
      "role read ABSENT against nine of eleven reference papers that carry a standalone one. "
      "Splitting the two puts both inside their bands and raises the section count to six, "
      "which is the corpus median.\n")
    A("**Roles are classified from section titles**, by the same regexes for every paper, "
      "ours included. A paper whose method section is titled after its artifact rather than "
      "after its function classifies elsewhere or nowhere, which is why the n column varies "
      "by role and why an ABSENT is a statement about titles, not about content.\n")

    # 4. structure conventions
    A("\n## 4. Structural conventions\n")
    rw = sum(1 for v in ref.values() if any(s["role"] == "related_work" for s in v["sections"]))
    rm = sum(1 for v in ref.values() if v["structure"]["has_roadmap_paragraph"])
    ps = sum(1 for v in ref.values() if v["structure"]["has_page_spans"])
    bw = sum(1 for v in ref.values() if v["structure"]["n_backmatter_words"] > 0)
    rows = [
        ["Related Work as its own section", f"{rw}/11",
         "yes" if any(s["role"] == "related_work" for s in ours["sections"]) else "no"],
        ["roadmap paragraph in the introduction", f"{rm}/11",
         "yes" if ours["structure"]["has_roadmap_paragraph"] else "no"],
        ["any back matter before REFERENCES", f"{bw}/11",
         "yes" if ours["structure"]["n_backmatter_words"] else "no"],
        ["marker page anchors present", f"{ps}/11",
         "yes" if ours["structure"]["has_page_spans"] else "no"],
    ]
    A(table(["convention", "corpus", "ours"], rows))

    # 5. sentence-length shape
    A("\n## 5. Sentence-length distribution\n")
    buckets = ["1-10", "11-20", "21-30", "31-40", "41-50", "51+"]
    rows = []
    for name in [OURS] + sorted(ref):
        h = papers[name]["body"]["sentence_length_histogram"]
        tot = sum(h.values()) or 1
        rows.append([("OURS " if name == OURS else "") + name[:40]]
                    + [f"{round(100 * h.get(b, 0) / tot)}%" for b in buckets])
    A(table(["paper"] + buckets, rows))
    A("**Percentages of that paper's own sentences.** The short-sentence column is where our "
      "draft separates from the corpus, and it is the same fact the mean and median "
      "words-per-sentence rows in Section 2 report.\n")

    # 6. limitations
    A("\n## 6. Read before quoting any number above\n")
    A("**Figure and table counts are caption counts.** They come from markdown produced by "
      "marker, so a caption lost in conversion is an undercount with no error raised. Numeral "
      "gaps are the only detector and are reported per paper in the manifest.\n")
    A("**The tense and passive columns are regex proxies, not parses.** No part-of-speech "
      "tagger was used. Past passive increments both the past and the passive counter, so the "
      "two are not independent and must never be summed. Between 18 and 54 percent of "
      "sentences carry no tense anchor at all in this corpus; read every tense figure against "
      "that share.\n")
    A("**Calibration was not done.** The proxies have not been hand-validated against labelled "
      "sentences here any more than they were in the NDSS study.\n")
    A(f"**Provenance.** Manifest `report/corpus_style_study.json`, normalisation v"
      f"{d['_normalisation_version']}, {len(d['_normalisation_rules'])} ordered cleaning steps. "
      f"Every source markdown file is sha256-pinned in the manifest, as is the PDF it was "
      f"converted from. Regenerate with `python3 scripts/corpus_style_study.py --md-dir ../md "
      f"--write-manifest` then `python3 scripts/build_report.py`.\n")

    OUT_MD.write_text("\n".join(md))
    cmd = ["pandoc", str(OUT_MD), "-o", str(OUT_PDF), "-V", "geometry:margin=2cm",
           "-V", "fontsize=9pt", "-V", "papersize=a4", "--toc", "-V", "colorlinks=true"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        return 1
    print(f"wrote {OUT_MD}\nwrote {OUT_PDF}  ({pages(OUT_PDF)} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
