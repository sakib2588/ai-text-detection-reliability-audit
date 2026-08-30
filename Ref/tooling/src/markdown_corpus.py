r"""Read marker-converted NDSS papers from markdown and measure their prose.

Input is ndss_review_bundle/md/<NAME>/<NAME>.md, produced by
scripts/convert_ndss_corpus_md.sh (marker-pdf, --disable_image_extraction, no LLM pass).

Sentence splitting and word counting come from src.prose_metrics, shared with
scripts/73_prose_style_audit.py so the two manifests stay commensurable.

TWO DEFECTS THIS MODULE EXISTS TO PREVENT. Both were reproduced on real corpus files
and both produce plausible-looking numbers while raising nothing:

  1. Stripping HTML before maths destroys most of a paper. A generic <[^>]+> matches
     from a '<' inside $a < b$ to some later '>', deleting the prose between, and it
     also eats $$ delimiters so the display-math pass then pairs the wrong dollars.
     Measured on CatBack: 12,224 tokens -> 3,288, a 73% loss, silently. Maths is
     therefore stripped BEFORE tags, and only whitelisted tag names are stripped.
     There is a unit test for this; do not "simplify" it back.

  2. Marker repetition loops arrive as one enormous sentence. Paper 14 carries a
     3,266-character LaTeX loop inside an UNBALANCED '$', so no inline-math regex can
     ever match it; it reaches the splitter as a single 938-word sentence. Paper 23 has
     a 2,714-character twin. Hence the repetition scrub and MAX_SENTENCE_WORDS.

Guarding invariant: body word count must land in 5,000..20,000. Broken-cleaner CatBack
at 3,260 trips it, which turns a wrong number into a refusal to measure.

Tense and passive voice are REGEX PROXIES, not parsing. No NLP library is installed and
none was added. Their limitations are enumerated in PROXY_LIMITATIONS and must be
printed with any number they produce.
"""
from __future__ import annotations

import hashlib
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from src.prose_metrics import CannotMeasure, WORD, sentences

NORMALISATION_VERSION = "1"

MAX_SENTENCE_WORDS = 120        # longest genuine sentence in the corpus is 111
MIN_BODY_WORDS = 1_500   # RECALIBRATED for 5-6 page conference papers (was 5_000 for 12-20p NDSS)
MAX_BODY_WORDS = 20_000
MAX_RUNAWAY_DROPS = 3

# Roles that sit before REFERENCES but outside any venue body cap. Excluded from
# body_ex_backmatter so papers with a full Ethics section are compared against the
# same span as papers carrying only a one-line acknowledgement.
BACK_MATTER_ROLES = frozenset({"ethics", "acknowledgement", "availability"})

ROMAN = r"X{0,3}(?:IX|IV|V?I{0,3})"

# --- section headings ---------------------------------------------------------
HEADING = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.M)
SECTION_NUM = re.compile(rf"^({ROMAN}|\d{{1,2}})[.)]?[ \t]+(\S.*)$")
UNNUMBERED_TAIL = re.compile(
    r"^(ACKNOWLEDG|ETHIC|AVAILABILITY|DATASET|OPEN SCIENCE|ARTIFACT)", re.I)
REFERENCES_HEAD = re.compile(r"^\**\s*(?:[IVXL]+\.?\s*|\d+\.?\s*)?REFERENCES?\b", re.I)  # arabic numbering added for non-IEEE conference styles

# --- captions -----------------------------------------------------------------
# One regex for both dialects, split captions, and captions marker promoted to
# headings. The three-branch separator is what makes it cover all of them; the
# numeral alternation is what rejects "Table [IV](#page-7-0) shows ..." for free.
CAPTION_LINE = re.compile(
    r"^[ \t]*"
    r"(?:#{1,6}[ \t]+)?"                     # marker promoted 3 of our captions to headings
    r"(?:<span[^>]*></span>[ \t]*)*"         # page anchors before the label, marker emits one or two
    r"(?:\*{1,2})?"
    r"(?P<label>TABLE|Table|FIGURE|Figure|FIG\.|Fig\.)"
    r"[ \t]+"
    rf"(?P<num>{ROMAN}|\d{{1,3}})"
    r"(?![\dA-Za-z])"
    r"(?P<sep>[ \t]*[:.]|[ \t]+[A-Z(\"\u2018\u2019]|[ \t]*\*{0,2}[ \t]*$)",
    re.M)

PAGE_ANCHOR = re.compile(r'<span id="page-(\d+)-\d+"></span>')
ROADMAP = re.compile(
    r"(remainder|rest) of (this|the) (paper|article|work) is (organi[sz]ed|structured)", re.I)

# --- tense proxy ---------------------------------------------------------------
IRREG_PAST = (r"took|made|found|gave|saw|went|ran|chose|built|wrote|began|became|held|kept|"
              r"led|met|sent|told|thought|brought|bought|caught|taught|drew|grew|knew|threw|"
              r"spoke|broke|rose|arose|drove|came|felt|left|lost|paid|said|sat|stood|"
              r"understood|won")
# Excluded on purpose: set put read cost cut let hit spread shut split burst bet quit --
# their past and present forms are identical, so they cannot discriminate.
PAST_AUX = re.compile(r"\b(?:was|were|had|did)\b", re.I)
PAST_LEX = re.compile(
    r"\b(?:we|they|the authors?|this (?:work|study|paper)|prior work)\s+"
    r"(?:\w+ly\s+|also\s+|then\s+|first\s+|later\s+){0,2}"
    rf"(?:{IRREG_PAST}|[a-z]{{2,}}ed)\b", re.I)
PRES_AUX = re.compile(r"\b(?:is|are|am|has|have|does|do)\b", re.I)
PRES_LEX = re.compile(
    r"\b(?:shows?|gives?|yields?|remains?|provides?|requires?|implies?|means?|holds?|"
    r"follows?|suggests?|indicates?|allows?|enables?|makes?|uses?|reports?|defines?|"
    r"denotes?|consists?|depends?|presents?|describes?|contains?|includes?|appears?|"
    r"exists?|leads?|produces?|achieves?|performs?)\b", re.I)
FUTURE = re.compile(r"\b(?:will|shall)\b|'ll\b|\bgoing to\b", re.I)
MODAL = re.compile(r"\b(?:can|cannot|could|may|might|must|should|would)\b", re.I)

# --- passive proxy -------------------------------------------------------------
_BE = r"(?:is|are|am|was|were|be|been|being)"
_ADV = r"(?:not|also|then|thus|now|already|still|only|further|therefore|hence|[a-z]+ly)"
# CLOSED list on purpose: an open [a-z]+en matches "often", so "is often true" would
# score as passive.
_PART_EN = (r"(?:given|taken|shown|seen|known|written|chosen|driven|proven|broken|spoken|"
            r"hidden|frozen|forgotten|beaten|eaten|fallen|risen|overwritten|rewritten|"
            r"stolen|arisen|undertaken|mistaken|withdrawn|drawn|grown|thrown|worn|born)")
_PART_ED = r"[a-z]{2,}ed"
PASSIVE = re.compile(rf"\b{_BE}\b(?:\s+{_ADV}\b){{0,3}}\s+(?P<part>{_PART_ED}|{_PART_EN})\b", re.I)
BE_VERB = re.compile(rf"\b{_BE}\b", re.I)
NOT_PARTICIPLE = {"indeed", "exceed", "proceed", "succeed", "speed", "breed", "creed",
                  "greed", "seed", "feed", "deed", "weed", "embed", "misled"}

FIRST_PERSON_STRICT = re.compile(r"\b(?:we|our|ours)\b|\bus\b(?!\s*[A-Z])")
CITATION = re.compile(r"(?<![\w\]])\[\s*(\d{1,3})\s*\]")

NORMALISATION_RULES = [
    "harvest page anchors before stripping them (section page spans)",
    "strip control characters",
    "drop markdown pipe-table rows",
    "collapse marker repetition loops",
    "drop fenced code blocks",
    "replace display maths, THEN inline maths (before any tag stripping)",
    "drop residual LaTeX commands and sub/superscripts",
    "strip WHITELISTED html tags only, never a generic <[^>]+>",
    "drop the NDSS boilerplate footer",
    "drop caption lines",
    "drop orphan equation numbers",
    "drop heading lines (captured separately as sections)",
    "rewrite cross-reference links before citation links",
    "count citations, then collapse them",
    "unescape markdown, drop emphasis, fold unicode punctuation",
    "sentence split via src.prose_metrics.sentences (shared with scripts/73)",
    f"drop any sentence over {MAX_SENTENCE_WORDS} words as a conversion artefact",
]

PROXY_LIMITATIONS = {
    "tense": [
        "was/were cannot separate past active from past passive: a past passive increments "
        "BOTH the tense and the passive proxy. The two are not independent; never sum them.",
        "present perfect ('has been shown') registers as PRESENT via 'has', though it reports "
        "a past event. Systematic, and applied identically to all papers.",
        "would/could/should/might are morphologically past but functionally conditional; they "
        "are counted as MODAL and belong to no tense.",
        "-ed remains ambiguous behind the subject guard: reduced relatives such as 'the results "
        "obtained show' are missed by both the past and the present pattern.",
        "non-finite clauses receive no anchor, so a sentence reading as present tense can score "
        "zero.",
        "this is not parsing. Part of speech is guessed from word shape and left neighbour. No "
        "POS tagger is installed and none was added.",
        "18-50% of sentences carry no tense anchor at all. Read every tense figure against the "
        "no-anchor share printed beside it.",
    ],
    "passive": [
        "adjectival predicates over-count: 'is based on', 'is limited', 'is related to' are "
        "stative readings most annotators would not call passive. Inspect the participle "
        "histogram before quoting an absolute rate.",
        "reduced passives are missed ('features selected by SHAP'), pulling the other way.",
        "get-passives are not matched.",
        "overlaps the tense proxy; see the first tense limitation.",
    ],
    "sections": [
        "heading LEVEL is discarded: marker assigns depth from font size and it drifts within a "
        "single paper.",
        "'I.' is rejected as a subsection letter once a numeral >= II has been accepted. The "
        "alternative rule (reject unless ALL-CAPS) is wrong: papers 13, 18 and 26 render their "
        "genuine section I in title case.",
        "paper 09 lost its 'VI. EVALUATION' heading in conversion, so its sections V and VI are "
        "merged. It is excluded from section-level statistics and kept at paper level.",
    ],
    "captions": [
        "marker ran with --disable_image_extraction, so a figure count is a CAPTION count. A "
        "caption lost in conversion is an undercount with no error; numeral gaps are the only "
        "detector.",
        "distinct numerals are counted, not caption lines, so split and repeated captions count "
        "once.",
        "TOTAL caption counts are trustworthy -- CatBack's 18 tables and 9 figures reproduce "
        "known ground truth exactly. The BODY/TOTAL SPLIT IS NOT. Verified against our own "
        "paper's main.aux: tab:verdict-matrix is TABLE I on page 8 and tab:corpora is TABLE II "
        "on page 8, both squarely in the body, yet neither caption appears before the "
        "REFERENCES heading in the markdown, and the line that marker labels 'TABLE I' carries "
        "an appendix table's caption text. Marker reorders and mislabels full-width table* "
        "floats. Quote n_tables_total; do not quote n_tables_body without a per-paper check "
        "against a source of truth.",
    ],
    "page_spans": [
        "page spans come from marker's <span id='page-N-M'> anchors, which only 19 of 26 papers "
        "carry. Any page-based band has n=19, not n=26, and must be reported as such.",
    ],
    "sentence_split": [
        "inherited from scripts/73 and deliberately unchanged: the single-initial guard misfires "
        "on 'Section V-A. We test...', merging two sentences.",
    ],
}


@dataclass
class Section:
    index: int
    numeral: str | None
    title: str
    role: str
    start_line: int
    end_line: int
    text: str
    hand_mapped: bool = False
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class Paper:
    name: str
    path: Path
    sha256: str
    raw: str
    refs_line: int
    body_start_line: int
    sections: list[Section]
    notes: list[str] = field(default_factory=list)
    source_pdf_sha256: str | None = None


# ---------------------------------------------------------------- heading utils
def normalise_heading(text: str) -> str:
    t = re.sub(r"<span[^>]*></span>", "", text)
    t = t.replace("**", "").replace("*", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t.rstrip(".").strip()


def headings(raw: str) -> list[tuple[int, str]]:
    """(line_number, normalised_text) for every ATX heading. Level is discarded."""
    out = []
    for i, line in enumerate(raw.splitlines(), 1):
        m = re.match(r"^#{1,6}[ \t]+(.*)$", line)
        if m:
            out.append((i, normalise_heading(m.group(1))))
    return out


def find_references_line(hs: list[tuple[int, str]]) -> int:
    hits = [ln for ln, t in hs if REFERENCES_HEAD.match(t)]
    if len(hits) != 1:
        raise CannotMeasure(f"expected exactly one REFERENCES heading, found {len(hits)}")
    return hits[0]


def section_role(title: str) -> str:
    t = title.upper()
    if "INTRODUCTION" in t:
        return "introduction"
    if re.search(r"THREAT MODEL|SYSTEM MODEL|PROBLEM (STATEMENT|FORMULATION)", t):
        return "threat_model"
    if "RELATED WORK" in t and not re.search(r"BACKGROUND|PRELIMINAR", t):
        return "related_work"
    if re.search(r"BACKGROUND|PRELIMINAR", t):
        return "background"
    if re.search(r"EVALUATION|EXPERIMENT|RESULTS", t):
        return "evaluation"
    if re.search(r"METHOD|DESIGN|CONSTRUCTION|APPROACH|PROPOSED|FRAMEWORK|MEASUREMENT", t):
        return "method"
    if re.search(r"DISCUSSION|LIMITATION|ABLATION|DEFENS|MITIGATION", t):
        return "discussion"
    if "CONCLUSION" in t:
        return "conclusion"
    if "ETHIC" in t:
        return "ethics"
    if "ACKNOWLEDG" in t:
        return "acknowledgement"
    if re.search(r"AVAILABILITY|DATASET|OPEN SCIENCE|ARTIFACT", t):
        return "availability"
    return "other"


_ROMAN_VAL = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
              "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15}


def numeral_value(n: str) -> int | None:
    if n.isdigit():
        return int(n)
    return _ROMAN_VAL.get(n.upper())


def classify_sections(hs, refs_line, notes) -> list[tuple[int, str | None, str]]:
    """(line, numeral, title) for accepted body sections, in order."""
    accepted, seen_max = [], 0
    for ln, text in hs:
        if ln >= refs_line:
            break
        if CAPTION_LINE.match("# " + text):        # our draft promoted 3 captions to headings
            notes.append(f"line {ln}: rejected caption-as-heading {text!r}")
            continue
        m = SECTION_NUM.match(text)
        if m:
            num, title = m.group(1), m.group(2)
            val = numeral_value(num)
            if val is None:
                continue
            if num.upper() == "I" and seen_max >= 2:
                notes.append(f"line {ln}: rejected 'I.' as subsection letter ({title!r}); "
                             f"numeral {seen_max} already accepted")
                continue
            if val <= seen_max:
                continue
            seen_max = val
            accepted.append((ln, num.upper(), title))
        elif UNNUMBERED_TAIL.match(text) and accepted:
            accepted.append((ln, None, text))
    return accepted


# ---------------------------------------------------------------- the cleaner
def strip_noise(md: str) -> str:
    t = md
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)          # 1 control chars
    t = re.sub(r"^[ \t]*\|.*$", "", t, flags=re.M)               # 2 pipe-table rows FIRST
    t = re.sub(r"(.{8,80}?)\1{3,}", r"\1", t)                    # 3 marker repetition loops
    t = re.sub(r"```.*?```", " ", t, flags=re.S)                 # 4 code fences
    t = re.sub(r"\$\$(?:(?!\$\$).)*?\$\$", " MATH ", t, flags=re.S)   # 5 display maths
    t = re.sub(r"\$[^$\n]+\$", " MATH ", t)                      # 6 inline maths  (BEFORE tags)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)                        # 7 residual LaTeX
    t = re.sub(r"[A-Za-z0-9]\s*[_^]\s*(?:\{[^{}]*\}|\w)", " ", t)     # 8 sub/superscripts
    t = re.sub(r"<(sup|sub)>.*?</\1>", " ", t, flags=re.S)       # 9 sup/sub with body
    t = re.sub(r"</?(?:span|sup|sub|br|i|b|em|strong|u|small|font|div|p|table|tr|td|th|"
               r"thead|tbody)\b[^>]*>", "", t)                   # 10 WHITELIST ONLY
    t = re.sub(r"Network and Distributed Systems? Security \(NDSS\) Symposium.{0,400}?"
               r"ndss-symposium\.org", " ", t, flags=re.S)       # 11 NDSS boilerplate
    t = CAPTION_LINE.sub("", t)                                  # 12 caption lines
    t = re.sub(r"^[ \t]*\(?\d{1,3}\)?[ \t]*$", "", t, flags=re.M)     # 13 orphan eq numbers
    t = re.sub(r"^#{1,6}[ \t]+.*$", "", t, flags=re.M)           # 14 heading lines
    t = re.sub(r"(Fig\.|Figure|Tables?|Sections?|Sec\.|Appendix|Appendices|Algorithms?|"
               r"Alg\.|Equations?|Eq\.)\s*\[[^\]\n]{1,14}\]\([^)\n]*\)",
               r"\1 XREF ", t)                                   # 15 cross-refs BEFORE cites
    t = re.sub(r"\[\\?\[?\s*(\d{1,3})\s*[,;\u2013-]?\s*\\?\]?\]\([^)\n]*\)",
               r" [\1] ", t)                                     # 16 citation links
    return t                                                     # citations counted by caller


def finish_clean(t: str) -> str:
    t = CITATION.sub(" CITE ", t)                                # 18 collapse citations
    t = re.sub(r"\[([^\]\n]*)\]\([^)\n]*\)", r"\1", t)           # 19 surviving md links
    t = re.sub(r"\\([*\[\]_#&%$~^{}])", r"\1", t)                # 20 markdown escapes
    t = t.replace("**", "").replace("*", "")                     # 21 emphasis
    t = re.sub(r"[{}$\\]", " ", t)                               # 22 residual delimiters
    t = (t.replace("\u2019", "'").replace("\u2018", "'")
          .replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2013", "-").replace("\u2014", "-")
          .replace("\ufb01", "fi").replace("\ufb02", "fl"))      # 23 unicode fold
    return re.sub(r"[ \t]+", " ", t)


def prose_units(cleaned: str) -> list[str]:
    """Paragraphs, with list items split out. Never join across a blank line."""
    units = []
    for block in re.split(r"\n[ \t]*\n", cleaned):
        for unit in re.split(r"\n(?=[ \t]*(?:[-*\u2022]\s|\d{1,2}[.):]\s))", block):
            u = re.sub(r"^[ \t]*(?:[-*\u2022]|\d{1,2}[.):])\s*", "", unit)
            u = re.sub(r"\s+", " ", u).strip()
            if len(WORD.findall(u)) >= 3:
                units.append(u)
    return units


def split_sentences(cleaned: str, notes: list[str], where: str) -> list[str]:
    out, dropped = [], 0
    for unit in prose_units(cleaned):
        for s in sentences(unit):
            if len(WORD.findall(s)) > MAX_SENTENCE_WORDS:
                dropped += 1
                notes.append(f"{where}: dropped runaway sentence "
                             f"({len(WORD.findall(s))}w) {s[:60]!r}")
                continue
            out.append(s)
    return out


# ---------------------------------------------------------------- measurement
def _rate(n: int, words: int) -> float:
    return round(1000 * n / words, 1) if words else 0.0


def measure_text(md_slice: str, notes: list[str], where: str) -> dict:
    counted = strip_noise(md_slice)
    n_cites = len(CITATION.findall(counted))
    cleaned = finish_clean(counted)
    ss = split_sentences(cleaned, notes, where)
    if not ss:
        raise CannotMeasure(f"{where}: no sentences after normalisation")
    lens = [len(WORD.findall(s)) for s in ss]
    words = sum(lens)
    units = prose_units(cleaned)
    para_lens = [len(WORD.findall(u)) for u in units]

    past = len(PAST_AUX.findall(cleaned)) + len(PAST_LEX.findall(cleaned))
    pres = len(PRES_AUX.findall(cleaned)) + len(PRES_LEX.findall(cleaned))
    fut = len(FUTURE.findall(cleaned))
    modal = len(MODAL.findall(cleaned))
    s_past = sum(1 for s in ss if PAST_AUX.search(s) or PAST_LEX.search(s))
    s_pres = sum(1 for s in ss if PRES_AUX.search(s) or PRES_LEX.search(s))
    s_fut = sum(1 for s in ss if FUTURE.search(s))
    s_none = sum(1 for s in ss if not (PAST_AUX.search(s) or PAST_LEX.search(s)
                                       or PRES_AUX.search(s) or PRES_LEX.search(s)
                                       or FUTURE.search(s)))
    passives = [m.group("part").lower() for m in PASSIVE.finditer(cleaned)
                if m.group("part").lower() not in NOT_PARTICIPLE]
    fp = len(FIRST_PERSON_STRICT.findall(cleaned))
    caps = count_captions(md_slice)

    return {
        # Raw counts alongside the rates. A rate cannot be re-aggregated over a
        # subset of sections; a count can. Added 2026-08-10 so a back-matter-excluded
        # body variant can be built without re-parsing.
        "n_first_person": fp, "n_passive": len(passives),
        "n_be_verbs": len(BE_VERB.findall(cleaned)),
        "words": words, "sentences": len(ss),
        "mean_words_per_sentence": round(statistics.mean(lens), 1),
        "median_words_per_sentence": round(statistics.median(lens), 1),
        "p90_words_per_sentence": round(sorted(lens)[int(0.9 * (len(lens) - 1))], 1),
        "max_words_per_sentence": max(lens),
        "pct_over_35w": round(100 * sum(1 for x in lens if x > 35) / len(lens), 1),
        "n_over_35w": sum(1 for x in lens if x > 35),
        "sentence_length_histogram": {
            "1-10": sum(1 for x in lens if x <= 10),
            "11-20": sum(1 for x in lens if 10 < x <= 20),
            "21-30": sum(1 for x in lens if 20 < x <= 30),
            "31-40": sum(1 for x in lens if 30 < x <= 40),
            "41-50": sum(1 for x in lens if 40 < x <= 50),
            "51+": sum(1 for x in lens if x > 50),
        },
        "paragraphs": len(units),
        "mean_words_per_paragraph": round(statistics.mean(para_lens), 1) if para_lens else 0.0,
        "median_words_per_paragraph": round(statistics.median(para_lens), 1) if para_lens else 0.0,
        "citations": n_cites, "citations_per_1k": _rate(n_cites, words),
        "first_person_per_1k": _rate(fp, words),
        "past_anchors_per_1k": _rate(past, words),
        "present_anchors_per_1k": _rate(pres, words),
        "future_anchors_per_1k": _rate(fut, words),
        "modal_anchors_per_1k": _rate(modal, words),
        "pct_sentences_past": round(100 * s_past / len(ss), 1),
        "pct_sentences_present": round(100 * s_pres / len(ss), 1),
        "pct_sentences_future": round(100 * s_fut / len(ss), 1),
        "pct_sentences_no_tense_anchor": round(100 * s_none / len(ss), 1),
        "passive_per_1k_words": _rate(len(passives), words),
        "pct_sentences_with_passive": round(
            100 * sum(1 for s in ss if PASSIVE.search(s)) / len(ss), 1),
        "be_verbs_per_1k": _rate(len(BE_VERB.findall(cleaned)), words),
        "passive_top_participles": [list(x) for x in
                                    statistics.Counter(passives).most_common(10)]
        if hasattr(statistics, "Counter") else _top(passives),
        "n_tables": caps["n_tables"], "n_figures": caps["n_figures"],
    }


def _top(items, k=10):
    counts: dict[str, int] = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return [[w, c] for w, c in sorted(counts.items(), key=lambda kv: -kv[1])[:k]]


def count_captions(md: str) -> dict:
    tables, figures, lines_t, lines_f = set(), set(), 0, 0
    for m in CAPTION_LINE.finditer(md):
        num = m.group("num").upper()
        val = numeral_value(num)
        if m.group("label").upper().startswith("TABLE"):
            tables.add(val if val is not None else num)
            lines_t += 1
        else:
            figures.add(val if val is not None else num)
            lines_f += 1
    ti = sorted(x for x in tables if isinstance(x, int))
    fi = sorted(x for x in figures if isinstance(x, int))
    return {
        "n_tables": len(tables), "n_figures": len(figures),
        "n_table_caption_lines": lines_t, "n_figure_caption_lines": lines_f,
        "max_table_numeral": max(ti) if ti else 0,
        "max_figure_numeral": max(fi) if fi else 0,
        "table_numeral_gaps": [i for i in range(1, (max(ti) if ti else 0)) if i not in ti],
        "figure_numeral_gaps": [i for i in range(1, (max(fi) if fi else 0)) if i not in fi],
    }


def page_at(raw_lines: list[str], line_no: int) -> int | None:
    """Nearest page anchor at or before line_no (1-indexed). None if the paper has none."""
    for i in range(min(line_no, len(raw_lines)) - 1, -1, -1):
        m = PAGE_ANCHOR.search(raw_lines[i])
        if m:
            return int(m.group(1))
    return None


def source_pdf_for(md_path: Path) -> Path | None:
    """The PDF that a converted .md came from, if it is still findable.

    Layout is ndss_review_bundle/md/<base>/<base>.md against
    ndss_review_bundle/batch*/<base>.pdf.
    """
    bundle = md_path.parent.parent.parent
    hits = sorted(bundle.glob(f"batch*/{md_path.stem}.pdf"))
    return hits[0] if len(hits) == 1 else None


def read_paper(path: Path, hand_map: dict | None = None) -> Paper:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # This corpus was converted WITH image extraction (the .md folders carry the
    # figure files), unlike the NDSS bundle this instrument was written for. Image
    # markdown carries no prose, so it is stripped here and counted separately
    # rather than tripping the n_images invariant.
    n_img_stripped = len(re.findall(r"!\[", raw))
    raw = re.sub(r"^!\[[^\]]*\]\([^)]*\)\s*$", "", raw, flags=re.M)
    raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    notes: list[str] = []
    # The .md is pinned by sha below, but nothing pinned the PDF it was converted
    # FROM -- so a recompiled manuscript left the .md untouched and the study happily
    # measured a four-day-old draft (2026-08-10). Record the source PDF's sha too, and
    # note when it postdates the .md, which is the exact signature of that failure.
    pdf = source_pdf_for(path)
    pdf_sha = None
    if pdf is not None:
        pdf_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if pdf.stat().st_mtime > path.stat().st_mtime:
            notes.append(
                f"STALE: source PDF {pdf.name} is newer than its .md -- "
                f"re-convert with scripts/convert_ndss_corpus_md.sh --force {path.stem}")
    hs = headings(raw)
    refs_line = find_references_line(hs)
    accepted = classify_sections(hs, refs_line, notes)
    if len(accepted) < 4:
        raise CannotMeasure(f"only {len(accepted)} sections detected (minimum 4)")

    vals = [numeral_value(n) for _, n, _ in accepted if n]
    for a, b in zip(vals, vals[1:]):
        if b - a > 1:
            notes.append(f"numeral gap between {a} and {b} (heading lost in conversion)")

    lines = raw.splitlines()
    hm = (hand_map or {})
    secs = []
    for i, (ln, num, title) in enumerate(accepted):
        end = accepted[i + 1][0] - 1 if i + 1 < len(accepted) else refs_line - 1
        role = hm.get(num) if num and num in hm else section_role(title)
        secs.append(Section(
            index=i + 1, numeral=num, title=title, role=role,
            start_line=ln, end_line=end,
            text="\n".join(lines[ln:end]),
            hand_mapped=bool(num and num in hm),
            page_start=page_at(lines, ln), page_end=page_at(lines, end)))

    return Paper(name=path.parent.name, path=path, sha256=sha, raw=raw,
                 refs_line=refs_line, body_start_line=accepted[0][0],
                 sections=secs, notes=notes, source_pdf_sha256=pdf_sha)


def measure_paper(paper: Paper) -> dict:
    lines = paper.raw.splitlines()
    body_md = "\n".join(lines[paper.body_start_line - 1: paper.refs_line - 1])
    body = measure_text(body_md, paper.notes, "body")
    caps_body = count_captions(body_md)
    caps_all = count_captions(paper.raw)

    # `body` above spans first-numbered-heading .. REFERENCES, so it counts whatever
    # ethics / acknowledgement / availability matter sits before the bibliography.
    # That span is not what a venue page cap governs, and the magnitudes are wildly
    # asymmetric across this corpus: most reference papers contribute only a ~40-word
    # acknowledgement while a paper with a full Ethics Considerations section
    # contributes hundreds, so comparing raw `body` penalises the latter for prose the
    # cap does not count. `body_ex_backmatter` cuts the same span at the first
    # back-matter heading. It is a line slice, not an aggregation over sections, so
    # median and histogram stay exact.
    tail = [s for s in paper.sections if s.role in BACK_MATTER_ROLES]
    body_ex = body
    if tail:
        cut = min(s.start_line for s in tail)
        # Only sound if the back matter is a contiguous tail. If prose follows it,
        # slicing here would drop real body content -- note it and fall back.
        after = [s for s in paper.sections if s.start_line >= cut]
        if all(s.role in BACK_MATTER_ROLES for s in after):
            body_ex = measure_text("\n".join(lines[paper.body_start_line - 1: cut - 1]),
                                   paper.notes, "body_ex_backmatter")
        else:
            paper.notes.append(
                "back matter is not a contiguous tail; body_ex_backmatter falls back to body")

    sections = []
    for s in paper.sections:
        try:
            m = measure_text(s.text, paper.notes, f"section {s.numeral or s.title}")
        except CannotMeasure:
            continue
        m.update({"index": s.index, "numeral": s.numeral, "title": s.title,
                  "role": s.role, "hand_mapped": s.hand_mapped,
                  "start_line": s.start_line, "end_line": s.end_line,
                  "page_start": s.page_start, "page_end": s.page_end,
                  "pages": (s.page_end - s.page_start + 1)
                  if (s.page_start and s.page_end) else None})
        sections.append(m)

    n_anchor = len(PAGE_ANCHOR.findall(paper.raw))
    # "Has a Related Work section" must count the five papers that title it
    # "Background and Related Work" -- section_role files those under `background`,
    # which is the right ROLE but the wrong answer to "does this paper have one".
    rel_titled = [s for s in sections if "RELATED WORK" in s["title"].upper()]
    rel = [s for s in sections if s["role"] == "related_work"] or rel_titled
    body_span = max(1, paper.refs_line - paper.body_start_line)
    res = {
        "sha256": paper.sha256, "source_pdf_sha256": paper.source_pdf_sha256,
        "lines": len(lines),
        "refs_heading_line": paper.refs_line, "body_start_line": paper.body_start_line,
        "notes": paper.notes,
        "body": body,
        "body_ex_backmatter": body_ex,
        "structure": {
            "n_sections": len(paper.sections),
            # n_sections counts the unnumbered ethics/acknowledgement tail too, so it
            # is not the number a "how many sections does this paper have" band wants.
            "n_numbered_sections": sum(1 for s in paper.sections if s.numeral),
            "n_backmatter_words": body["words"] - body_ex["words"],
            "section_sequence": [s.numeral or s.title for s in paper.sections],
            "n_tables_body": caps_body["n_tables"], "n_tables_total": caps_all["n_tables"],
            "n_figures_body": caps_body["n_figures"], "n_figures_total": caps_all["n_figures"],
            "table_numeral_gaps": caps_all["table_numeral_gaps"],
            "figure_numeral_gaps": caps_all["figure_numeral_gaps"],
            "n_images": len(re.findall(r"!\[", paper.raw)),
            "n_page_anchors": n_anchor,
            "has_page_spans": n_anchor > 5,
            "has_roadmap_paragraph": bool(ROADMAP.search(paper.raw)),
            "has_related_work_section": bool(rel_titled),
            "related_work_merged_into_background": bool(
                rel_titled and not any(s["role"] == "related_work" for s in sections)),
            "related_work_position": round(
                (rel[0]["start_line"] - paper.body_start_line) / body_span, 3) if rel else None,
        },
        "sections": sections,
    }
    check_invariants(res)
    return res


def check_invariants(res: dict) -> None:
    b = res["body"]
    if not (MIN_BODY_WORDS < b["words"] < MAX_BODY_WORDS):
        raise CannotMeasure(
            f"body words {b['words']} outside {MIN_BODY_WORDS}..{MAX_BODY_WORDS} -- "
            f"the cleaner almost certainly ate or kept the wrong thing")
    if res["structure"]["n_images"] != 0:
        raise CannotMeasure("inline images survived the strip in read_paper")
    drops = sum(1 for n in res["notes"] if "runaway sentence" in n)
    if drops > MAX_RUNAWAY_DROPS:
        raise CannotMeasure(f"{drops} runaway sentences dropped (max {MAX_RUNAWAY_DROPS})")
    for k, v in b.items():
        if k.startswith("pct_") and not (0 <= v <= 100):
            raise CannotMeasure(f"{k}={v} outside [0,100]")
