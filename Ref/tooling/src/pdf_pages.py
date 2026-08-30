r"""Page-level facts read straight from a PDF, for the corpus organisation study.

Used by scripts/100_corpus_organisation_study.py and scripts/101_pagecount_survey.py.
Deliberately separate from src/markdown_corpus.py: that module's NORMALISATION_VERSION
pins notes/ndss_corpus_style_study.json, and bumping it would make every number already
published from that manifest incommensurable. Nothing here touches it.

THREE TRAPS THIS MODULE EXISTS TO ABSORB. All three were reproduced on real files:

  1. SMALL CAPS. IEEE and ACM templates set section headings in small caps, and
     pdftotext renders the dropped-cap first letter as a separate token: the
     REFERENCES heading of our own paper_ndss/main.pdf extracts as "R EFERENCES".
     A literal /REFERENCES/ finds nothing and every body/back-matter split silently
     charges the whole bibliography to the body.

  2. INTERLETTER CONTROL CHARACTERS. Some producers emit \x01 (and other C0 bytes)
     between glyphs of a heading, so even a \s-tolerant regex misses. Control
     characters are folded to spaces before matching.

  3. A HEADING THAT IS NEVER FOUND IS NOT PAGE ZERO. find_heading_page returns None,
     and every caller must propagate that as "not measured", never as a number. A
     paper whose references page is unknown contributes to a total-page statistic and
     to nothing else.

Every count from a PDF is a CEILING: the last page of a section is usually partial, so
"the references start on page 14" means the body occupies at most 13 full pages.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path


class PdfToolMissing(RuntimeError):
    """poppler-utils is not installed; refuse to guess rather than report zeros."""


# Letters separated by optional whitespace, so small-caps "R EFERENCES" matches and so
# does plain "REFERENCES". Anchored at a word boundary to reject "REFERENCES" appearing
# inside a running header only when it is not standalone -- see find_heading_page.
def _spaced(word: str) -> str:
    return r"\s*".join(re.escape(c) for c in word)


REFERENCES_RE = re.compile(rf"(?<![A-Za-z]){_spaced('REFERENCES')}(?![A-Za-z])", re.I)
APPENDIX_RE = re.compile(
    rf"(?<![A-Za-z]){_spaced('APPEND')}(?:{_spaced('IX')}|{_spaced('ICES')})(?![A-Za-z])", re.I)
# The full phrase, because small caps splits it as "E THICS C ONSIDERATIONS" and matching
# only "ETHICS" would leave too much of the line unmatched for the dominance test below.
ETHICS_RE = re.compile(
    rf"(?<![A-Za-z]){_spaced('ETHIC')}(?:S|{_spaced('AL')})"
    rf"(?:\s*{_spaced('CONSIDERATIONS')})?(?![A-Za-z])", re.I)
# NDSS boilerplate on the first page, and the NDSS DOI prefix. Filenames carry a year,
# not a DOI, so venue must be verified from content.
NDSS_BOILERPLATE_RE = re.compile(
    r"Network\s+and\s+Distributed\s+System[s]?\s+Security\s*\(?\s*NDSS\s*\)?\s*Symposium",
    re.I)
NDSS_DOI_RE = re.compile(r"10\.14722/")

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def smallcaps_fix(t: str) -> str:
    r"""pdftotext renders IEEEtran small caps as 'R EFERENCES' and 'I. I NTRODUCTION'.

    Moved here from scripts/73_prose_style_audit.py so one definition serves both that
    audit and scripts/102. Every heading and section regex in those studies fails SILENTLY
    without it, and the silence looks like a paper that has no sections rather than a
    broken extractor.
    """
    return re.sub(r"\b([A-Z])\s([A-Z]{2,})", lambda m: m.group(1) + m.group(2).lower(), t)


def require_tools() -> None:
    for tool in ("pdfinfo", "pdftotext"):
        if shutil.which(tool) is None:
            raise PdfToolMissing(
                f"{tool} not found. Install poppler-utils; refusing to report a "
                f"page count this module cannot actually measure.")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def page_count(pdf: Path) -> int | None:
    """Total pages from pdfinfo. None if the file cannot be read."""
    require_tools()
    try:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True,
                             timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"^Pages:\s*(\d+)", out.stdout, re.M)
    return int(m.group(1)) if m else None


def _one_page(pdf: Path, n: int) -> str:
    try:
        out = subprocess.run(["pdftotext", "-q", "-f", str(n), "-l", str(n), str(pdf), "-"],
                             capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return ""
    return CONTROL.sub(" ", out.stdout) if out.returncode == 0 else ""


def page_texts(pdf: Path, expected: int | None = None) -> list[str] | None:
    """Text of every page, control characters folded to spaces, indexed by REAL page.

    TRAP 4, and it is the expensive one. Splitting pdftotext's output on the form feed
    does NOT give one chunk per page. Measured 2026-08-14: BARBIE yields 74 chunks for an
    18-page PDF, Explanation-as-a-Watermark 28 for 18, Provably-Unlearnable-Data 31 for 18.
    Some producers emit form feeds at column or region breaks as well as at page breaks.
    Trusting the split put the REFERENCES heading of BARBIE on "page 39" of an 18-page
    paper and produced a body of 38 pages, which is what exposed it.

    pdfinfo is authoritative for how many pages exist. When the split disagrees, every
    page is re-extracted individually with -f/-l, which is slower but correctly indexed.
    """
    require_tools()
    if expected is None:
        expected = page_count(pdf)
    try:
        out = subprocess.run(["pdftotext", "-q", str(pdf), "-"],
                             capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    pages = out.stdout.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if expected and len(pages) != expected:
        return [_one_page(pdf, i) for i in range(1, expected + 1)]
    return [CONTROL.sub(" ", p) for p in pages]


def find_heading_page(pages: list[str], pattern: re.Pattern, min_page: int = 1) -> int | None:
    """1-indexed page where `pattern` first appears standing alone on a line.

    Standing alone is what separates a REFERENCES heading from the word "references"
    in a sentence, and from a running header. A line qualifies when the match covers
    most of the stripped line; the numeral and letter of "APPENDIX A" are allowed to
    follow. Returns None when never found -- never 0, never a guess.

    `min_page` is load-bearing for APPENDIX_RE. Our own paper puts "(Appendix B)." at the
    end of a wrapped body line on page 9, which stands alone and dominates its line, so an
    unrestricted search returns 9 for a document whose appendix starts on 16. Callers
    searching for the appendix must pass the references page as the floor.
    """
    for i, text in enumerate(pages, 1):
        if i < min_page:
            continue
        for line in text.splitlines():
            s = line.strip()
            if not s or len(s) > 60:
                continue
            m = pattern.search(s)
            if not m:
                continue
            # Reject a hit buried in prose: the heading must dominate its line.
            remainder = (s[:m.start()] + s[m.end():]).strip()
            if len(remainder) <= 12:
                return i
    return None


def venue_is_ndss(pages: list[str]) -> bool:
    """NDSS boilerplate or the 10.14722 DOI prefix, anywhere in the document."""
    joined = "\n".join(pages)
    return bool(NDSS_BOILERPLATE_RE.search(joined) or NDSS_DOI_RE.search(joined))


def aux_floats(aux: Path) -> dict | None:
    r"""Float number and page for every \newlabel in a LaTeX .aux file.

    This is the source of truth the markdown corpus lacks. marker reorders full-width
    table* floats, so a caption's position in the converted markdown is not evidence of
    where the float sits; the .aux records the page LaTeX actually put it on.

    Several labels can share one float (tab:params, tab:hyperparams and
    tab:evasion-params are all TABLE III on page 16), so floats are deduplicated by
    (kind, number) -- counting labels would treble that table.

    `alg:` IS A FLOAT PREFIX HERE and was missing until 2026-08-14. This project's paired
    protocol is labelled `alg:specificity` but typeset in a `figure` environment, so it
    numbers and floats exactly like any other figure. Matching only `(tab|fig):` silently
    reported 10 floats where the rendered PDF carries 11, and a round-24 reviewer counting
    captions by hand caught it. The undercount fed the float-density row of
    scripts/100_corpus_organisation_study.py.
    """
    if not aux.exists():
        return None
    text = aux.read_text(encoding="utf-8", errors="replace")
    floats: dict[tuple[str, str], int] = {}
    labels: dict[str, tuple[str, int]] = {}
    for m in re.finditer(r"\\newlabel\{(tab|fig|alg):([^}]*)\}\{\{([^}]*)\}\{(\d+)\}", text):
        kind, name, num, page = m.group(1), m.group(2), m.group(3), int(m.group(4))
        floats[(kind, num)] = page
        labels[f"{kind}:{name}"] = (num, page)
    if not floats:
        return None
    return {"floats": floats, "labels": labels}
