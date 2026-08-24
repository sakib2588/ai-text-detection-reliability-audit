"""Shared text cleaning + adversarial perturbation functions for the artifact-
cleaning ablation (Gap 4) and adversarial robustness evaluation (Gap 2).

Grounded in the 2026-08-22 recon of the actual project data, not just the
literature's claims about HC3/DAIGT V2 in general:
  - HC3 whitespace-before-punctuation: human rows 88.7% (mean 10.7/doc) vs
    ChatGPT rows 0.28% (mean 0.009/doc), on the full 53,806-row corpus.
  - DAIGT V2 non-ASCII: a naive check is BACKWARDS (human 30.1% vs AI 11.1%)
    because human essays carry \xa0 / mojibake encoding noise (60,247
    instances), not signal. Excluding \xa0, non-ASCII flips to human 1.2%
    vs AI 11.1% (~9x), matching the literature direction. Emoji specifically
    is one-sided and clean: human 0/17,497 vs AI 562/17,497 (3.2%).

Pure functions only, no I/O -- imported by run_adversarial_robustness.py and
run_artifact_cleaning_zeroshot.py (and later run_artifact_cleaning_full.py).
"""
import random
import re
import unicodedata

# ---------------------------------------------------------------------------
# Cleaning (Gap 4)
# ---------------------------------------------------------------------------

# Mojibake byte sequences observed in the DAIGT V2 human-side text (recon,
# 2026-08-22) -- these are encoding noise from the PERSUADE corpus's source
# pipeline, not a label-correlated artifact, so they get normalized in BOTH
# the raw and cleaned conditions (see run_artifact_cleaning_zeroshot.py).
_NBSP_MAP = {
    '\xa0': ' ',
    '\x92': "'",
    '\x94': '"',
    'Â': '',   # stray Â from UTF-8-as-Latin-1 mojibake
    'Ã': '',   # stray Ã from UTF-8-as-Latin-1 mojibake
}


def normalize_nbsp(text):
    """Repair encoding noise (non-breaking spaces, mojibake) -- NOT part of
    the artifact being measured, applied identically to raw and cleaned
    conditions so the raw-vs-cleaned delta isolates label-correlated signal
    only, not encoding hygiene."""
    t = str(text)
    for bad, good in _NBSP_MAP.items():
        t = t.replace(bad, good)
    return t


_WS_BEFORE_PUNCT = re.compile(r' +([.,;:!?])')


def clean_hc3_whitespace(text):
    """Remove the space-before-punctuation artifact that near-perfectly
    separates HC3's human (88.7%) from ChatGPT (0.28%) answers."""
    return _WS_BEFORE_PUNCT.sub(r'\1', str(text))


# Emoji + common pictograph ranges -- matches the "emoji specifically" signal
# from recon (human 0%, AI 3.2% on DAIGT V2), not a blanket non-ASCII strip
# (which would remove legitimate accented characters, currency symbols, etc.
# unrelated to the artifact).
_EMOJI_RE = re.compile(
    '['
    '\U0001F300-\U0001FAFF'  # symbols & pictographs, supplemental, extended-A
    '\U00002600-\U000027BF'  # misc symbols, dingbats
    '\U0001F1E6-\U0001F1FF'  # regional indicators (flag emoji)
    ']',
    flags=re.UNICODE,
)


def clean_daigt_unicode(text):
    """Strip the emoji/pictograph signal that appears in 3.2% of AI-labeled
    DAIGT V2 essays and 0% of human-labeled ones. Call AFTER normalize_nbsp
    so encoding noise isn't conflated with this label-correlated signal."""
    return _EMOJI_RE.sub('', str(text))


def length_match(df, label_col='label', text_col='text', unit='chars', rng=None):
    """Length-match the two classes by truncating the longer-average class's
    texts to the shorter class's mean length (per-row truncation, not
    subsampling, so row count / split membership is unchanged -- important
    since split indices must stay reusable across raw/cleaned conditions).
    `unit='chars'` truncates by character count; `unit='words'` by whitespace
    token count. Returns a new column, does not mutate `df` in place."""
    rng = rng or random.Random(0)
    lengths = df[text_col].map(lambda t: len(str(t).split()) if unit == 'words' else len(str(t)))
    mean_by_label = lengths.groupby(df[label_col]).mean()
    target = int(mean_by_label.min())

    def _truncate(text):
        t = str(text)
        if unit == 'words':
            words = t.split()
            return ' '.join(words[:target]) if len(words) > target else t
        return t[:target] if len(t) > target else t

    return df[text_col].map(_truncate)


# ---------------------------------------------------------------------------
# Adversarial perturbations (Gap 2) -- applied to TEST TEXT ONLY, never train
# ---------------------------------------------------------------------------

_ADJACENT_KEYS = {
    'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr', 'f': 'dg', 'g': 'fh',
    'h': 'gj', 'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k', 'm': 'n', 'n': 'bm',
    'o': 'ip', 'p': 'o', 'q': 'wa', 'r': 'et', 's': 'ad', 't': 'ry', 'u': 'yi',
    'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu', 'z': 'x',
}


def inject_typos(text, rate, rng):
    """Character-level swap/delete/insert-adjacent-key at `rate` fraction of
    alphabetic characters, seeded by `rng` (a random.Random instance, caller-
    owned so results are reproducible across a run)."""
    chars = list(str(text))
    n_eligible = sum(1 for c in chars if c.isalpha())
    n_perturb = round(n_eligible * rate)
    eligible_idx = [i for i, c in enumerate(chars) if c.isalpha()]
    targets = set(rng.sample(eligible_idx, min(n_perturb, len(eligible_idx))))
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if i in targets:
            op = rng.choice(('swap', 'delete', 'insert'))
            if op == 'delete':
                i += 1
                continue
            elif op == 'insert' and c.lower() in _ADJACENT_KEYS:
                out.append(c)
                out.append(rng.choice(_ADJACENT_KEYS[c.lower()]))
            elif op == 'swap' and i + 1 < len(chars):
                out.append(chars[i + 1])
                out.append(c)
                i += 2
                continue
            else:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return ''.join(out)


_HOMOGLYPHS = {
    'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
    'x': 'х', 'y': 'у', 'i': 'і', 'A': 'А', 'E': 'Е',
    'O': 'О', 'P': 'Р', 'C': 'С', 'X': 'Х',
}  # Latin -> visually-identical Cyrillic lookalikes


def homoglyph_substitute(text, rate, rng):
    """Swap Latin characters for visually-identical Cyrillic lookalikes at
    `rate` fraction of eligible (homoglyph-mappable) characters. A human
    reader sees no difference; the tokenizer sees a different codepoint."""
    chars = list(str(text))
    eligible_idx = [i for i, c in enumerate(chars) if c in _HOMOGLYPHS]
    n_perturb = round(len(eligible_idx) * rate)
    targets = rng.sample(eligible_idx, min(n_perturb, len(eligible_idx)))
    for i in targets:
        chars[i] = _HOMOGLYPHS[chars[i]]
    return ''.join(chars)


# ---------------------------------------------------------------------------
# Back-translation (Gap 2) -- English -> German -> English round trip
# ---------------------------------------------------------------------------

_BT_MODELS = {}


def _get_bt_model(direction, device='cpu'):
    """direction: 'en-de' or 'de-en'. Lazily loads + caches MarianMT models
    (downloaded from Helsinki-NLP on first use, ~300MB each)."""
    if direction not in _BT_MODELS:
        from transformers import MarianMTModel, MarianTokenizer
        name = 'Helsinki-NLP/opus-mt-%s' % direction
        tok = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name).to(device)
        model.eval()
        _BT_MODELS[direction] = (tok, model)
    return _BT_MODELS[direction]


def _translate_batch(texts, direction, device='cpu', batch_size=16, max_len=256):
    import torch
    tok, model = _get_bt_model(direction, device)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(chunk, truncation=True, max_length=max_len, padding=True, return_tensors='pt').to(device)
            gen = model.generate(**enc, max_length=max_len)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def backtranslate(texts, device='cpu', batch_size=16):
    """Round-trip English -> German -> English paraphrase attack. `texts` is
    a list of strings; returns a list of paraphrased strings, same order."""
    german = _translate_batch(list(texts), 'en-de', device=device, batch_size=batch_size)
    english = _translate_batch(german, 'de-en', device=device, batch_size=batch_size)
    return english
