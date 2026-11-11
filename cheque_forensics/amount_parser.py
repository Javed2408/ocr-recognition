"""Parsing and formatting of Indian cheque amounts, amount-in-words and dates.

Reuses the number-word tables and words_to_number() from aiproject.py and adds
the Indian conventions: lakh/crore scales, 1,25,000 digit grouping, "/-" and
"=00" terminators, "Rupees ... Only" wording and OCR-tolerant word matching.
"""
import datetime as dt
import difflib
import re

import numpy as np

import aiproject

UNITS, TENS, SCALES = aiproject.UNITS, aiproject.TENS, aiproject.SCALES
NUMBER_WORDS = set(UNITS) | set(TENS) | set(SCALES)
# Spelling variants seen on cheques, mapped to the canonical word
VARIANTS = {
    "lac": "lakh", "lacs": "lakh", "lakhs": "lakh", "laks": "lakh", "lakhs.": "lakh",
    "crores": "crore", "cr": "crore", "thousands": "thousand", "hundreds": "hundred",
    "fourty": "forty", "ninty": "ninety", "hundered": "hundred", "thousend": "thousand",
}
FILLER_WORDS = {"rupees", "rupee", "rs", "only", "and", "inr", "paise", "paisa"}
UNIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
              "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
              "eighteen", "nineteen"]
TENS_NAMES = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


# ---------------------------------------------------------------------------
# Formatting (used by the synthetic cheque generator and the reports)
# ---------------------------------------------------------------------------
def _below_hundred(n):
    if n < 20:
        return UNIT_NAMES[n]
    return TENS_NAMES[n // 10] + ("" if n % 10 == 0 else " " + UNIT_NAMES[n % 10])


def _below_thousand(n):
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(UNIT_NAMES[hundreds] + " hundred")
    if rest:
        parts.append(_below_hundred(rest))
    return " ".join(parts)


def number_to_indian_words(n):
    """125000 -> 'one lakh twenty five thousand' (Indian numbering)."""
    n = int(n)
    if n == 0:
        return "zero"
    parts = []
    for scale, name in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        count, n = divmod(n, scale)
        if count:
            parts.append((number_to_indian_words(count) if count >= 100 else _below_hundred(count)) + " " + name)
    if n:
        parts.append(_below_thousand(n))
    return " ".join(parts)


def format_indian_number(n):
    """1250000 -> '12,50,000'"""
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def format_rupees(n):
    return "₹" + format_indian_number(n)


# ---------------------------------------------------------------------------
# Numeric amount box
# ---------------------------------------------------------------------------
# Characters OCR commonly confuses with digits in a numeric-only field
DIGIT_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "i": "1",
                                  "|": "1", "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8", "g": "9", "q": "9"})


def parse_numeric_amount(text):
    """Value of a handwritten amount-box string, or None.

    Handles '₹ 1,25,000/-', 'Rs.5000.00', '50,000=00', '5000/-' and OCR letter/digit
    confusions. Paise after '.', '=' or '-' are kept as the fractional part.
    """
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^(?:₹|rs\.?|inr|r5)\s*", "", t, flags=re.I)
    t = t.translate(DIGIT_CONFUSIONS)
    t = re.sub(r"\s+", "", t)
    m = re.match(r"^(\d[\d,]*)(?:[.=\-](\d{2}))?(?:/-|/=|-|=|/)?$", t)
    if not m:
        m = re.search(r"(\d[\d,]*\d|\d)", t)
        if not m:
            return None
        rupees, paise = m.group(1), None
    else:
        rupees, paise = m.group(1), m.group(2)
    digits = rupees.replace(",", "")
    if not digits or len(digits) > 12:
        return None
    return float(int(digits)) + (int(paise) / 100 if paise else 0.0)


STRICT_FIGURES = re.compile(r"^(?:\u20b9|rs\.?)?\d{1,3}(?:,\d{2,3})*(?:[.=]\d{2}|/-|/=)?$|^(?:\u20b9|rs\.?)?\d+(?:[.=]\d{2}|/-|/=)?$", re.I)


def figures_are_clean(text):
    """The figures read as a well-formed amount (digits, valid grouping, optional /- or =00)."""
    t = re.sub(r"\s+", "", text or "")
    return bool(STRICT_FIGURES.match(t)) and grouping_is_valid(re.split(r"[.=/]", t)[0])   # paise are not grouped


def words_reading_quality(text):
    """(share of word tokens recognised, mean match ratio); (1.0, 1.0) for a perfectly read amount in words."""
    tokens, unknown, ratio = normalise_amount_words(text, details=True)
    total = len(tokens) + len(unknown)
    return (len(tokens) / total if total else 0.0), ratio


def grouping_is_valid(text):
    """True if the commas follow Indian (12,34,567) or Western (1,234,567) grouping, or there are none."""
    t = re.sub(r"[^\d,]", "", (text or "").translate(DIGIT_CONFUSIONS))
    if "," not in t:
        return True
    return bool(re.fullmatch(r"\d{1,2}(?:,\d{2})*,\d{3}", t) or re.fullmatch(r"\d{1,3}(?:,\d{3})+", t))


# ---------------------------------------------------------------------------
# Amount in words
# ---------------------------------------------------------------------------
def _match(w, vocab, cutoff=0.75):
    w = VARIANTS.get(w, w)
    if w in NUMBER_WORDS or w in FILLER_WORDS:
        return w, 1.0
    if len(w) < 3:
        return None, 0.0
    m = difflib.get_close_matches(w, vocab, n=1, cutoff=cutoff)
    if not m:
        return None, 0.0
    return VARIANTS.get(m[0], m[0]), difflib.SequenceMatcher(None, w, m[0]).ratio()


def normalise_amount_words(text, fuzzy=True, details=False):
    """Lower-case number-word tokens with OCR-tolerant matching. Returns (tokens, unknown_tokens).

    A token that matches nothing is tried as two run-together words ('twothousand').
    With details=True also returns the mean match ratio (1.0 = every token read exactly).
    """
    tokens, unknown, ratios = [], [], []
    vocab = sorted(NUMBER_WORDS | FILLER_WORDS)
    for raw in re.findall(r"[a-zA-Z]+", text or ""):
        w = raw.lower()
        if fuzzy:
            word, ratio = _match(w, vocab)
        else:
            word, ratio = (w, 1.0) if w in NUMBER_WORDS | FILLER_WORDS else (None, 0.0)
        if word is None and fuzzy and len(w) >= 6:
            best = None
            for k in range(3, len(w) - 2):
                (a, ra), (b, rb) = _match(w[:k], vocab, 0.8), _match(w[k:], vocab, 0.8)
                if a and b and (best is None or ra + rb > best[2]):
                    best = (a, b, ra + rb)
            if best:
                tokens += [best[0], best[1]]
                ratios.append(best[2] / 2)
                continue
        if word is not None:
            tokens.append(word)
            ratios.append(ratio)
        else:
            unknown.append(raw)
    if details:
        return tokens, unknown, (float(np.mean(ratios)) if ratios else 0.0)
    return tokens, unknown


def parse_amount_words(text, fuzzy=True):
    """'Rupees One Lakh Twenty Five Thousand Only' -> 125000.0 (None if no number words).

    Paise written as '... and paise fifty' are added as the fractional part.
    """
    tokens, _ = normalise_amount_words(text, fuzzy)
    if not any(t in NUMBER_WORDS for t in tokens):
        return None
    if "paise" in tokens or "paisa" in tokens:
        cut = tokens.index("paise" if "paise" in tokens else "paisa")
        rupee_words = [t for t in tokens[:cut] if t in NUMBER_WORDS]
        paise_words = [t for t in tokens[cut + 1:] if t in NUMBER_WORDS]
        if not paise_words and cut > 0:   # "fifty paise" written before the word
            rupee_tail = tokens[:cut]
            last_and = max((i for i, t in enumerate(rupee_tail) if t == "and"), default=-1)
            if last_and >= 0:
                paise_words = [t for t in rupee_tail[last_and + 1:] if t in NUMBER_WORDS]
                rupee_words = [t for t in rupee_tail[:last_and] if t in NUMBER_WORDS]
        rupees = aiproject.words_to_number(rupee_words) if rupee_words else 0
        paise = aiproject.words_to_number(paise_words) if paise_words else 0
        return float(rupees) + paise / 100
    return float(aiproject.words_to_number([t for t in tokens if t in NUMBER_WORDS]))


def amount_words_are_well_formed(text):
    """Weak grammar check: scales must decrease (crore > lakh > thousand > hundred)."""
    tokens, _ = normalise_amount_words(text)
    order = [SCALES[t] for t in tokens if t in SCALES and t != "hundred"]
    return all(a > b for a, b in zip(order, order[1:]))


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------
def parse_cheque_date(text):
    """Date from 'DDMMYYYY' box digits, 'DD/MM/YYYY', 'DD-MM-YY' or 'DD.MM.YYYY'. Returns (date, error)."""
    if not text:
        return None, "no date text"
    t = text.translate(DIGIT_CONFUSIONS)
    digits = re.sub(r"\D", "", t)
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})", t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    elif len(digits) == 8:
        d, mo, y = int(digits[:2]), int(digits[2:4]), int(digits[4:])
    elif len(digits) == 6:
        d, mo, y = int(digits[:2]), int(digits[2:4]), int(digits[4:])
    else:
        return None, f"expected 8 date digits (DDMMYYYY), read {len(digits)}"
    if y < 100:
        y += 2000
    try:
        return dt.date(y, mo, d), None
    except ValueError as exc:
        return None, f"not a valid calendar date ({d:02d}-{mo:02d}-{y}): {exc}"


def date_validity(date, today=None, validity_months=3):
    """('valid' | 'post-dated' | 'stale', explanation) under the RBI three-month validity rule."""
    today = today or dt.date.today()
    if date > today:
        return "post-dated", f"dated {date:%d-%m-%Y}, which is after today ({today:%d-%m-%Y})"
    month = today.month - validity_months
    year = today.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(today.day, [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28,
                          31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    cutoff = dt.date(year, month, day)
    if date < cutoff:
        return "stale", f"dated {date:%d-%m-%Y}, more than {validity_months} months before today"
    return "valid", f"dated {date:%d-%m-%Y}, within the {validity_months}-month validity period"


# ---------------------------------------------------------------------------
# Printed codes
# ---------------------------------------------------------------------------
IFSC_RE = re.compile(r"[A-Z]{4}0[A-Z0-9]{6}")


def parse_ifsc(text):
    """IFSC is 4 letters, a zero, then 6 alphanumerics (e.g. ABCD0123456)."""
    t = re.sub(r"[^A-Za-z0-9]", "", (text or "").upper()).replace("IFSC", "").replace("1FSC", "")
    m = IFSC_RE.search(t)
    if m:
        return m.group(0)
    if len(t) >= 11:   # fix the common O/0 confusion at the fifth character
        cand = t[:4] + "0" + t[5:11]
        if IFSC_RE.fullmatch(cand):
            return cand
    return None


def parse_account_number(text):
    digits = re.sub(r"\D", "", (text or "").translate(DIGIT_CONFUSIONS))
    return digits if 9 <= len(digits) <= 18 else None


def parse_micr(text):
    """Split a CTS MICR line into cheque number (6), MICR code (9), account id (6) and transaction code (2)."""
    groups = re.findall(r"\d+", (text or "").translate(DIGIT_CONFUSIONS))
    out = {"cheque_number": None, "micr_code": None, "account_id": None, "transaction_code": None}
    for g in groups:
        if len(g) == 6 and out["cheque_number"] is None:
            out["cheque_number"] = g
        elif len(g) == 9 and out["micr_code"] is None:
            out["micr_code"] = g
        elif len(g) == 6 and out["account_id"] is None:
            out["account_id"] = g
        elif len(g) == 2 and out["transaction_code"] is None:
            out["transaction_code"] = g
    return out
