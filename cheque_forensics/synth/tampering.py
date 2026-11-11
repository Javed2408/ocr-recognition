"""Controlled cheque alterations with pixel-level ground truth, plus scan simulation.

Physical alterations (pen overwriting, added strokes, erasure + rewriting,
squeezed insertions) are applied before the simulated scan; digital
alterations (pasted patches, copied glyphs) after it, followed by a second
JPEG compression, as happens when a scanned image is edited.
"""
import copy
import datetime as dt
from dataclasses import dataclass

import cv2
import numpy as np

from ..amount_parser import number_to_indian_words, parse_numeric_amount
from ..schema import BBox
from . import signatures as sigs
from .cheque_generator import TEMPLATE_SPLITS, generate_cheque, place_signature, random_payee
from .render import composite_ink, random_writer, render_text

DIGIT_OVERWRITES = {"1": "74", "3": "8", "5": "86", "0": "869", "6": "8", "7": "9", "2": "3", "4": "9"}
STROKE_EDITS = {"1": "74", "0": "69", "3": "8"}
TY_WORDS = {"six": "ty", "seven": "ty", "nine": "ty", "eight": "y", "four": "ty"}
WORDS_ONLY_FAMILIES = ("kunstler", "palace", "french_script", "vladimir", "rage")


@dataclass
class TamperEvent:
    kind: str
    field: str
    bbox: BBox
    detail: str

    def to_dict(self):
        return {"kind": self.kind, "field": self.field, "bbox": self.bbox.to_dict(), "detail": self.detail}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_forger(cheque, rng, digits=True):
    """Who makes the alteration: often the same hand/pen family (hard), sometimes clearly different."""
    r = rng.random()
    base = cheque.writer if cheque.writer.family not in WORDS_ONLY_FAMILIES or not digits else cheque.payee_writer
    if r < 0.45:     # same writer style, possibly a different pen of the same colour
        return base.variant(rng, ink_shift=rng.random() < 0.6)
    return random_writer(rng, cheque.split, digits=digits,
                         ink_name=cheque.writer.ink_name if rng.random() < 0.75 else None)


def glyph_alpha(text, writer, rng, height, max_width=None):
    alpha, _, _ = render_text(text, writer, rng)
    scale = height / max(alpha.shape[0], 1)
    w = max(3, int(alpha.shape[1] * scale))
    if max_width:
        w = min(w, int(max_width))
    return cv2.resize(alpha, (w, max(3, int(height))), interpolation=cv2.INTER_AREA)


def ink_mask(img, paper, box, thresh=28):
    """Pixels inside `box` that carry ink (darker than the blank leaf)."""
    b = box.clip(img.shape[1], img.shape[0])
    diff = paper[b.y:b.y2, b.x:b.x2].astype(np.int16).sum(2) - img[b.y:b.y2, b.x:b.x2].astype(np.int16).sum(2)
    m = np.zeros(img.shape[:2], np.uint8)
    m[b.y:b.y2, b.x:b.x2] = (diff > thresh).astype(np.uint8)
    return m


def erase(img, paper, region_mask, rng):
    """Simulate chemical or mechanical erasure of the masked ink, leaving traces on the paper."""
    m = cv2.dilate(region_mask, np.ones((5, 5), np.uint8)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    clean = paper.astype(np.float32)
    if rng.random() < 0.5:   # chemical: bleached pattern, slight yellowing
        blurred = cv2.GaussianBlur(clean, (0, 0), 2.5)
        clean = blurred * 0.7 + clean * 0.3 + np.array([-6, 2, 6], np.float32)
        clean += rng.uniform(4, 12)
    else:                    # mechanical: abraded fibres, brighter, noisier
        clean = cv2.GaussianBlur(clean, (0, 0), 1.2) + rng.uniform(6, 16)
        clean += rng.normal(0, 5, clean.shape[:2])[..., None]
    ghost = rng.uniform(0.04, 0.18)                       # residual ink
    s = (m * rng.uniform(0.9, 1.0))[..., None]
    out = img.astype(np.float32) * (1 - s) + (clean * (1 - ghost) + img.astype(np.float32) * ghost) * s
    img[:] = np.clip(out, 0, 255).astype(np.uint8)


def _word_spans(text, char_boxes):
    """[(word, [BBox,...])] matching char boxes (which skip spaces) to words."""
    spans, i = [], 0
    for word in text.split():
        n = len(word)
        spans.append((word, [b for _, b in char_boxes[i:i + n]]))
        i += n
    return spans


def _union(boxes):
    return BBox.from_xyxy(min(b.x for b in boxes), min(b.y for b in boxes),
                          max(b.x2 for b in boxes), max(b.y2 for b in boxes))


def _median_h(boxes):
    return float(np.median([b.h for b in boxes]))


# ---------------------------------------------------------------------------
# Amount (figures)
# ---------------------------------------------------------------------------
def _digit_chars(cheque):
    return [(i, c, b) for i, (c, b) in enumerate(cheque.chars["amount_numeric"]) if c.isdigit()]


def amount_prefix_digit(cheque, img, content, rng):
    chars = cheque.chars["amount_numeric"]
    zone = cheque.fields["amount_numeric"]
    if not chars:
        return None
    first = chars[0][1]
    h = _median_h([b for c, b in chars if c.isdigit()])
    gap = first.x - zone.x
    if gap < h * 0.45:
        return None
    d = str(rng.integers(1, 10))
    forger = make_forger(cheque, rng)
    alpha = glyph_alpha(d, forger, rng, h * rng.uniform(0.9, 1.05), max_width=gap - 4)
    x = first.x - alpha.shape[1] - int(rng.uniform(1, 6))
    y = first.y2 - alpha.shape[0] + int(rng.normal(0, 2))
    box = composite_ink(img, alpha, x, y, forger.ink, forger.pressure, rng)
    content.amount_text = d + content.amount_text
    return TamperEvent("prefix_digit", "amount_numeric", box, f"digit '{d}' squeezed in before the amount")


def amount_append_digit(cheque, img, content, rng):
    chars = cheque.chars["amount_numeric"]
    zone = cheque.fields["amount_numeric"]
    digits = [b for c, b in chars if c.isdigit()]
    if not digits:
        return None
    h = _median_h(digits)
    forger = make_forger(cheque, rng)
    last_digit_idx = max(i for i, (c, _) in enumerate(chars) if c.isdigit())
    tail = chars[last_digit_idx + 1:]
    boxes = []
    if tail:   # remove the "/-", "=00" terminator first, then write "0" + terminator
        tail_box = _union([b for _, b in tail]).pad(2, 2)
        erase(img, cheque.paper, ink_mask(img, cheque.paper, tail_box), rng)
        boxes.append(tail_box)
        x = chars[last_digit_idx][1].x2 + int(rng.uniform(1, 5))
        term = content.amount_text[len(content.amount_text.rstrip("/-=.0")):] if not content.amount_text[-1].isdigit() else ""
        term = "/-" if "/" in content.amount_text[-3:] else ("=00" if "=" in content.amount_text else ".00")
        text = "0" + term
    else:
        x = chars[-1][1].x2 + int(rng.uniform(2, 8))
        text = "0"
    room = zone.x2 - x - 2
    if room < h * 0.5:
        return None
    alpha = glyph_alpha(text, forger, rng, h * rng.uniform(0.9, 1.05), max_width=room)
    y = digits[-1].y2 - alpha.shape[0] + int(rng.normal(0, 2))
    boxes.append(composite_ink(img, alpha, x, y, forger.ink, forger.pressure, rng))
    head = content.amount_text.rstrip("/-=.")
    head = head[:-2] if content.amount_text.endswith(("=00", ".00")) else head
    content.amount_text = head + text
    return TamperEvent("append_digit", "amount_numeric", _union([b for b in boxes if b]),
                       f"'{text}' written after the original figures")


def amount_overwrite_digit(cheque, img, content, rng, field="amount_numeric"):
    chars = cheque.chars[field]
    candidates = [(i, c, b) for i, (c, b) in enumerate(chars) if c in DIGIT_OVERWRITES]
    if field == "date":
        candidates = [(i, c, b) for i, c, b in candidates if i not in (4, 5)]   # keep "20" of the year plausible
    if not candidates:
        return None
    i, c, b = candidates[rng.integers(len(candidates))]
    new = DIGIT_OVERWRITES[c][rng.integers(len(DIGIT_OVERWRITES[c]))]
    forger = make_forger(cheque, rng)
    if field == "date":
        forger.size = cheque.writer.size
    alpha = glyph_alpha(new, forger, rng, b.h * rng.uniform(0.95, 1.1))
    if alpha.shape[1] < b.w * 0.6:   # cover the old digit's width
        alpha = cv2.resize(alpha, (int(b.w * rng.uniform(0.7, 1.0)), alpha.shape[0]))
    x = b.x + (b.w - alpha.shape[1]) // 2 + int(rng.normal(0, 1.5))
    y = b.y + (b.h - alpha.shape[0]) // 2 + int(rng.normal(0, 1.5))
    box = composite_ink(img, alpha, x, y, forger.ink, forger.pressure, rng)
    _replace_char(content, field, cheque, i, new)
    return TamperEvent("overwritten_digit", field, _union([box, b]), f"'{c}' overwritten as '{new}'")


def amount_add_stroke(cheque, img, content, rng, field="amount_numeric"):
    chars = cheque.chars[field]
    candidates = [(i, c, b) for i, (c, b) in enumerate(chars) if c in STROKE_EDITS]
    if field == "date":
        candidates = [(i, c, b) for i, c, b in candidates if i not in (4, 5)]
    if not candidates:
        return None
    i, c, b = candidates[rng.integers(len(candidates))]
    new = STROKE_EDITS[c][rng.integers(len(STROKE_EDITS[c]))]
    forger = make_forger(cheque, rng)
    canvas = np.zeros((b.h + 20, b.w + 40), np.uint8)
    ox, oy = 20, 10          # canvas offset of the char box
    t = max(2, int(round(b.h / 14)) + forger.thickness)
    if c == "1" and new == "7":
        pts = [(ox - int(b.w * 0.8), oy + 2), (ox + b.w, oy + int(rng.uniform(0, 3)))]
    elif c == "1" and new == "4":
        pts = [(ox + b.w // 2, oy + 2), (ox - int(b.w * 0.9), oy + int(b.h * 0.62)), (ox + b.w + 6, oy + int(b.h * 0.62))]
    elif c == "0" and new == "9":
        pts = [(ox + b.w - 2, oy + int(b.h * 0.5)), (ox + b.w - 1, oy + b.h + 6)]
    elif c == "0" and new == "6":
        pts = [(ox + 2, oy + int(b.h * 0.4)), (ox + int(b.w * 0.3), oy - 6), (ox + b.w, oy - 8)]
    else:   # 3 -> 8: close the left side with two arcs
        pts = [(ox + int(b.w * 0.45), oy + int(b.h * 0.5)), (ox + 1, oy + int(b.h * 0.25)), (ox + int(b.w * 0.4), oy + 2)]
        cv2.polylines(canvas, [np.array([(ox + int(b.w * 0.45), oy + int(b.h * 0.5)), (ox, oy + int(b.h * 0.75)),
                                         (ox + int(b.w * 0.4), oy + b.h - 1)], np.int32)], False, 255, t, cv2.LINE_AA)
    cv2.polylines(canvas, [np.array(pts, np.int32)], False, 255, t, cv2.LINE_AA)
    box = composite_ink(img, canvas.astype(np.float32) / 255.0, b.x - ox, b.y - oy, forger.ink, forger.pressure, rng)
    _replace_char(content, field, cheque, i, new)
    ys, xs = np.nonzero(canvas)
    stroke_box = BBox.from_xyxy(b.x - ox + xs.min(), b.y - oy + ys.min(), b.x - ox + xs.max() + 1, b.y - oy + ys.max() + 1)
    return TamperEvent("added_stroke", field, _union([stroke_box, b]), f"pen stroke added turning '{c}' into '{new}'")


def amount_erase_rewrite(cheque, img, content, rng, field="amount_numeric"):
    chars = cheque.chars[field]
    candidates = [(i, c, b) for i, (c, b) in enumerate(chars) if c.isdigit()]
    if field == "date":
        candidates = [(i, c, b) for i, c, b in candidates if i not in (4, 5)]
    if not candidates:
        return None
    i, c, b = candidates[rng.integers(len(candidates))]
    new = str((int(c) + int(rng.integers(1, 9))) % 10)
    erase(img, cheque.paper, ink_mask(img, cheque.paper, b.pad(2, 2)), rng)
    forger = make_forger(cheque, rng)
    alpha = glyph_alpha(new, forger, rng, b.h * rng.uniform(0.95, 1.05))
    x = b.x + (b.w - alpha.shape[1]) // 2
    box = composite_ink(img, alpha, x, b.y + int(rng.normal(0, 1.5)), forger.ink, forger.pressure, rng)
    _replace_char(content, field, cheque, i, new)
    return TamperEvent("erased_rewritten", field, _union([b.pad(3, 3), box]), f"'{c}' erased and rewritten as '{new}'")


def _replace_char(content, field, cheque, index, new):
    """Update the ground-truth content after changing the index-th written character."""
    if field == "amount_numeric":
        text, k = content.amount_text, -1
        out = []
        for ch in text:
            if ch != " ":
                k += 1
            out.append(new if k == index and ch != " " else ch)
        content.amount_text = "".join(out)
    elif field == "date":
        digits = list(content.date.strftime("%d%m%Y"))
        digits[index] = new
        s = "".join(digits)
        try:
            content.date = dt.date(int(s[4:]), int(s[2:4]), int(s[:2]))
        except ValueError:
            content.date_text_invalid = s


# ---------------------------------------------------------------------------
# Amount in words, payee
# ---------------------------------------------------------------------------
def words_append_suffix(cheque, img, content, rng):
    spans = _word_spans(content.words_text, cheque.chars["amount_words"])
    options = [(k, w, boxes) for k, (w, boxes) in enumerate(spans[:-1]) if w.lower() in TY_WORDS and boxes]
    if not options:
        return None
    k, word, boxes = options[rng.integers(len(options))]
    suffix = TY_WORDS[word.lower()]
    suffix = suffix.upper() if word.isupper() else suffix
    forger = make_forger(cheque, rng, digits=False)
    nxt = spans[k + 1][1]
    gap = (nxt[0].x - boxes[-1].x2) if nxt else 40
    h = _median_h(boxes) * 0.75
    alpha = glyph_alpha(suffix, forger, rng, h, max_width=max(8, gap - 2))
    x = boxes[-1].x2 + 1
    y = boxes[-1].y2 - alpha.shape[0]
    box = composite_ink(img, alpha, x, y, forger.ink, forger.pressure, rng)
    words = content.words_text.split()
    words[k] = word + suffix
    content.words_text = " ".join(words)
    return TamperEvent("squeezed_word_suffix", "amount_words", box, f"'{word}' extended to '{word + suffix}'")


def words_rewrite(cheque, img, content, rng, new_amount=None):
    """Erase the whole amount-in-words and write a new amount (consistent forgery when new_amount matches)."""
    zone = cheque.fields["amount_words"]
    erase(img, cheque.paper, ink_mask(img, cheque.paper, zone), rng)
    new_amount = new_amount or int(cheque.content.amount * rng.choice([2, 3, 5, 10]))
    text = number_to_indian_words(new_amount).title() + " Only"
    forger = make_forger(cheque, rng, digits=False)
    ink_box = cheque.ink_boxes["amount_words"]
    first_line_h = _median_h([b for _, b in cheque.chars["amount_words"]]) if cheque.chars["amount_words"] else 30
    alpha = glyph_alpha(text, forger, rng, first_line_h * 1.25, max_width=zone.w - 30)
    box = composite_ink(img, alpha, ink_box.x, ink_box.y, forger.ink, forger.pressure, rng)
    content.words_text = text
    return TamperEvent("erased_rewritten_words", "amount_words", _union([ink_box, box]),
                       f"amount in words erased and rewritten as '{text}'")


def payee_append(cheque, img, content, rng):
    boxes = [b for _, b in cheque.chars["payee"]]
    zone = cheque.fields["payee"]
    if not boxes:
        return None
    extra = [" & Co", " Kumar", " Traders", " Enterprises", " & Sons"][rng.integers(5)]
    forger = make_forger(cheque, rng, digits=False)
    x = boxes[-1].x2 + int(rng.uniform(6, 16))
    room = zone.x2 - x - 4
    h = _median_h(boxes) * rng.uniform(0.9, 1.1)
    if room < h:
        return None
    alpha = glyph_alpha(extra.strip(), forger, rng, h, max_width=room)
    y = int(np.median([b.y2 for b in boxes])) - alpha.shape[0] + int(rng.normal(0, 2))
    box = composite_ink(img, alpha, x, y, forger.ink, forger.pressure, rng)
    content.payee = content.payee + extra
    return TamperEvent("payee_appended", "payee", box, f"'{extra.strip()}' added after the payee name")


def payee_rewrite(cheque, img, content, rng):
    ink_box = cheque.ink_boxes["payee"]
    if ink_box is None:
        return None
    erase(img, cheque.paper, ink_mask(img, cheque.paper, ink_box.pad(3, 3)), rng)
    name = random_payee(rng)
    forger = make_forger(cheque, rng, digits=False)
    alpha = glyph_alpha(name, forger, rng, ink_box.h * rng.uniform(0.9, 1.05),
                        max_width=cheque.fields["payee"].x2 - ink_box.x - 10)
    box = composite_ink(img, alpha, ink_box.x, ink_box.y, forger.ink, forger.pressure, rng)
    content.payee = name
    return TamperEvent("payee_erased_rewritten", "payee", _union([ink_box, box]), f"payee erased and rewritten as '{name}'")


# ---------------------------------------------------------------------------
# Digital (post-scan) edits
# ---------------------------------------------------------------------------
def digital_glyph_copy(cheque, img, content, rng):
    """Copy a written digit (e.g. from the date) over a digit of the amount: copy-move forgery."""
    src = [(c, b) for c, b in cheque.chars["date"] if c.isdigit()]
    dst = _digit_chars(cheque)
    if not src or not dst:
        return None
    c, sb = src[rng.integers(len(src))]
    i, dc, db = dst[rng.integers(len(dst))]
    patch = img[sb.y - 2:sb.y2 + 2, sb.x - 2:sb.x2 + 2].copy()
    patch = cv2.resize(patch, (db.w + 4, db.h + 4), interpolation=cv2.INTER_LINEAR)
    y0, x0 = db.y - 2, db.x - 2
    img[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = patch[:img.shape[0] - y0, :img.shape[1] - x0]
    _replace_char(content, "amount_numeric", cheque, i, c)
    return TamperEvent("copied_glyph", "amount_numeric", db.pad(2, 2), f"digit '{dc}' replaced by a pasted copy of '{c}'")


def digital_splice(cheque, img, content, rng):
    """Paste a rectangular patch from another leaf (different paper/print) over part of a field."""
    field = ["amount_numeric", "amount_words", "payee", "date"][rng.integers(4)]
    ink_box = cheque.ink_boxes.get(field)
    if ink_box is None:
        return None
    other = generate_cheque(int(rng.choice(TEMPLATE_SPLITS[cheque.split])), np.random.default_rng(rng.integers(1 << 30)),
                            split=cheque.split)
    w = int(min(ink_box.w, rng.uniform(0.25, 0.6) * ink_box.w + 20))
    h = ink_box.h + 6
    ob = other.ink_boxes.get(field) or other.fields[field]
    sx = int(np.clip(ob.x + rng.uniform(0, max(1, ob.w - w)), 0, img.shape[1] - w))
    sy = int(np.clip(ob.y, 0, img.shape[0] - h))
    dx = int(np.clip(ink_box.x + rng.uniform(0, max(1, ink_box.w - w)), 0, img.shape[1] - w))
    dy = int(np.clip(ink_box.y - 3, 0, img.shape[0] - h))
    patch = other.image[sy:sy + h, sx:sx + w].astype(np.float32)
    patch *= rng.uniform(0.95, 1.05)    # mismatched exposure
    img[dy:dy + h, dx:dx + w] = np.clip(patch, 0, 255).astype(np.uint8)
    return TamperEvent("pasted_region", field, BBox(dx, dy, w, h), "rectangular region pasted from another document")


# ---------------------------------------------------------------------------
# Scan simulation and sample assembly
# ---------------------------------------------------------------------------
def jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def scan(img, rng, jpeg_prob=0.6):
    """Scanner/camera degradations. Returns (image, info)."""
    out = img.astype(np.float32)
    out = out * rng.uniform(0.92, 1.05) + rng.uniform(-8, 8)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.8))
    out += rng.normal(0, rng.uniform(0.5, 3.0), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    info = {"jpeg_quality": None}
    if rng.random() < jpeg_prob:
        q = int(rng.integers(70, 97))
        out = jpeg(out, q)
        info["jpeg_quality"] = q
    return out, info


PHYSICAL_OPS = {
    "amount_prefix_digit": (amount_prefix_digit, 3.0), "amount_append_digit": (amount_append_digit, 2.0),
    "amount_overwrite_digit": (amount_overwrite_digit, 3.0), "amount_add_stroke": (amount_add_stroke, 2.0),
    "amount_erase_rewrite": (amount_erase_rewrite, 2.0), "words_append_suffix": (words_append_suffix, 1.5),
    "words_rewrite": (words_rewrite, 1.0),
    "date_overwrite_digit": (lambda c, i, ct, r: amount_overwrite_digit(c, i, ct, r, field="date"), 1.5),
    "date_add_stroke": (lambda c, i, ct, r: amount_add_stroke(c, i, ct, r, field="date"), 1.0),
    "date_erase_rewrite": (lambda c, i, ct, r: amount_erase_rewrite(c, i, ct, r, field="date"), 1.0),
    "payee_append": (payee_append, 1.5), "payee_rewrite": (payee_rewrite, 1.5),
}
DIGITAL_OPS = {"digital_glyph_copy": (digital_glyph_copy, 1.0), "digital_splice": (digital_splice, 1.0)}


def _pick(ops, rng):
    names = list(ops)
    w = np.array([ops[n][1] for n in names])
    return names[rng.choice(len(names), p=w / w.sum())]


def make_sample(index, split, tamper_prob=0.5, seed=0, ops=None, forged_signature_prob=0.25,
                n_references=4, with_original=False):
    """One labelled cheque sample. Deterministic for a given (seed, split, index).

    Returns a dict with: image (scanned, possibly altered), label (1 = altered or forged
    signature), mask (uint8 0/1 of altered pixels), events, fields, content (final),
    original_content, signature info and registered reference signatures; plus
    `original` (an independent scan of the unaltered leaf) when with_original=True.
    """
    split_offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[split]
    rng = np.random.default_rng(seed * 7_919 + split_offset + index)
    template_id = int(rng.choice(TEMPLATE_SPLITS[split]))
    tampered = rng.random() < tamper_prob
    forge_sig = tampered and sigs.cedar_available() and rng.random() < forged_signature_prob and ops is None
    signature = None
    if sigs.cedar_available():
        wid = int(rng.choice(sigs.WRITER_SPLITS[split]))
        idx = int(rng.integers(1, sigs.SAMPLES_PER_WRITER + 1))
        signature = (wid, idx)
    cheque = generate_cheque(template_id, rng, split=split, signature=signature)
    references = []
    if signature:
        pool = [i for i in range(1, sigs.SAMPLES_PER_WRITER + 1) if i != signature[1]]
        references = [(signature[0], int(i)) for i in rng.choice(pool, n_references, replace=False)]
    clean = cheque.image.copy()
    if forge_sig:   # replace the signature with a skilled forgery of the account holder's signature
        img = cheque.image.copy()
        zone = cheque.fields["signature"]
        img[zone.y:zone.y2, zone.x:zone.x2] = cheque.paper[zone.y:zone.y2, zone.x:zone.x2]
        forged_idx = int(rng.integers(1, sigs.SAMPLES_PER_WRITER + 1))
        place_signature(img, zone, sigs.load_cedar(signature[0], forged_idx, True), cheque.writer.ink,
                        np.random.default_rng(rng.integers(1 << 30)))
        cheque.image = img
        cheque.signature = dict(cheque.signature, forged=True, index=forged_idx)
        clean = img.copy()   # the forged signature is not a pixel alteration of this leaf
    content = copy.deepcopy(cheque.content)
    img = cheque.image.copy()
    events = []
    if tampered:
        chosen = list(ops) if ops else []
        if not chosen and not forge_sig:
            chosen = [_pick(PHYSICAL_OPS, rng)]
            if rng.random() < 0.25:
                chosen.append(_pick(PHYSICAL_OPS, rng))
            if rng.random() < 0.15:
                chosen = [_pick(DIGITAL_OPS, rng)]
        elif not chosen and rng.random() < 0.3:
            chosen = [_pick(PHYSICAL_OPS, rng)]
        for name in [c for c in chosen if c in PHYSICAL_OPS]:
            ev = PHYSICAL_OPS[name][0](cheque, img, content, rng)
            if ev is not None:
                events.append(ev)
        if not events and not forge_sig and not any(c in DIGITAL_OPS for c in chosen):
            ev = amount_overwrite_digit(cheque, img, content, rng)   # always applicable fallback
            if ev is not None:
                events.append(ev)
    diff = np.abs(img.astype(np.int16) - clean.astype(np.int16)).max(2) > 10
    scanned, info = scan(img, rng)
    digital = [c for c in (ops or []) if c in DIGITAL_OPS] if ops else \
        ([c for c in chosen if c in DIGITAL_OPS] if tampered and not forge_sig else [])
    if digital:
        before = scanned.copy()
        for name in digital:
            ev = DIGITAL_OPS[name][0](cheque, scanned, content, rng)
            if ev is not None:
                events.append(ev)
                diff[ev.bbox.y:ev.bbox.y2, ev.bbox.x:ev.bbox.x2] |= (
                    np.abs(scanned.astype(np.int16) - before.astype(np.int16)).max(2) > 6
                )[ev.bbox.y:ev.bbox.y2, ev.bbox.x:ev.bbox.x2]
        q = int(rng.integers(75, 96))
        scanned = jpeg(scanned, q)
        info["second_jpeg_quality"] = q
    mask = cv2.dilate(diff.astype(np.uint8), np.ones((3, 3), np.uint8))
    label = int(bool(events) or forge_sig)
    sample = {
        "image": scanned, "label": label, "mask": mask, "events": events, "fields": cheque.fields,
        "content": content, "original_content": cheque.content, "signature": cheque.signature,
        "references": references, "template_id": template_id, "split": split, "scan": info,
        "writer_family": cheque.writer.family, "forged_signature": bool(forge_sig), "chars": cheque.chars,
    }
    if with_original:
        orng = np.random.default_rng(rng.integers(1 << 30))
        original, _ = scan(clean if not forge_sig else cheque.image, orng, jpeg_prob=0.3)
        # independent placement on the scanner bed: small rotation and shift
        m = cv2.getRotationMatrix2D((original.shape[1] / 2, original.shape[0] / 2), orng.uniform(-0.8, 0.8), 1.0)
        m[:, 2] += orng.uniform(-6, 6, 2)
        sample["original"] = cv2.warpAffine(original, m, (original.shape[1], original.shape[0]),
                                            borderMode=cv2.BORDER_REPLICATE)
    return sample


def amount_value(content):
    return parse_numeric_amount(content.amount_text)
