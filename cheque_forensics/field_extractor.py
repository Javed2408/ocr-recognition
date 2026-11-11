"""Cheque field detection, extraction and OCR.

Localisation combines three sources, from most to least specific:
  1. a U-Net field segmenter trained on many synthetic CTS-2010 layouts
     (pixel classes for the nine fields; boxes from the largest component),
  2. structural evidence found with OpenCV in the normalised cheque: ruled
     lines, the row of eight date boxes and the amount box rectangle, used to
     snap the learned boxes,
  3. CTS-2010 zone priors, used only when the first two find nothing.
Every field keeps its source and confidence so downstream modules can
discount uncertain localisations.
"""
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from . import amount_parser as ap
from .config import FIELD_SEG_SIZE, FIELD_SEG_WEIGHTS, FIELDS, MIN_TEXT_CONFIDENCE, SEG_BASE_CHANNELS
from .schema import BBox, FieldResult

# Typical CTS-2010 zones as fractions of the leaf (x1, y1, x2, y2)
FIELD_PRIORS = {
    "date": (0.69, 0.04, 0.98, 0.21),
    "payee": (0.08, 0.20, 0.88, 0.37),
    "amount_words": (0.03, 0.34, 0.76, 0.60),
    "amount_numeric": (0.73, 0.36, 0.97, 0.56),
    "account_number": (0.03, 0.60, 0.45, 0.76),
    "ifsc": (0.07, 0.14, 0.40, 0.23),
    "signature": (0.60, 0.58, 0.98, 0.84),
    "micr": (0.15, 0.86, 0.85, 0.98),
    "cheque_number": (0.15, 0.86, 0.36, 0.98),
}
MIN_COMPONENT_FRAC = 0.002   # of the segmentation map


# ---------------------------------------------------------------------------
# Learned segmentation
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _seg_model():
    from .device import load_checkpoint
    from .nets import UNet
    model = UNet(3, len(FIELDS) + 1, base=SEG_BASE_CHANNELS, depth=4)
    try:
        load_checkpoint(model, FIELD_SEG_WEIGHTS)
        return model
    except FileNotFoundError:
        return None


def segmentation_available():
    return _seg_model() is not None


def segment_fields(color):
    """Class probabilities (C, h, w) at the segmenter resolution, or None when untrained."""
    model = _seg_model()
    if model is None:
        return None
    import torch
    from .device import get_device
    h, w = FIELD_SEG_SIZE
    x = cv2.resize(color, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    with torch.no_grad():
        out = model(torch.from_numpy(x.transpose(2, 0, 1)[None]).to(get_device()))
        return out.softmax(1)[0].cpu().numpy()


def boxes_from_segmentation(probs, width, height):
    """{field: (BBox in canonical coords, confidence)} from the largest component of each class."""
    _, h, w = probs.shape
    labels = probs.argmax(0)
    sx, sy = width / w, height / h
    out = {}
    for ci, name in enumerate(FIELDS, start=1):
        m = (labels == ci).astype(np.uint8)
        if m.sum() < MIN_COMPONENT_FRAC * h * w:
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if n <= 1:
            continue
        mass = [probs[ci][lab == k].sum() for k in range(1, n)]
        k = int(np.argmax(mass)) + 1
        x, y, bw, bh = stats[k, :4]
        conf = float(probs[ci][lab == k].mean())
        out[name] = (BBox(int(x), int(y), int(bw), int(bh)).scale(sx, sy), conf)
    return out


def layout_score(color):
    """How well the segmenter's evidence matches the CTS layout; higher for upright cheques.
    Used as the 0/180-degree orientation scorer."""
    probs = segment_fields(color)
    if probs is None:
        return 0.0
    _, h, w = probs.shape
    scores = []
    for ci, name in enumerate(FIELDS, start=1):
        x1, y1, x2, y2 = FIELD_PRIORS[name]
        scores.append(probs[ci, int(y1 * h):int(y2 * h) + 1, int(x1 * w):int(x2 * w) + 1].mean())
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Structural evidence
# ---------------------------------------------------------------------------
@dataclass
class Structure:
    hlines: list          # (y, x1, x2)
    rects: list           # BBox
    date_row: BBox = None
    amount_box: BBox = None


def detect_structure(binary):
    H, W = binary.shape
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W // 18), 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    hlines = [(int(y + h / 2), int(x), int(x + w)) for x, y, w, h, _ in stats[1:] if w > W * 0.08 and h <= 8]
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(c) > 0.7 * w * h:
            rects.append(BBox(x, y, w, h))
    s = Structure(hlines=hlines, rects=rects)
    # date: a row of >= 6 small boxes of similar size, aligned
    small = [r for r in rects if W * 0.014 <= r.w <= W * 0.045 and H * 0.035 <= r.h <= H * 0.13 and r.y < H * 0.35]
    best = []
    for r in small:
        row = [q for q in small if abs(q.y - r.y) < 8 and abs(q.h - r.h) < 8]
        row = sorted({(q.x, q.y): q for q in row}.values(), key=lambda q: q.x)
        if len(row) > len(best):
            best = row
    if len(best) >= 6:
        s.date_row = BBox.from_xyxy(best[0].x, min(q.y for q in best), best[-1].x2, max(q.y2 for q in best))
    big = [r for r in rects if W * 0.12 <= r.w <= W * 0.38 and H * 0.06 <= r.h <= H * 0.2 and r.x > W * 0.5
           and H * 0.2 < r.y < H * 0.7]
    if big:
        s.amount_box = max(big, key=lambda r: r.area)
    return s


def refine_box(name, box, structure, W, H):
    """Snap a learned/prior box to structural evidence. Returns (box, snapped?)."""
    if name == "date" and structure.date_row is not None and \
            (box.iou(structure.date_row) > 0.05 or structure.date_row.overlap_fraction(box) > 0.3):
        return structure.date_row.pad(3, 3), True
    if name == "amount_numeric" and structure.amount_box is not None and \
            (box.iou(structure.amount_box) > 0.1 or structure.amount_box.overlap_fraction(box) > 0.3):
        r = structure.amount_box
        cut = int(r.h * 0.45)          # the printed rupee sign sits at the left of the box
        return BBox(r.x + cut, r.y + 2, r.w - cut - 2, r.h - 4), True
    if name in ("payee", "amount_words"):
        lines = [l for l in structure.hlines if box.x - 20 < (l[1] + l[2]) / 2 < box.x2 + 20
                 and abs(l[0] - box.y2) < max(25, box.h * 0.35)]
        if lines:
            y = max(l[0] for l in lines)
            return BBox(box.x, box.y, box.w, max(10, y + 4 - box.y)), True
    return box, False


def prior_box(name, W, H):
    x1, y1, x2, y2 = FIELD_PRIORS[name]
    return BBox.from_xyxy(x1 * W, y1 * H, x2 * W, y2 * H)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def handwriting_ink_ratio(crop):
    """Share of the crop covered by ink once ruled lines and box borders are removed."""
    g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 18)
    for k in ((max(15, g.shape[1] // 5), 1), (1, max(15, int(g.shape[0] * 0.7)))):
        ink &= ~cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    return float(np.count_nonzero(ink)) / ink.size


def parse_value(name, text):
    if name == "amount_numeric":
        return ap.parse_numeric_amount(text)
    if name == "amount_words":
        return ap.parse_amount_words(text)
    if name == "date":
        return ap.parse_cheque_date(text)[0]
    if name == "ifsc":
        return ap.parse_ifsc(text)
    if name == "account_number":
        return ap.parse_account_number(text)
    if name == "micr":
        return ap.parse_micr(text)
    if name == "cheque_number":
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits[:6] if len(digits) >= 6 else None
    if name == "payee":
        return " ".join(text.split()) or None
    return None


def locate_fields(pre, use_model=True):
    """{name: (BBox, confidence, source)} in canonical coordinates."""
    H, W = pre.color.shape[:2]
    learned = {}
    if use_model:
        probs = segment_fields(pre.color)
        if probs is not None:
            learned = boxes_from_segmentation(probs, W, H)
    structure = detect_structure(pre.binary)
    out = {}
    for name in FIELDS:
        if name in learned:
            box, conf = learned[name]
            source = "segmentation"
        else:
            box, conf, source = prior_box(name, W, H), 0.3, "prior"
        box, snapped = refine_box(name, box, structure, W, H)
        if snapped:
            source += "+structure"
            conf = min(1.0, conf + 0.1)
        out[name] = (box.clip(W, H), conf, source)
    return out, structure


def extract_fields(pre, ocr, use_model=True, pad=4):
    """Detect, crop and read all fields. Returns ({name: FieldResult}, Structure)."""
    located, structure = locate_fields(pre, use_model)
    H, W = pre.color.shape[:2]
    results = {}
    for name, (box, conf, source) in located.items():
        fr = FieldResult(name=name, bbox=box, confidence=conf, source=source)
        fr.original_polygon = pre.box_to_original(box)
        if source == "prior":
            fr.notes.append("located from the CTS-2010 layout prior only (no visual confirmation)")
        crop = box.pad(pad, pad).crop(pre.color)
        if name == "signature":
            fr.value = None
        elif name == "cheque_number":
            pass   # read from the MICR line below
        else:
            ink = handwriting_ink_ratio(crop)
            if name in ("payee", "date", "amount_numeric", "amount_words") and ink < 0.004:
                fr.notes.append("field appears empty")
                fr.text, fr.text_confidence, fr.ocr_engine = "", 0.0, None
            else:
                r = ocr.read(crop, name)
                fr.text, fr.text_confidence, fr.ocr_engine = r.text, r.confidence, r.engine
                fr.value = parse_value(name, r.text)
                if r.engine == "none":
                    fr.notes.append("no OCR engine available")
                elif r.confidence < MIN_TEXT_CONFIDENCE:
                    fr.notes.append(f"low OCR confidence ({r.confidence:.2f})")
                if fr.text and fr.value is None and name not in ("payee",):
                    fr.notes.append("text read but could not be parsed")
        results[name] = fr
    micr = results["micr"]
    chq = results["cheque_number"]
    if isinstance(micr.value, dict) and micr.value.get("cheque_number"):
        chq.text, chq.value = micr.value["cheque_number"], micr.value["cheque_number"]
        chq.text_confidence, chq.ocr_engine = micr.text_confidence, micr.ocr_engine
    else:
        r = ocr.read(chq.bbox.pad(pad, pad).crop(pre.color), "cheque_number")
        chq.text, chq.text_confidence, chq.ocr_engine = r.text, r.confidence, r.engine
        chq.value = parse_value("cheque_number", r.text)
    return results, structure
