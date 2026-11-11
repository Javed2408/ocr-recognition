"""Amount alteration and overwriting detection (figures and words).

Builds on field_alteration_detector.analyze_field (classical per-character
forensics + CNN/Grad-CAM + autoencoder + U-Net) and adds amount-specific checks:
  * digit ambiguity from the original MNIST digit CNN (aiproject.py): an
    overwritten digit (1->7, 3->8, 0->6/9) often reads as two digits at once;
  * Indian digit grouping (1,25,000) broken by an inserted digit;
  * figures not closed with '/-' or '=00' and with room left to append digits
    (a process risk, reported as information);
  * amount in words that is not well formed (e.g. 'thousand lakh').
"""
import numpy as np

from . import amount_parser as ap
from . import forensic_features as ff
from .field_alteration_detector import analyze_field
from .schema import BBox, Indicator

AMBIGUITY_ENTROPY = 0.55     # normalised entropy of the MNIST CNN output (0 = certain, 1 = uniform)


def digit_ambiguity(crop, ocr, offset):
    """Indicators for digits whose shape the MNIST CNN finds ambiguous."""
    if ocr is None or not ocr.use_digit_cnn:
        return []
    mask = ff.ink_mask(crop)
    out = []
    for (x, y, w, h), digit, conf, ent in ocr.digit_analysis(mask):
        if ent is None or ent < AMBIGUITY_ENTROPY:
            continue
        out.append(Indicator(
            "amount_tampering", "ambiguous_digit", "low", float(ent),
            f"A character in the amount reads ambiguously to the digit CNN (best guess {digit} at {conf:.0%}, "
            f"entropy {ent:.2f}); overwritten digits often look like two digits at once, but untidy writing does too.",
            "amount_numeric", BBox(int(x), int(y), int(w), int(h)).shift(offset.x, offset.y),
            {"best_digit": digit, "confidence": conf, "entropy": ent}))
    return out[:3]


def analyze_amount(color, fields, ocr=None, use_dl=True):
    """Analyse the amount in figures and in words. Returns {field: FieldAnalysis}."""
    results = {}
    num = fields.get("amount_numeric")
    if num is not None and num.bbox is not None:
        res = analyze_field(color, "amount_numeric", num.bbox, use_dl=use_dl)
        if res.crop_box is not None:
            crop = res.crop_box.crop(color)
            res.indicators += digit_ambiguity(crop, ocr, res.crop_box)
            chars = sorted(res.char_boxes, key=lambda b: b.x)
            text = num.text or ""
            if chars and text and text[-1:].isdigit():
                room = num.bbox.x2 - chars[-1].x2
                if room > 1.5 * np.median([c.w for c in chars]):
                    res.indicators.append(Indicator(
                        "amount_tampering", "open_ended_amount", "info", 0.2,
                        "The figures are not closed with '/-' or '=00' and there is space after them, so digits "
                        "could be added later. This is a drafting risk, not evidence of alteration.",
                        "amount_numeric", num.bbox))
            if text and not ap.grouping_is_valid(text):
                res.indicators.append(Indicator(
                    "amount_tampering", "invalid_digit_grouping", "medium", 0.6,
                    f"The comma grouping in '{text}' fits neither Indian (12,34,567) nor Western (1,234,567) "
                    "style, which can happen when a digit is inserted (or when OCR misreads a character).",
                    "amount_numeric", num.bbox, {"ocr_text": text}))
        results["amount_numeric"] = res
    words = fields.get("amount_words")
    if words is not None and words.bbox is not None:
        res = analyze_field(color, "amount_words", words.bbox, use_dl=use_dl)
        if words.text and words.value is not None and not ap.amount_words_are_well_formed(words.text):
            res.indicators.append(Indicator(
                "amount_tampering", "malformed_amount_words", "medium", 0.5,
                f"The amount in words '{words.text}' has scale words out of order (e.g. 'thousand' before "
                "'lakh'), which can result from inserted words or from an OCR error.",
                "amount_words", words.bbox, {"ocr_text": words.text}))
        results["amount_words"] = res
    return results
