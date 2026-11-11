"""OCR for cheque fields with pluggable engines.

  * crnn      - built-in CRNN-CTC recogniser (text_recognizer.py), field-constrained alphabets
  * tesseract - Tesseract via aiproject.setup_tesseract when the binary is installed
  * digit_cnn - the original MNIST CNN from aiproject.py, used for per-digit segmentation
                and confidence of handwritten figures (not for whole-field reading)
"""
import os
from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

import aiproject
from . import text_recognizer
from .config import PRINTED_FIELDS

ALLOWED = {
    "amount_numeric": set("0123456789,./-="),
    "date": set("0123456789/"),
    "cheque_number": set("0123456789 "),
    "account_number": set("0123456789"),
    "micr": set("0123456789 "),
    "ifsc": set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789: "),
    "payee": set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ &.,/-"),
    "amount_words": set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -"),
}
MULTILINE = {"amount_words", "payee"}
HANDWRITTEN = {"amount_numeric", "amount_words", "date", "payee"}


def clean_for_ocr(crop, margin=6):
    """Remove printed rules and box borders (inpainted), then crop tightly to the remaining ink.

    The recogniser was trained on line crops without box borders; a border next to the
    figures is otherwise read as an extra '1'.
    """
    from .signature_extractor import ink_strength
    s = ink_strength(crop)
    thr, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (s > max(thr, 30)).astype(np.uint8) * 255
    h, w = ink.shape
    rules = np.zeros_like(ink)
    for k in ((max(20, w // 5), 1), (1, max(12, int(h * 0.6)))):
        rules |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))
    cleaned = cv2.inpaint(crop, rules, 3, cv2.INPAINT_TELEA) if rules.any() else crop.copy()
    writing = ink & ~rules
    n, lab, stats, _ = cv2.connectedComponentsWithStats(writing, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 12]
    if not keep:
        return cleaned
    x1 = min(stats[i, 0] for i in keep)
    y1 = min(stats[i, 1] for i in keep)
    x2 = max(stats[i, 0] + stats[i, 2] for i in keep)
    y2 = max(stats[i, 1] + stats[i, 3] for i in keep)
    return cleaned[max(0, y1 - margin):min(h, y2 + margin), max(0, x1 - margin):min(w, x2 + margin)]


@dataclass
class OCRResult:
    text: str
    confidence: float          # 0-1
    engine: str
    alternatives: dict = dc_field(default_factory=dict)   # engine -> (text, confidence)


class OCREngine:
    def __init__(self, tesseract_cmd=None, use_crnn=True, use_tesseract=True, use_digit_cnn=True):
        self.has_tesseract = bool(use_tesseract and aiproject.setup_tesseract(tesseract_cmd))
        self.has_crnn = bool(use_crnn and text_recognizer.available())
        self.use_digit_cnn = use_digit_cnn and os.path.isfile(aiproject.DEFAULT_MODEL)
        self._digit_model = None

    @property
    def engines(self):
        return [n for n, ok in (("crnn", self.has_crnn), ("tesseract", self.has_tesseract),
                                ("digit_cnn", self.use_digit_cnn)) if ok]

    # -- engines ---------------------------------------------------------
    def _crnn(self, crop, field):
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lines = text_recognizer.split_lines(gray) if field in MULTILINE else [gray]
        results = text_recognizer.recognize_lines(lines, ALLOWED.get(field))
        text = " ".join(t for t, _ in results if t).strip()
        confs = [c for t, c in results if t]
        return text, float(np.mean(confs)) if confs else 0.0

    def _tesseract(self, crop, field):
        import pytesseract
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        psm = 6 if field in MULTILINE else 7
        allowed = "".join(sorted(ALLOWED.get(field, set()) - {" "}))
        config = f"--oem 3 --psm {psm}" + (f" -c tessedit_char_whitelist={allowed}" if allowed else "")
        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
        words = [(w.strip(), float(c)) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            return "", 0.0
        return " ".join(w for w, _ in words), float(np.mean([c for _, c in words])) / 100.0

    def read(self, crop, field):
        """Read one field crop. Returns OCRResult (empty text and confidence 0 if no engine can read it)."""
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 6:
            return OCRResult("", 0.0, "none")
        alts = {}
        crop = clean_for_ocr(crop) if field in HANDWRITTEN or field == "account_number" else crop
        if self.has_crnn:
            alts["crnn"] = self._crnn(crop, field)
        if self.has_tesseract:
            try:
                alts["tesseract"] = self._tesseract(crop, field)
            except Exception as exc:   # a broken Tesseract install should not stop the analysis
                alts["tesseract"] = ("", 0.0)
                alts["tesseract_error"] = (str(exc)[:120], 0.0)
        if not alts:
            return OCRResult("", 0.0, "none")
        # Handwriting: the CRNN (Tesseract is only a fallback when the CRNN read nothing).
        # Printed fields: Tesseract when it is confident, otherwise whichever engine is more confident.
        texts = {n: v for n, v in alts.items() if n in ("crnn", "tesseract") and v[0]}
        if not texts:
            return OCRResult("", 0.0, "none" if not alts else next(iter(alts)), alts)
        if field in HANDWRITTEN:
            name = "crnn" if "crnn" in texts else "tesseract"
        elif "tesseract" in texts and texts["tesseract"][1] >= 0.6:
            name = "tesseract"
        else:
            name = max(texts, key=lambda n: texts[n][1])
        return OCRResult(texts[name][0], texts[name][1], name, alts)

    # -- per-digit analysis with the existing MNIST CNN --------------------
    def digit_model(self):
        if self._digit_model is None and self.use_digit_cnn:
            try:
                self._digit_model = aiproject.load_model(aiproject.DEFAULT_MODEL)
            except Exception as exc:   # TensorFlow missing or incompatible
                print(f"digit CNN unavailable: {exc}")
                self.use_digit_cnn = False
        return self._digit_model

    def digit_analysis(self, binary):
        """Per-character boxes (connected components) with MNIST CNN digit and confidence.

        `binary` is an ink=255 crop. Returns [(box (x,y,w,h), digit, confidence, entropy)].
        """
        boxes = aiproject.find_digit_candidates(binary)
        model = self.digit_model()
        if not boxes or model is None:
            return [(b, None, None, None) for b in boxes]
        batch = np.stack([aiproject.to_mnist_format(binary, b) for b in boxes])[..., None]
        probs = model.predict(batch, verbose=0)
        ent = -(probs * np.log(probs + 1e-9)).sum(1) / np.log(10)
        return [(b, int(p.argmax()), float(p.max()), float(e)) for b, p, e in zip(boxes, probs, ent)]
