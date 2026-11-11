"""End-to-end cheque analysis: preprocessing -> fields -> signature -> alterations -> forensics -> validation -> risk."""
import os
import time
from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

from . import __version__
from . import signature_verifier as sv
from . import tamper_models, text_recognizer
from .amount_tampering_detector import analyze_amount
from .cheque_preprocessor import preprocess
from .config import (CHEQUE_SEG_WEIGHTS, FIELD_SEG_WEIGHTS, MAX_IMAGE_PIXELS, RISK_CALIBRATION, SIAMESE_WEIGHTS,
                     TAMPER_CALIBRATION)
from .device import get_device
from .document_forensics import analyze_document
from .field_alteration_detector import analyze_field
from .field_extractor import extract_fields, layout_score, segmentation_available
from .ocr_engine import OCREngine
from .risk_assessor import assess
from .signature_extractor import extract_signature
from .suspicious_region_detector import detect_regions
from .validation_engine import validate


@dataclass
class AnalysisResult:
    pre: object
    fields: dict
    structure: object
    signature_extraction: object
    signature: object
    references: list
    analyses: dict
    forensics: object
    regions: list
    heatmap: np.ndarray
    checks: list
    risk: object
    timings: dict
    info: dict = dc_field(default_factory=dict)

    def to_dict(self):
        return {
            "risk": self.risk.to_dict(),
            "checks": [c.to_dict() for c in self.checks],
            "fields": {k: f.to_dict() for k, f in self.fields.items()},
            "signature": self.signature.to_dict() if self.signature else None,
            "signature_present": bool(self.signature_extraction and self.signature_extraction.present),
            "field_analyses": {k: a.to_dict() for k, a in self.analyses.items()},
            "forensics": self.forensics.to_dict() if self.forensics else None,
            "regions": [r.to_dict() for r in self.regions],
            "preprocessing": {"detection": self.pre.detection.to_dict(), "deskew_degrees": round(self.pre.rotation, 2),
                              "rotated_180": self.pre.flipped, "normalized_size": list(self.pre.color.shape[1::-1]),
                              "quality": [{"name": c.name, "status": c.status, "detail": c.detail} for c in self.pre.quality],
                              "metrics": self.pre.quality_metrics},
            "timings_seconds": {k: round(v, 3) for k, v in self.timings.items()},
            "info": self.info,
        }


def model_status():
    """Which trained components exist (shown in the UI and report)."""
    return {
        "cheque_boundary_unet": os.path.isfile(CHEQUE_SEG_WEIGHTS),
        "field_segmentation_unet": os.path.isfile(FIELD_SEG_WEIGHTS),
        "text_recognizer_crnn": text_recognizer.available(),
        "siamese_signature": os.path.isfile(SIAMESE_WEIGHTS),
        "tamper_models": tamper_models.available(),
        "tamper_calibration": os.path.isfile(TAMPER_CALIBRATION),
        "risk_calibration": os.path.isfile(RISK_CALIBRATION),
    }


def decode_image(data):
    """Bytes -> BGR image with size validation. Raises ValueError for unusable input."""
    if not data:
        raise ValueError("empty file")
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a readable image (supported: PNG, JPEG, TIFF, BMP, WebP)")
    if img.shape[0] * img.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(f"image too large ({img.shape[1]}x{img.shape[0]} pixels)")
    return img


class ChequeAnalyzer:
    def __init__(self, use_models=True, tesseract_cmd=None, use_digit_cnn=True, use_tesseract=True):
        self.use_models = use_models
        self.ocr = OCREngine(tesseract_cmd, use_tesseract=use_tesseract, use_digit_cnn=use_digit_cnn)

    def analyze(self, image, file_bytes=None, references=(), original=None, today=None):
        """Analyse one cheque.

        image:       BGR array (or bytes) of the questioned cheque
        file_bytes:  the uploaded file, for JPEG-specific forensics
        references:  registered genuine signature images of the account holder
        original:    optional trusted image of the same cheque before it left the drawer
        """
        if isinstance(image, (bytes, bytearray)):
            file_bytes = file_bytes or bytes(image)
            image = decode_image(bytes(image))
        timings = {}
        t = time.perf_counter()

        def lap(name):
            nonlocal t
            now = time.perf_counter()
            timings[name] = now - t
            t = now

        scorer = layout_score if self.use_models and segmentation_available() else None
        pre = preprocess(image, use_models=self.use_models, orientation_scorer=scorer)
        lap("preprocessing")
        fields, structure = extract_fields(pre, self.ocr, use_model=self.use_models)
        lap("field_detection_and_ocr")

        zone = fields["signature"].bbox.crop(pre.color)
        sig_ex = extract_signature(zone)
        signature = None
        refs = [r for r in references if r is not None]
        if refs:
            if sig_ex.present:
                signature = sv.verify(sig_ex.image, refs)
            else:
                signature = sv.SignatureResult("inconclusive", explanation="No signature ink was found in the "
                                               "signature zone, so it could not be compared.", notes=sig_ex.notes)
        else:
            signature = sv.SignatureResult("unavailable", explanation="No registered reference signatures were "
                                           "supplied for this account.")
        lap("signature_verification")

        analyses = analyze_amount(pre.color, fields, self.ocr, use_dl=self.use_models)
        lap("amount_alteration")
        for name in ("date", "payee"):
            if fields[name].bbox is not None:
                analyses[name] = analyze_field(pre.color, name, fields[name].bbox, use_dl=self.use_models)
        lap("date_payee_alteration")

        orig_canon = None
        if original is not None:
            if isinstance(original, (bytes, bytearray)):
                original = decode_image(bytes(original))
            orig_canon = cv2.resize(preprocess(original, use_models=self.use_models, orientation_scorer=scorer).color,
                                    (pre.color.shape[1], pre.color.shape[0]), interpolation=cv2.INTER_AREA)
        chars = [b for k in ("amount_numeric", "date") if k in analyses for b in analyses[k].char_boxes]
        forensics = analyze_document(pre.color, file_bytes, orig_canon, fields, structure, chars)
        lap("document_forensics")

        regions, heat = detect_regions(pre.color.shape, analyses, forensics)
        fmt = {"aspect": round(pre.color.shape[1] / pre.color.shape[0], 3),
               "layout_score": round(layout_score(pre.color), 3) if scorer else None}
        checks = validate(fields, analyses, signature, forensics, pre.quality, self.ocr.engines,
                          sig_ex.present, pre.detection, today, fmt)
        risk = assess(checks, analyses, signature, forensics)
        lap("validation_and_risk")
        timings["total"] = sum(timings.values())
        info = {"version": __version__, "device": str(get_device()), "ocr_engines": self.ocr.engines,
                "models": model_status(), "trusted_original_used": original is not None,
                "n_reference_signatures": len(refs), "input_size": list(image.shape[1::-1]),
                "file_format": "JPEG" if forensics.jpeg.get("is_jpeg") else ("unknown" if not file_bytes else "lossless/other")}
        return AnalysisResult(pre, fields, structure, sig_ex, signature, refs, analyses, forensics, regions, heat,
                              checks, risk, timings, info)
