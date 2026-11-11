"""Fuse field-level and document-level evidence into suspicious regions and one heatmap.

Inputs: FieldAnalysis results (classical + CNN/Grad-CAM + autoencoder + U-Net
per handwritten field), the document ForensicsResult (image-forensics maps,
registration difference, duplicated glyphs) and the signature result.
Output: a list of merged regions, each with the evidence sources and the
explanations that support it, and a heatmap for display.
"""
from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

from .schema import SEVERITY_ORDER, BBox

FIELD_NAMES = {"amount_numeric": "Amount (figures)", "amount_words": "Amount (words)", "payee": "Payee",
               "date": "Date", "signature": "Signature", "cheque_number": "Cheque no.", "micr": "MICR",
               "account_number": "Account no.", "ifsc": "IFSC"}


@dataclass
class SuspiciousRegion:
    bbox: BBox
    field: str
    severity: str
    score: float
    sources: list = dc_field(default_factory=list)
    explanations: list = dc_field(default_factory=list)

    def to_dict(self):
        return {"bbox": self.bbox.to_dict(), "field": self.field, "severity": self.severity,
                "score": round(float(self.score), 3), "sources": self.sources, "explanations": self.explanations}


def _merge(regions, iou=0.2):
    out = []
    for r in sorted(regions, key=lambda r: (-SEVERITY_ORDER[r.severity], -r.score)):
        for m in out:
            if r.bbox.iou(m.bbox) > iou or r.bbox.overlap_fraction(m.bbox) > 0.6 or m.bbox.overlap_fraction(r.bbox) > 0.6:
                m.bbox = BBox.from_xyxy(min(m.bbox.x, r.bbox.x), min(m.bbox.y, r.bbox.y),
                                        max(m.bbox.x2, r.bbox.x2), max(m.bbox.y2, r.bbox.y2))
                m.sources = sorted(set(m.sources) | set(r.sources))
                m.explanations += [e for e in r.explanations if e not in m.explanations]
                m.score = max(m.score, r.score)
                if SEVERITY_ORDER[r.severity] > SEVERITY_ORDER[m.severity]:
                    m.severity = r.severity
                break
        else:
            out.append(SuspiciousRegion(r.bbox, r.field, r.severity, r.score, list(r.sources), list(r.explanations)))
    return out


def detect_regions(shape, field_analyses, forensics=None, min_severity="medium"):
    """Returns (regions sorted by severity/score, heatmap float32 0-1 of `shape`)."""
    H, W = shape[:2]
    heat = np.zeros((H, W), np.float32)
    candidates = []
    for name, fa in field_analyses.items():
        if fa.heatmap is not None and fa.crop_box is not None:
            b = fa.crop_box.clip(W, H)
            hm = cv2.resize(fa.heatmap, (b.w, b.h)) if fa.heatmap.shape != (b.h, b.w) else fa.heatmap
            heat[b.y:b.y2, b.x:b.x2] = np.maximum(heat[b.y:b.y2, b.x:b.x2], hm)
        for ind in fa.indicators:
            if ind.bbox is not None and SEVERITY_ORDER[ind.severity] >= SEVERITY_ORDER[min_severity]:
                candidates.append(SuspiciousRegion(ind.bbox.clip(W, H), name, ind.severity, ind.score,
                                                   [ind.module + ":" + ind.name], [ind.explanation]))
    if forensics is not None:
        if forensics.fused is not None:
            heat = np.maximum(heat, cv2.resize(forensics.fused, (W, H)) * 0.8)
        for ind in forensics.indicators:
            if ind.bbox is not None and SEVERITY_ORDER[ind.severity] >= SEVERITY_ORDER[min_severity]:
                candidates.append(SuspiciousRegion(ind.bbox.clip(W, H), ind.field or "document", ind.severity,
                                                   ind.score, [ind.module + ":" + ind.name], [ind.explanation]))
    regions = _merge(candidates)
    regions.sort(key=lambda r: (-SEVERITY_ORDER[r.severity], -r.score))
    return regions, heat


DISPLAY_FLOOR = 0.35   # weaker evidence is not painted, so the overlay does not wash over the whole cheque


def overlay(color, heat=None, regions=(), fields=None, alpha=0.45):
    """Cheque image with heatmap, field boxes (thin grey) and suspicious regions (red/orange)."""
    out = color.copy()
    if heat is not None:
        heat = np.where(heat >= DISPLAY_FLOOR, heat, 0)
    if heat is not None and heat.max() > 0:
        cm = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        a = (np.clip(heat, 0, 1) * alpha)[..., None]
        out = (out * (1 - a) + cm * a).astype(np.uint8)
    for name, f in (fields or {}).items():
        if f.bbox is None:
            continue
        b = f.bbox
        cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), (140, 110, 60), 1)
        cv2.putText(out, FIELD_NAMES.get(name, name), (b.x + 2, b.y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 110, 60), 1)
    for i, r in enumerate(regions, start=1):
        col = (30, 30, 220) if r.severity == "high" else (0, 140, 255)
        b = r.bbox
        cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), col, 2)
        cv2.putText(out, str(i), (b.x2 + 2, max(b.y + 12, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return out


def heatmap_image(heat, base=None):
    heat = np.where(heat >= DISPLAY_FLOOR, heat, 0)
    cm = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if base is None:
        return cm
    return cv2.addWeighted(base, 0.45, cm, 0.55, 0)
