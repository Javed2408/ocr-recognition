"""Alteration analysis of handwritten fields (date, payee, amount in words; shared by the amount module).

For each field it combines
  * classical evidence per character and per word (stroke width, ink colour
    and darkness, double inking, doubled edges, paper texture around the
    writing, character spacing), always relative to the rest of the field;
  * deep evidence: tampering-classifier probability with Grad-CAM, autoencoder
    reconstruction error and U-Net tampering mask (tamper_models.py).
Flagging thresholds are calibrated on synthetic validation cheques
(training/train_tamper_models.py). Different handwriting or ink between
fields is reported as information only; within one field it is an indicator,
never proof.
"""
from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

from . import forensic_features as ff
from . import tamper_models
from .schema import BBox, Indicator

FIELD_LABEL = {"amount_numeric": "amount in figures", "amount_words": "amount in words", "date": "date",
               "payee": "payee name"}
DETECTORS = ("classical", "spacing", "cnn", "ae", "unet")
DEFAULT_THRESHOLDS = {"classical": 1.0, "spacing": 1.0, "cnn": 0.5, "ae": None, "unet": 0.5, "fused": 1.0}


@dataclass
class FieldAnalysis:
    field: str
    bbox: BBox                                   # field box in canonical coordinates
    scores: dict = dc_field(default_factory=dict)
    thresholds: dict = dc_field(default_factory=dict)
    indicators: list = dc_field(default_factory=list)
    heatmap: np.ndarray = None                   # float32 0-1 over `crop_box`
    crop_box: BBox = None                        # padded field box the heatmap covers
    char_boxes: list = dc_field(default_factory=list)   # canonical BBoxes
    calibrated: bool = False

    @property
    def suspicious(self):
        """Fused evidence above its calibrated threshold (5% false alarms on validation fields)."""
        fused = self.scores.get("fused")
        if fused is not None and self.calibrated:
            return fused >= self.thresholds.get("fused", 1.0)
        return any(i.severity in ("medium", "high") for i in self.indicators)

    def to_dict(self):
        return {"field": self.field, "bbox": self.bbox.to_dict(), "suspicious": self.suspicious,
                "scores": {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in self.scores.items()},
                "thresholds": self.thresholds, "calibrated": self.calibrated,
                "indicators": [i.to_dict() for i in self.indicators],
                "n_characters": len(self.char_boxes)}


def thresholds_for(field):
    cal = tamper_models.calibration()
    if not cal:
        return dict(DEFAULT_THRESHOLDS), False
    t = {k: (cal.get(k, {}).get(field, cal.get(k, {}).get("all")) if isinstance(cal.get(k), dict) else cal.get(k))
         for k in DEFAULT_THRESHOLDS}
    return {k: (v if v is not None else DEFAULT_THRESHOLDS[k]) for k, v in t.items()}, True


def fused_score(scores, thresholds):
    """Strongest detector relative to its own calibrated threshold (1.0 = at threshold)."""
    vals = [scores[d] / thresholds[d] for d in DETECTORS if d in scores and thresholds.get(d)]
    return float(max(vals)) if vals else 0.0


def spacing_score(anomalies):
    """0 when no squeezed character, larger the narrower and tighter the worst one is."""
    return max(((1 - wr) + (1 - min(gr, 1.0)) for _, _, wr, gr in anomalies), default=0.0)


def severity(score, threshold):
    if threshold is None or score is None:
        return None
    if score >= 2 * threshold:
        return "high"
    if score >= threshold:
        return "medium"
    if score >= 0.8 * threshold:
        return "low"
    return None


def _classical(crop, field):
    """Per-character and per-word anomalies. Returns (indicators, best score, char boxes, char heat)."""
    mask = ff.ink_mask(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    codes = ff.lbp(gray)
    regions, chars_all = [], []
    for y1, y2 in ff.split_line_boxes(mask):
        line = np.zeros_like(mask)
        line[y1:y2] = mask[y1:y2]
        chars = ff.segment_characters(line)
        chars_all += chars
        regions += [("character", c) for c in chars]
        if field in ("payee", "amount_words"):
            regions += [("word", BBox.from_xyxy(min(c.x for c in w), min(c.y for c in w), max(c.x2 for c in w),
                                                max(c.y2 for c in w))) for w in ff.group_words(chars) if len(w) >= 2]
    feats = [(kind, ff.region_features(crop, gray, lab, mask, codes, b)) for kind, b in regions]
    feats = [(k, f) for k, f in feats if f is not None]
    heat = np.zeros(crop.shape[:2], np.float32)
    best, best_info = 0.0, None
    found = []
    for kind in ("character", "word"):
        group = [f for k, f in feats if k == kind]
        if len(group) < 3:
            continue
        rows, combined = ff.anomaly_table(group)
        for f, comp, score in zip(group, rows, combined):
            if score > best:
                best, best_info = float(score), (kind, f, comp)
            if score >= 0.8:
                found.append((float(score), kind, f, comp))
            b = f.box
            heat[b.y:b.y2, b.x:b.x2] = np.maximum(heat[b.y:b.y2, b.x:b.x2], min(score / 2, 1.0))
    return found, best, chars_all, heat, mask


def spacing_anomalies(chars):
    """Squeezed characters: much narrower and closer to neighbours than the rest (inserted later)."""
    if len(chars) < 4:
        return []
    widths = np.array([c.w for c in chars], np.float32)
    gaps = np.array([b.x - a.x2 for a, b in zip(chars, chars[1:])], np.float32)
    med_w, med_gap = np.median(widths), max(np.median(gaps), 1.0)
    out = []
    for i, c in enumerate(chars):
        left = gaps[i - 1] if i > 0 else None
        right = gaps[i] if i < len(gaps) else None
        tight = [g for g in (left, right) if g is not None]
        if c.w < 0.6 * med_w and c.h > 0.6 * np.median([x.h for x in chars]) and tight and \
                min(tight) < 0.3 * med_gap and med_gap >= 3:
            out.append((i, c, float(c.w / med_w), float(min(tight) / med_gap)))
    return out


def analyze_field(color, field, bbox, use_dl=True, pad=3):
    """Analyse one handwritten field of the normalised cheque image."""
    label = FIELD_LABEL.get(field, field)
    box = bbox.pad(pad, pad).clip(color.shape[1], color.shape[0])
    crop = box.crop(color)
    th, calibrated = thresholds_for(field)
    result = FieldAnalysis(field=field, bbox=bbox, thresholds=th, calibrated=calibrated)
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        result.indicators.append(Indicator("field_alteration", "field_unreadable", "info", 0.0,
                                           f"The {label} region is too small to analyse.", field, bbox))
        return result
    found, best, chars, heat, mask = _classical(crop, field)
    result.char_boxes = [c.shift(box.x, box.y) for c in chars]
    result.scores["classical"] = best
    for score, kind, f, comp in sorted(found, key=lambda x: -x[0])[:4]:
        sev = severity(score, th["classical"])
        if sev is None:
            continue
        reasons = [ff.COMPONENT_TEXT[k] for k, v in sorted(comp.items(), key=lambda kv: -kv[1]) if v >= 0.8][:3]
        pos = sorted(chars, key=lambda c: c.x).index(f.box) + 1 if f.box in chars else None
        where = f"character {pos}" if kind == "character" and pos else kind
        result.indicators.append(Indicator(
            "field_alteration", f"{field}_{kind}_inconsistency", sev, min(score / (2 * th['classical']), 1.0),
            f"In the {label}, {where}: " + "; ".join(reasons) +
            f" (anomaly {score:.2f} vs calibrated threshold {th['classical']:.2f}). Natural handwriting varies, "
            "so this is a reason to inspect the original, not proof of alteration.",
            field, f.box.shift(box.x, box.y), {k: round(v, 2) for k, v in comp.items()}))
    squeezed = spacing_anomalies(sorted(chars, key=lambda c: c.x))
    result.scores["spacing"] = spacing_score(squeezed)
    for i, c, wr, gr in squeezed:
        sev = severity((1 - wr) + (1 - min(gr, 1.0)), th["spacing"])
        if sev is None:
            continue
        result.indicators.append(Indicator(
            "field_alteration", f"{field}_squeezed_character", sev, 0.6,
            f"In the {label}, character {i + 1} is narrower ({wr:.0%} of typical width) and closer to its "
            f"neighbour ({gr:.0%} of the typical gap) than the rest, as when a character is squeezed in later.",
            field, c.shift(box.x, box.y), {"width_ratio": wr, "gap_ratio": gr}))
    dl = tamper_models.analyze_crop(crop) if use_dl else {}
    maps = [heat * 0.7]
    if "classifier_prob" in dl:
        p = dl["classifier_prob"]
        result.scores["cnn"] = p
        sev = severity(p, th["cnn"])
        if sev:
            cam = dl["cam"]
            ys, xs = np.nonzero(cam > 0.5 * cam.max()) if cam.max() > 0 else ([], [])
            cam_box = BBox.from_xyxy(min(xs), min(ys), max(xs) + 1, max(ys) + 1).shift(box.x, box.y) if len(xs) else bbox
            result.indicators.append(Indicator(
                "field_alteration", f"{field}_cnn_tampering", sev, p,
                f"The tampering CNN gives the {label} a {p:.0%} alteration probability (flagging threshold "
                f"{th['cnn']:.2f}, set for a 5% false-alarm rate on validation cheques); Grad-CAM points to the "
                "highlighted area.", field, cam_box, {"window_probs": [round(w, 3) for w in dl["window_probs"]]}))
        maps.append(dl["cam"] * 0.8)
    if "ae_score" in dl:
        result.scores["ae"] = dl["ae_score"]
        if th.get("ae"):
            sev = severity(dl["ae_score"], th["ae"])
            ae_sev = None
            if sev:
                # reconstruction error alone is weak evidence: one severity step lower than the detector level
                ae_sev = {"high": "medium", "medium": "low"}.get(sev)
            if th.get("ae") and ae_sev:
                result.indicators.append(Indicator(
                    "field_alteration", f"{field}_autoencoder_anomaly", ae_sev,
                    min(dl["ae_score"] / (2 * th["ae"]), 1.0),
                    f"The {label} is reconstructed poorly by an autoencoder trained only on unaltered fields "
                    f"(error {dl['ae_score']:.3f} vs {th['ae']:.3f} at the 95th percentile of genuine fields).",
                    field, bbox))
            maps.append(np.clip(dl["ae_map"] / (2 * th["ae"]), 0, 1) * 0.6)
    if "unet_max" in dl:
        result.scores["unet"] = dl["unet_max"]
        um = dl["unet_map"]
        maps.append(um)
        if um.max() >= th["unet"]:
            n, lab_, stats, _ = cv2.connectedComponentsWithStats((um >= th["unet"]).astype(np.uint8))
            for k in range(1, n):
                x, y, w, h, area = stats[k]
                if area < 30:
                    continue
                peak = float(um[lab_ == k].max())
                result.indicators.append(Indicator(
                    "field_alteration", f"{field}_unet_region", severity(peak, th["unet"]) or "low", peak,
                    f"The U-Net localiser marks this part of the {label} as altered (peak {peak:.2f}, "
                    f"threshold {th['unet']:.2f}).", field, BBox(int(x), int(y), int(w), int(h)).shift(box.x, box.y)))
    result.scores["fused"] = fused_score(result.scores, th)
    result.heatmap = np.clip(np.max(np.stack(maps), axis=0), 0, 1).astype(np.float32)
    result.crop_box = box
    return result
