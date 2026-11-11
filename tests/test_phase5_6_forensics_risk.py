"""Phases 5-6: document forensics, suspicious-region fusion, validation engine and risk assessment."""
import cv2
import numpy as np
import pytest

from cheque_forensics import risk_assessor as ra
from cheque_forensics.document_forensics import analyze_document, difference_map, duplicated_glyphs, jpeg_info
from cheque_forensics.schema import BBox, FieldResult, Indicator
from cheque_forensics.signature_verifier import SignatureResult
from cheque_forensics.suspicious_region_detector import detect_regions
from cheque_forensics.synth.tampering import make_sample
from cheque_forensics.validation_engine import ValidationCheck, check_signature


def fields_of(sample):
    return {k: FieldResult(k, b, 1.0, "gt") for k, b in sample["fields"].items()}


# ------------------------------------------------------------------ registration with a trusted original
@pytest.mark.parametrize("i", [1, 2])
def test_original_comparison_localises_alteration(i):
    s = make_sample(i, "test", tamper_prob=1.0, seed=700, ops=["amount_erase_rewrite"], with_original=True)
    r = analyze_document(s["image"], None, s["original"], fields_of(s))
    hits = [x for x in r.indicators if x.name == "difference_from_original"]
    assert hits, "no difference found"
    gt = s["events"][0].bbox
    # a detection covering the changed strokes lies inside the altered character's box
    assert any(h.bbox.iou(gt) > 0.2 or h.bbox.overlap_fraction(gt) > 0.5 for h in hits)
    assert hits[0].field == "amount_numeric" and hits[0].severity == "high"


def test_original_comparison_of_unaltered_cheque_is_clean(genuine_sample):
    _, change, info = difference_map(genuine_sample["image"], genuine_sample["original"])
    assert info["inliers"] > 50 and change.mean() < 0.001


# ------------------------------------------------------------------ digital edits
def test_pasted_copy_of_a_glyph_is_detected():
    s = make_sample(5, "test", tamper_prob=1.0, seed=701, ops=["digital_glyph_copy"])
    boxes = [b for f in ("amount_numeric", "date") for _, b in s["chars"][f]]
    gray = cv2.cvtColor(s["image"], cv2.COLOR_BGR2GRAY)
    pairs = duplicated_glyphs(gray, boxes)
    assert any(p[0].iou(s["events"][0].bbox) > 0.3 or p[1].iou(s["events"][0].bbox) > 0.3 for p in pairs)


def test_handwritten_repeats_are_not_called_copies(genuine_sample):
    boxes = [b for f in ("amount_numeric", "date") for _, b in genuine_sample["chars"][f]]
    assert duplicated_glyphs(cv2.cvtColor(genuine_sample["image"], cv2.COLOR_BGR2GRAY), boxes) == []


def test_jpeg_quality_estimate(genuine_sample):
    data = cv2.imencode(".jpg", genuine_sample["image"], [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    info = jpeg_info(data)
    assert info["is_jpeg"] and abs(info["estimated_quality"] - 80) <= 8
    assert not jpeg_info(cv2.imencode(".png", genuine_sample["image"])[1].tobytes())["is_jpeg"]


def test_image_level_cues_need_corroboration(genuine_sample):
    r = analyze_document(genuine_sample["image"], None, None, fields_of(genuine_sample))
    for ind in r.indicators:
        if ind.name in ("ela_anomaly", "jpeg_ghost_anomaly", "noise_anomaly", "texture_anomaly") and ind.severity == "medium":
            assert ind.evidence["corroborating_maps"], ind.explanation
        assert "supporting evidence" in ind.explanation or ind.name in ("duplicated_glyph", "difference_from_original")
    assert r.fused.shape == genuine_sample["image"].shape[:2]


# ------------------------------------------------------------------ region fusion
def test_overlapping_evidence_is_merged():
    class FA:
        heatmap, crop_box = None, None
        indicators = [Indicator("m", "a", "medium", 0.5, "first", "amount_numeric", BBox(10, 10, 20, 20)),
                      Indicator("m", "b", "high", 0.9, "second", "amount_numeric", BBox(15, 12, 20, 20)),
                      Indicator("m", "c", "medium", 0.4, "far away", "payee", BBox(200, 200, 10, 10))]
    regions, heat = detect_regions((300, 400, 3), {"amount_numeric": FA()})
    assert len(regions) == 2 and regions[0].severity == "high" and len(regions[0].explanations) == 2
    assert heat.shape == (300, 400)


# ------------------------------------------------------------------ validation + risk rules
def test_signature_below_threshold_requires_manual_verification():
    c = check_signature(SignatureResult("suspicious_mismatch", 0.3, 1.2, 0.9, explanation="x"))
    assert c.status == "FAIL" and "Signature requires manual verification" in c.explanation
    assert "Natural variation" in c.explanation
    assert check_signature(SignatureResult("inconclusive", explanation="y")).status == "WARN"


def checks(**statuses):
    ids = ["amount_consistency", "signature", "amount_overwriting", "date_anomalies", "payee_anomalies",
           "image_manipulation", "missing_fields", "ocr_confidence", "image_quality", "cheque_format", "cross_field_ink"]
    out = []
    for cid in ids:
        st = statuses.get(cid, "PASS")
        out.append(ValidationCheck(cid, cid, st, "high" if st == "FAIL" else "info", f"{cid} {st}",
                                   {"missing_or_unreadable": [], "confidence": {"amount_numeric": 0.95}}))
    return out


CAL = {"features": list(ra.FEATURES), "mean": [0.0] * len(ra.FEATURES), "std": [1.0] * len(ra.FEATURES),
       "coef": [1.0] * len(ra.FEATURES), "intercept": -1.0, "t_low": 30.0, "t_high": 60.0}


@pytest.fixture
def calibrated(monkeypatch):
    monkeypatch.setattr(ra, "load_calibration", lambda: CAL)


def test_high_risk_needs_two_significant_indicators(calibrated):
    sig = SignatureResult("suspicious_mismatch", forgery_probability=0.99)
    r = ra.assess(checks(amount_consistency="FAIL", signature="FAIL"), {}, sig, None)
    assert r.category == "HIGH RISK" and len(r.significant) == 2
    one = ra.assess(checks(amount_consistency="FAIL"), {}, sig, None)
    assert one.category == "REQUIRES REVIEW"


def test_low_risk_only_without_failures(calibrated):
    ok = SignatureResult("likely_match", forgery_probability=0.0)
    r = ra.assess(checks(), {}, ok, None)
    assert r.category == "LOW RISK" and r.score < CAL["t_low"]
    assert ra.assess(checks(image_quality="FAIL"), {}, ok, None).category == "REQUIRES REVIEW"


def test_factors_explain_score(calibrated):
    r = ra.assess(checks(amount_consistency="FAIL"), {}, SignatureResult("likely_match", forgery_probability=0.1), None)
    top = r.factors[0]
    assert top["name"] == "amount_mismatch" and top["contribution"] > 0
    assert r.disclaimer and "not proof" in r.disclaimer


def test_out_of_format_image_is_never_high_or_low(calibrated):
    sig = SignatureResult("suspicious_mismatch", forgery_probability=0.99)
    r = ra.assess(checks(amount_consistency="FAIL", signature="FAIL", cheque_format="WARN"), {}, sig, None)
    assert r.category == "REQUIRES REVIEW" and "CTS-2010" in r.reasons[0]


def test_uncalibrated_risk_has_no_score(monkeypatch):
    monkeypatch.setattr(ra, "load_calibration", lambda: None)
    r = ra.assess(checks(), {}, SignatureResult("likely_match", forgery_probability=0.1), None)
    assert r.score is None and r.category == "REQUIRES REVIEW" and not r.calibrated
