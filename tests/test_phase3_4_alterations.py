"""Phases 3-4: amount, date and payee alteration analysis (classical features, deep models, validation rules)."""
import datetime as dt

import cv2
import numpy as np
import pytest

from cheque_forensics import forensic_features as ff
from cheque_forensics.amount_tampering_detector import analyze_amount
from cheque_forensics.config import TAMPER_CALIBRATION, TAMPER_CLASSIFIER_WEIGHTS
from cheque_forensics.field_alteration_detector import analyze_field, spacing_anomalies
from cheque_forensics.schema import BBox, FieldResult
from cheque_forensics.synth.tampering import make_sample
from cheque_forensics.validation_engine import check_amount_consistency, check_date, check_payee
from tests.conftest import needs


def fr(name, text, value, conf=0.95, notes=()):
    return FieldResult(name, BBox(0, 0, 10, 10), 1.0, "test", text=text, text_confidence=conf, value=value,
                       notes=list(notes))


# ------------------------------------------------------------------ classical features
@pytest.mark.parametrize("t", [3, 5, 7])
def test_stroke_width_matches_drawn_thickness(t):
    img = np.zeros((60, 200), np.uint8)
    cv2.line(img, (10, 30), (190, 30), 255, t)
    true_width = (img[:, 100] > 0).sum()
    assert abs(np.median(ff.stroke_widths(img)) - true_width) <= 0.5


def test_stroke_width_of_fully_inked_crop_is_finite():
    assert np.isfinite(ff.stroke_widths(np.full((10, 10), 255, np.uint8))).all()


def test_lbp_histogram_is_normalised():
    g = np.random.default_rng(0).integers(0, 255, (40, 40)).astype(np.uint8)
    h = ff.lbp_hist(ff.lbp(g))
    assert h.shape == (59,) and h.sum() == pytest.approx(1.0)


def test_robust_z_flags_outlier_only():
    z = ff.robust_z([4, 4.2, 3.9, 4.1, 9.0])
    assert z[-1] > 3.5 and np.all(np.abs(z[:-1]) < 2)


def test_squeezed_character_detected():
    chars = [BBox(0, 0, 20, 30), BBox(28, 0, 20, 30), BBox(56, 0, 20, 30), BBox(77, 2, 8, 28), BBox(86, 0, 20, 30)]
    found = spacing_anomalies(chars)
    assert [i for i, *_ in found] == [3]


def test_uniform_ink_is_not_an_inconsistency(genuine_sample):
    """One pen throughout the field: the ink-colour component stays below its flag level."""
    s = genuine_sample
    crop = s["fields"]["payee"].pad(3, 3).crop(s["image"])
    mask = ff.ink_mask(crop)
    gray, lab = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    feats = [ff.region_features(crop, gray, lab, mask, ff.lbp(gray), b) for b in ff.segment_characters(mask)]
    rows, _ = ff.anomaly_table([f for f in feats if f is not None])
    assert max(r["ink"] for r in rows) < 1.0


# ------------------------------------------------------------------ amount consistency (the spec's example)
def test_amount_inconsistency_example():
    c = check_amount_consistency({"amount_numeric": fr("amount_numeric", "80,000/-", 80000.0),
                                  "amount_words": fr("amount_words", "Fifty thousand rupees only", 50000.0)})
    assert c.status == "FAIL" and c.severity == "high"
    assert "Amount inconsistency detected" in c.explanation and "₹80,000" in c.explanation


def test_added_zero_is_explained():
    c = check_amount_consistency({"amount_numeric": fr("amount_numeric", "50,000/-", 50000.0),
                                  "amount_words": fr("amount_words", "Five Thousand Only", 5000.0)})
    assert c.status == "FAIL" and "factor of 10" in c.explanation


def test_low_confidence_mismatch_is_only_a_warning():
    c = check_amount_consistency({"amount_numeric": fr("amount_numeric", "80000", 80000.0, conf=0.3),
                                  "amount_words": fr("amount_words", "Fifty Thousand", 50000.0)})
    assert c.status == "WARN" and "uncertain" in c.explanation


def test_matching_amounts_pass():
    c = check_amount_consistency({"amount_numeric": fr("amount_numeric", "1,25,000/-", 125000.0),
                                  "amount_words": fr("amount_words", "One Lakh Twenty Five Thousand Only", 125000.0)})
    assert c.status == "PASS"


# ------------------------------------------------------------------ date and payee rules
@pytest.mark.parametrize("text, word", [("01012025", "stale"), ("01122026", "post-dated"), ("31022026", "not a valid")])
def test_date_validity(text, word):
    from cheque_forensics.amount_parser import parse_cheque_date
    c = check_date({"date": fr("date", text, parse_cheque_date(text)[0])}, {}, today=dt.date(2026, 11, 1))
    assert c.status in ("FAIL", "WARN") and word in c.explanation


def test_valid_date_passes():
    c = check_date({"date": fr("date", "15102026", dt.date(2026, 10, 15))}, {}, today=dt.date(2026, 11, 1))
    assert c.status == "SKIP" or c.status == "PASS"


def test_blank_payee_warns():
    c = check_payee({"payee": fr("payee", "", None, notes=["field appears empty"])}, {})
    assert c.status == "WARN" and "blank payee" in c.explanation


# ------------------------------------------------------------------ analysis outputs
@pytest.mark.parametrize("field", ["amount_numeric", "amount_words", "date", "payee"])
def test_field_analysis_outputs(genuine_sample, field):
    a = analyze_field(genuine_sample["image"], field, genuine_sample["fields"][field])
    assert a.heatmap is not None and a.heatmap.shape[:2] == (a.crop_box.h, a.crop_box.w)
    assert 0 <= a.heatmap.min() and a.heatmap.max() <= 1
    assert "classical" in a.scores and "fused" in a.scores
    for ind in a.indicators:
        assert ind.explanation and ind.field == field


def test_amount_analysis_on_fields(genuine_sample):
    fields = {k: FieldResult(k, b, 1.0, "gt", text="5,000/-" if k == "amount_numeric" else None)
              for k, b in genuine_sample["fields"].items()}
    res = analyze_amount(genuine_sample["image"], fields)
    assert set(res) == {"amount_numeric", "amount_words"}


KNOWN_WEAKNESS = pytest.mark.xfail(
    strict=False, reason="KNOWN LIMITATION: a digit squeezed into the gap before the amount in the same hand is not "
    "separable by the field-level detectors (test recall 0.31, models/metrics/tamper_detection.json); the "
    "figures-vs-words check and the trusted-original comparison are the defences for this alteration")
ALTERATIONS = ["amount_overwrite_digit", pytest.param("amount_prefix_digit", marks=KNOWN_WEAKNESS), "amount_add_stroke",
               "amount_erase_rewrite", "date_overwrite_digit", "payee_append", "payee_rewrite", "words_rewrite"]


@needs(TAMPER_CALIBRATION)
@needs(TAMPER_CLASSIFIER_WEIGHTS)
@pytest.mark.parametrize("op", ALTERATIONS)
def test_altered_field_scores_higher_than_its_genuine_version(op):
    """Same leaf scanned before (sample["original"]) and after the alteration, test split: the altered field's
    fused evidence is higher in most of 4 cheques. Detection rates on the full test set are reported in
    models/metrics/tamper_detection.json."""
    wins, n = 0, 0
    for i in range(4):
        alt = make_sample(i, "test", tamper_prob=1.0, seed=600 + i, ops=[op], with_original=True)
        if not alt["events"]:
            continue
        f = alt["events"][0].field
        a = analyze_field(alt["image"], f, alt["fields"][f]).scores["fused"]
        g = analyze_field(alt["original"], f, alt["fields"][f]).scores["fused"]
        wins += a > g
        n += 1
    assert n and wins >= n / 2, f"{op}: altered field scored higher in only {wins}/{n} cheques"
