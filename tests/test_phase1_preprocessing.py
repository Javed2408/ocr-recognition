"""Phase 1: cheque preprocessing, boundary detection and field localisation."""
import cv2
import numpy as np
import pytest

from cheque_forensics import amount_parser as ap
from cheque_forensics.cheque_detector import detect_cheque, warp_to_canonical
from cheque_forensics.cheque_preprocessor import preprocess
from cheque_forensics.config import CHEQUE_SEG_WEIGHTS, FIELD_SEG_WEIGHTS, FIELDS
from cheque_forensics.field_extractor import layout_score, locate_fields
from tests.conftest import needs
from training.evaluate_fields import photo_of


def statuses(pre):
    return {c.name: c.status for c in pre.quality}


def test_perspective_photo_is_found_and_flattened(genuine_sample):
    photo, corners = photo_of(genuine_sample["image"], np.random.default_rng(7))
    det = detect_cheque(photo)
    assert det.method != "full_image"
    assert np.abs(det.quad - corners).max() < 0.02 * np.linalg.norm(corners[1] - corners[0])
    warped, H = warp_to_canonical(photo, det.quad)
    assert warped.shape[1] == 1200 and 1.9 < warped.shape[1] / warped.shape[0] < 2.4


def test_scan_without_background_uses_full_image(genuine_sample):
    det = detect_cheque(genuine_sample["image"])
    assert det.method == "full_image"


def test_rotated_scan_is_deskewed(genuine_sample):
    img = genuine_sample["image"]
    m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 4.0, 1.0)
    rotated = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    pre = preprocess(rotated, use_models=False)
    assert abs(abs(pre.rotation) - 4.0) < 1.0


@needs(FIELD_SEG_WEIGHTS)
def test_upside_down_cheque_is_flipped(genuine_sample):
    flipped = cv2.rotate(genuine_sample["image"], cv2.ROTATE_180)
    pre = preprocess(flipped, orientation_scorer=layout_score)
    assert pre.flipped


def test_pipeline_outputs(genuine_sample):
    pre = preprocess(genuine_sample["image"], use_models=False)
    h, w = pre.color.shape[:2]
    for img in (pre.gray, pre.denoised, pre.flattened, pre.enhanced, pre.binary):
        assert img.shape == (h, w) and img.dtype == np.uint8
    assert set(np.unique(pre.binary)) <= {0, 255}
    assert np.count_nonzero(pre.binary) < pre.binary.size * 0.3   # ink is the minority, shown white
    assert pre.normalized.dtype == np.float32 and 0 <= pre.normalized.min() and pre.normalized.max() <= 1
    # homography maps canonical corners back inside the uploaded image
    back = pre.to_original([[0, 0], [w - 1, h - 1]])
    assert np.all(back >= -2) and np.all(back <= np.array(genuine_sample["image"].shape[1::-1]) + 2)


@pytest.mark.parametrize("degrade, check", [
    (lambda im: cv2.GaussianBlur(im, (0, 0), 4), "Sharpness"),
    (lambda im: (im * 0.12).astype(np.uint8), "Brightness"),
])
def test_quality_problems_are_reported(genuine_sample, degrade, check):
    assert statuses(preprocess(degrade(genuine_sample["image"]), use_models=False))[check] == "WARN"


def test_low_resolution_fails_quality(genuine_sample):
    tiny = cv2.resize(genuine_sample["image"], (280, 130))
    assert statuses(preprocess(tiny, use_models=False))["Resolution"] == "FAIL"


def test_poor_lighting_is_flattened(genuine_sample):
    img = genuine_sample["image"].astype(np.float32)
    shade = np.linspace(0.35, 1.0, img.shape[1], dtype=np.float32)[None, :, None]
    pre = preprocess((img * shade).astype(np.uint8), use_models=False)
    left, right = pre.flattened[:, :200].mean(), pre.flattened[:, -200:].mean()
    assert abs(left - right) < 25    # shading largely removed


@needs(FIELD_SEG_WEIGHTS)
@pytest.mark.parametrize("variant", ["clean", "jpeg30", "blur"])
def test_fields_are_localised_with_boxes(genuine_sample, variant):
    img = genuine_sample["image"]
    if variant == "jpeg30":
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], cv2.IMREAD_COLOR)
    elif variant == "blur":
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
    pre = preprocess(img, orientation_scorer=layout_score)
    located, _ = locate_fields(pre)
    assert set(located) == set(FIELDS)
    for name in FIELDS:
        box, conf, source = located[name]
        assert box.w > 0 and box.h > 0 and 0 <= conf <= 1
        assert box.iou(genuine_sample["fields"][name]) >= 0.4, (name, source)


@needs(CHEQUE_SEG_WEIGHTS)
def test_boundary_segmenter_runs(genuine_sample):
    from cheque_forensics.cheque_detector import segment_cheque
    photo, _ = photo_of(genuine_sample["image"], np.random.default_rng(3))
    prob = segment_cheque(photo)
    assert prob.shape == photo.shape[:2] and 0.15 < (prob > 0.5).mean() < 0.85


@pytest.mark.parametrize("text, value", [
    ("₹ 1,25,000/-", 125000.0), ("Rs.5000.00", 5000.0), ("50,000=00", 50000.0), ("5000/-", 5000.0),
    ("12,5O,OOO/-", 1250000.0), ("75000", 75000.0),
])
def test_numeric_amounts(text, value):
    assert ap.parse_numeric_amount(text) == value


@pytest.mark.parametrize("text, value", [
    ("Rupees One Lakh Twenty Five Thousand Only", 125000), ("FIFTY THOUSAND ONLY", 50000),
    ("Two Crore Five Lakh", 20_500_000), ("Fourty Five Thousand", 45000), ("Ruppes Fiftyy thousnd only", 50000),
    ("Five Hundred and Paise Fifty", 500.5),
])
def test_amount_words(text, value):
    assert ap.parse_amount_words(text) == value


def test_number_to_words_round_trip():
    for n in (7, 99, 105, 1000, 12345, 100000, 125000, 9999999, 12000000):
        assert ap.parse_amount_words(ap.number_to_indian_words(n)) == n
    assert ap.format_indian_number(1250000) == "12,50,000"


def test_dates_and_codes():
    import datetime as dt
    assert ap.parse_cheque_date("12032026")[0] == dt.date(2026, 3, 12)
    assert ap.parse_cheque_date("31022026")[0] is None
    assert ap.date_validity(dt.date(2026, 1, 1), today=dt.date(2026, 11, 1))[0] == "stale"
    assert ap.date_validity(dt.date(2026, 12, 1), today=dt.date(2026, 11, 1))[0] == "post-dated"
    assert ap.parse_ifsc("IFSC: QABC0123456") == "QABC0123456"
    assert ap.parse_ifsc("QABCO123456") == "QABC0123456"
    assert ap.parse_micr("123456 400002005 654321 31")["cheque_number"] == "123456"
