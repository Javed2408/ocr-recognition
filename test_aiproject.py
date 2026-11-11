"""Tests for the image checks and cheque validation. Run with: python -m pytest -q"""
import cv2
import numpy as np
import pytest

import aiproject as ap


def make_page(width=800, height=400, angle=0.0):
    """White page with a border, an underline and some text, optionally rotated."""
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (0, 0, 0), 3)
    cv2.line(img, (40, 300), (width - 40, 300), (0, 0, 0), 2)
    cv2.putText(img, "PAY 12345", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    if angle:
        m = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (width, height), borderValue=(255, 255, 255))
    return img


def run_checks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary, inverted = ap.binarize(gray)
    return {c.name: c.status for c in ap.validate_input_image(img, gray, binary, inverted)}


def test_good_page_passes():
    statuses = run_checks(make_page())
    assert set(statuses.values()) == {"PASS"}


def test_sample_cheque_passes():
    img = cv2.imread(ap.DEFAULT_IMAGE)
    assert img is not None
    assert "FAIL" not in run_checks(img).values()


def test_tiny_image_fails():
    assert run_checks(make_page(200, 100))["Resolution"] == "FAIL"


def test_blank_image_fails():
    blank = np.full((400, 800, 3), 255, np.uint8)
    assert run_checks(blank)["Content"] == "FAIL"


def test_blurry_image_warns():
    blurry = cv2.GaussianBlur(make_page(), (31, 31), 0)
    assert run_checks(blurry)["Sharpness"] == "WARN"


def test_dark_image_warns():
    dark = (make_page() * 0.1).astype(np.uint8)
    statuses = run_checks(dark)
    assert statuses["Brightness"] == "WARN"
    assert statuses["Contrast"] == "WARN"


def test_skewed_image_warns():
    assert run_checks(make_page(angle=8))["Skew"] == "WARN"


def test_inverted_image_is_flipped():
    inverted = cv2.bitwise_not(make_page())
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    binary, was_inverted = ap.binarize(gray)
    assert was_inverted
    assert np.count_nonzero(binary) < binary.size / 2  # text, not background, is white


@pytest.mark.parametrize("words, value", [
    ("four hundred ten", 410),
    ("twenty-five", 25),
    ("one thousand two hundred and thirty four", 1234),
    ("ninety nine", 99),
])
def test_written_amount(words, value):
    assert ap.parse_written_amount(f"Pay {words} dollars") == value


def test_numeric_amount():
    assert ap.parse_numeric_amount("PAY $1,250.75") == 1250.75
    assert ap.parse_numeric_amount("$ 410") == 410.0
    assert ap.parse_numeric_amount("no amount here") is None


def test_amount_match():
    text = "PAY Someone $410.00\nFour Hundred Ten DOLLARS"
    assert ap.check_amounts(text).status == "PASS"
    assert ap.check_amounts(text.replace("Ten", "Twenty")).status == "FAIL"
    assert ap.check_amounts("hello world").status == "SKIP"
    assert ap.check_amounts(None).status == "SKIP"


def test_save_and_verify(tmp_path):
    img = make_page()
    assert ap.save_and_verify(img, str(tmp_path / "out.png")).status == "PASS"
    assert ap.save_and_verify(img, str(tmp_path / "missing_dir" / "out.png")).status == "FAIL"
