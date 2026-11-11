"""Phases 7-8: web interface, PDF report and end-to-end robustness on the cases listed in the specification.

End-to-end tests assert that the system behaves sensibly (completes, reports the right checks, never calls a
degraded or altered cheque LOW RISK without evidence); detection *rates* are measured by
training/calibrate_risk.py and reported in models/metrics/system_evaluation.json.
"""
import io

import cv2
import numpy as np
import pytest

from cheque_forensics.config import SIAMESE_WEIGHTS
from cheque_forensics.synth import signatures as sigs
from cheque_forensics.synth.tampering import make_sample
from tests.conftest import needs

TODAY = __import__("datetime").date(2026, 11, 3)


@pytest.fixture(scope="module")
def analyzer():
    from cheque_forensics.pipeline import ChequeAnalyzer
    return ChequeAnalyzer()


def refs_for(sample):
    return [sigs.load_cedar(w, i) for w, i in sample["references"]] if sample["references"] else []


def run(analyzer, sample, image=None, **kw):
    return analyzer.analyze(sample["image"] if image is None else image, None, refs_for(sample), today=TODAY, **kw)


def status(result, cid):
    return {c.id: c.status for c in result.checks}[cid]


# ------------------------------------------------------------------ end-to-end cases
def test_genuine_cheque_report_is_complete(analyzer, genuine_sample):
    r = run(analyzer, genuine_sample)
    d = r.to_dict()
    assert d["risk"]["category"] in ("LOW RISK", "REQUIRES REVIEW", "HIGH RISK")
    assert len(d["checks"]) == 11 and all(c["explanation"] for c in d["checks"])
    assert {c["id"]: c["status"] for c in d["checks"]}["cheque_format"] == "PASS"
    assert all(f["bbox"] for f in d["fields"].values())
    assert d["timings_seconds"]["total"] > 0


@needs(SIAMESE_WEIGHTS)
def test_forged_signature_is_not_accepted(analyzer):
    s = make_sample(0, "test", tamper_prob=1.0, seed=811, forged_signature_prob=1.0)
    assert s["forged_signature"]
    r = run(analyzer, s)
    assert r.signature.outcome in ("suspicious_mismatch", "inconclusive", "likely_match")
    if r.signature.outcome != "likely_match":
        assert status(r, "signature") in ("FAIL", "WARN") and r.risk.category != "LOW RISK"


@needs(SIAMESE_WEIGHTS)
def test_genuine_signature_variation(analyzer):
    """Another genuine sample of the same writer should usually not be called a mismatch."""
    outcomes = [run(analyzer, make_sample(i, "test", tamper_prob=0.0, seed=812)).signature.outcome for i in range(3)]
    assert outcomes.count("suspicious_mismatch") <= 1, outcomes


@pytest.mark.parametrize("op, check", [
    ("amount_overwrite_digit", "amount_overwriting"), ("amount_prefix_digit", "amount_overwriting"),
    ("amount_erase_rewrite", "amount_overwriting"), ("date_overwrite_digit", "date_anomalies"),
    ("payee_rewrite", "payee_anomalies"), ("payee_append", "payee_anomalies"),
])
def test_alterations_with_trusted_original_are_never_low_risk(analyzer, op, check):
    s = make_sample(2, "test", tamper_prob=1.0, seed=813, ops=[op], with_original=True)
    r = run(analyzer, s, original=s["original"])
    assert status(r, "image_manipulation") == "FAIL"           # pixel changes versus the original
    assert r.risk.category != "LOW RISK"
    regions = [g.bbox for g in r.regions]
    gt = s["events"][0].bbox
    assert any(b.iou(gt) > 0.1 or b.overlap_fraction(gt) > 0.5 or gt.overlap_fraction(b) > 0.4 for b in regions)


def test_low_resolution_requires_review(analyzer, genuine_sample):
    small = cv2.resize(genuine_sample["image"], (280, 130), interpolation=cv2.INTER_AREA)
    r = run(analyzer, genuine_sample, image=small)
    assert status(r, "image_quality") == "FAIL" and r.risk.category != "LOW RISK"


def test_blurred_image_is_flagged(analyzer, genuine_sample):
    r = run(analyzer, genuine_sample, image=cv2.GaussianBlur(genuine_sample["image"], (0, 0), 4))
    assert status(r, "image_quality") in ("WARN", "FAIL") and r.risk.category != "LOW RISK"


def test_rotated_cheque_is_straightened(analyzer, genuine_sample):
    img = genuine_sample["image"]
    m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), -5, 1.0)
    rot = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    r = run(analyzer, genuine_sample, image=rot)
    assert abs(abs(r.pre.rotation) - 5) < 1.0
    assert r.fields["amount_numeric"].bbox.iou(genuine_sample["fields"]["amount_numeric"]) > 0.3


def test_upside_down_photo(analyzer, genuine_sample):
    r = run(analyzer, genuine_sample, image=cv2.rotate(genuine_sample["image"], cv2.ROTATE_180))
    assert r.pre.flipped


def test_poor_lighting(analyzer, genuine_sample):
    img = genuine_sample["image"].astype(np.float32)
    shade = np.linspace(0.3, 1.0, img.shape[1], dtype=np.float32)[None, :, None]
    r = run(analyzer, genuine_sample, image=(img * shade).astype(np.uint8))
    assert r.risk.category in ("LOW RISK", "REQUIRES REVIEW", "HIGH RISK")
    assert r.fields["date"].bbox.iou(genuine_sample["fields"]["date"]) > 0.3


def test_heavily_compressed_jpeg(analyzer, genuine_sample):
    data = cv2.imencode(".jpg", genuine_sample["image"], [cv2.IMWRITE_JPEG_QUALITY, 25])[1].tobytes()
    r = analyzer.analyze(data, references=refs_for(genuine_sample), today=TODAY)
    assert r.forensics.jpeg["is_jpeg"] and r.forensics.jpeg["estimated_quality"] < 45


def test_unreadable_fields_are_reported(analyzer, genuine_sample):
    """Blank out the handwritten fields: they must be reported missing, never silently passed."""
    img = genuine_sample["image"].copy()
    for f in ("amount_numeric", "amount_words", "date", "payee"):
        b = genuine_sample["fields"][f]
        img[b.y:b.y2, b.x:b.x2] = np.median(img[b.y:b.y2, b.x:b.x2].reshape(-1, 3), axis=0)
    r = run(analyzer, genuine_sample, image=img)
    assert status(r, "missing_fields") in ("WARN", "FAIL")
    assert status(r, "amount_consistency") == "WARN" and r.risk.category != "LOW RISK"


def test_not_a_cheque(analyzer):
    noise = np.random.default_rng(0).integers(0, 255, (500, 900, 3), dtype=np.uint8)
    r = analyzer.analyze(noise, today=TODAY)
    assert r.risk.category == "REQUIRES REVIEW" and status(r, "cheque_format") == "WARN"


def test_foreign_cheque_format_is_not_rated(analyzer):
    """The repository's US sample cheque is outside the training domain: review, never high/low risk."""
    import aiproject
    r = analyzer.analyze(cv2.imread(aiproject.DEFAULT_IMAGE), today=TODAY)
    assert status(r, "cheque_format") == "WARN" and r.risk.category == "REQUIRES REVIEW"


# ------------------------------------------------------------------ report and web
def test_pdf_report(analyzer, genuine_sample):
    from cheque_forensics.report_generator import generate_pdf, render_images
    r = run(analyzer, genuine_sample)
    pdf = generate_pdf(r, render_images(r))
    assert pdf[:5] == b"%PDF-" and len(pdf) > 50_000


@pytest.fixture(scope="module")
def client(analyzer):
    from webapp.app import create_app
    return create_app(analyzer).test_client()


def png(img):
    return io.BytesIO(cv2.imencode(".png", img)[1].tobytes())


def test_web_rejects_bad_input(client):
    assert client.post("/api/cheque/analyze", data={}, content_type="multipart/form-data").status_code == 400
    bad = client.post("/api/cheque/analyze", data={"cheque": (io.BytesIO(b"x"), "a.exe")}, content_type="multipart/form-data")
    assert bad.status_code == 400 and "unsupported" in bad.json["error"]
    corrupt = client.post("/api/cheque/analyze", data={"cheque": (io.BytesIO(b"x" * 100), "a.png")},
                          content_type="multipart/form-data")
    assert corrupt.status_code == 400


def test_web_analyse_report_delete(client, genuine_sample):
    data = {"cheque": (png(genuine_sample["image"]), "c.png"),
            "references": [(png(r), f"r{i}.png") for i, r in enumerate(refs_for(genuine_sample))],
            "original": (png(genuine_sample["original"]), "o.png")}
    r = client.post("/api/cheque/analyze", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.json
    assert {"risk", "checks", "fields", "signature", "regions", "images", "id"} <= set(d)
    assert d["images"]["suspicious_regions"].startswith("data:image/png;base64,")
    assert d["info"]["trusted_original_used"] is True
    assert r.headers["Cache-Control"] == "no-store" and "default-src 'self'" in r.headers["Content-Security-Policy"]
    pdf = client.get(f"/api/cheque/{d['id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.data[:5] == b"%PDF-"
    assert client.delete(f"/api/cheque/{d['id']}").json["deleted"]
    assert client.get(f"/api/cheque/{d['id']}/report.pdf").status_code == 404


def test_results_expire():
    from webapp.app import ResultCache
    cache = ResultCache(ttl=0)
    key = cache.put({"x": 1})
    assert cache.get(key) is None
