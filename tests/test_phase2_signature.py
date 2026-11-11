"""Phase 2: signature extraction, preprocessing, Siamese network, verification and metrics."""
import os

import numpy as np
import pytest
import torch

from cheque_forensics import signature_verifier as sv
from cheque_forensics.config import SIAMESE_WEIGHTS, SIGNATURE_SIZE
from cheque_forensics.nets import SiameseNetwork, contrastive_loss
from cheque_forensics.signature_extractor import extract_signature, preprocess_signature
from cheque_forensics.synth import signatures as sigs
from tests.conftest import needs
from training.train_siamese import eer, rates

cedar = pytest.mark.skipif(not sigs.cedar_available(), reason="CEDAR not downloaded")


def wide_signature():
    img = np.full((120, 400), 255, np.uint8)
    import cv2
    cv2.putText(img, "Ramesh K", (10, 80), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.0, 30, 3)
    return img


def test_preprocess_keeps_aspect_and_normalises():
    x = preprocess_signature(wide_signature())
    assert x.shape == SIGNATURE_SIZE and x.dtype == np.float32
    assert 0 <= x.min() and x.max() <= 1
    cols = np.nonzero(x.max(0) > 0.2)[0]
    rows = np.nonzero(x.max(1) > 0.2)[0]
    assert (cols[-1] - cols[0]) > 2.5 * (rows[-1] - rows[0])     # stays wide, not stretched to the box
    assert x[0].max() == 0 and x[-1].max() == 0                    # background suppressed


def test_preprocess_blank_is_empty():
    assert preprocess_signature(np.full((50, 100), 250, np.uint8)).max() == 0


def test_extract_signature_from_cheque(genuine_sample):
    zone = genuine_sample["fields"]["signature"]
    ex = extract_signature(zone.crop(genuine_sample["image"]))
    assert ex.present and ex.ink_pixels > 150
    assert ex.bbox.w > 30 and ex.image.dtype == np.uint8


def test_empty_zone_has_no_signature():
    paper = np.full((140, 400, 3), (230, 240, 245), np.uint8)
    paper += np.random.default_rng(0).integers(0, 4, paper.shape, dtype=np.uint8)
    assert not extract_signature(paper).present


def test_siamese_shares_weights():
    net = SiameseNetwork().eval()
    x = torch.rand(2, 1, *SIGNATURE_SIZE)
    ea, eb, d = net(x, x)
    assert torch.allclose(ea, eb) and torch.allclose(d, torch.zeros(2), atol=1e-4)   # pairwise_distance adds eps
    assert torch.allclose(ea.norm(dim=1), torch.ones(2), atol=1e-5)
    assert len({id(p) for p in net.parameters()}) == len(list(net.embedder.parameters()))


def test_contrastive_loss():
    d = torch.tensor([0.0, 1.5])
    assert contrastive_loss(d, torch.tensor([1.0, 0.0]), margin=1.0).item() == pytest.approx(0.0)
    assert contrastive_loss(torch.tensor([0.2]), torch.tensor([0.0])).item() > 0


def test_far_frr_eer_definitions():
    g, f = np.array([0.1, 0.2, 0.3]), np.array([0.7, 0.8, 0.9])
    assert rates(g, f, 0.5) == (0.0, 0.0)          # FAR, FRR
    assert eer(g, f)[0] == 0.0
    same = np.array([0.5, 0.5])
    assert eer(same, same)[0] == pytest.approx(0.5)


def test_three_way_decision(monkeypatch):
    class Fake:
        def __call__(self, imgs):
            return self.e
    cal = {"accept": 0.4, "reject": 0.8, "eer": 0.6, "aggregation": "mean_distance", "n_refs_calibrated": 4,
           "logistic": {"coef": 5.0, "intercept": -3.0}, "genuine_spread_p90": 9}
    monkeypatch.setattr(sv, "_load", lambda: (None, cal))
    refs = [np.zeros((10, 10), np.uint8)] * 4
    for dist, outcome in ((0.2, "likely_match"), (0.6, "inconclusive"), (1.0, "suspicious_mismatch")):
        base = np.zeros(128, np.float32)
        base[0] = 1
        q = np.zeros(128, np.float32)
        q[0], q[1] = np.cos(np.arcsin(dist / 2) * 2), np.sin(np.arcsin(dist / 2) * 2)
        monkeypatch.setattr(sv, "embed", lambda imgs, q=q, base=base: np.stack([q] + [base] * 4))
        r = sv.verify(np.zeros((10, 10), np.uint8), refs)
        assert r.outcome == outcome, (dist, r.distance)
        assert r.explanation and 0 <= r.forgery_probability <= 1


def test_no_references_is_unavailable():
    assert sv.verify(np.zeros((10, 10), np.uint8), []).outcome == "unavailable"


def test_registry_hashes_account_and_deletes(tmp_path):
    reg = sv.SignatureRegistry(root=str(tmp_path), salt="t")
    assert reg.enroll("123456789012", [wide_signature(), wide_signature()]) == 2
    assert len(reg.get("1234 5678 9012")) == 2
    assert not any("123456789012" in d for d in os.listdir(tmp_path))
    assert reg.delete("123456789012") and reg.get("123456789012") == []


@cedar
@needs(SIAMESE_WEIGHTS)
def test_trained_model_separates_test_writers():
    """Writers 46-55 were never seen in training or calibration."""
    gen_d, forg_d = [], []
    for w in sigs.WRITER_SPLITS["test"][:6]:
        refs = [sigs.load_cedar(w, i) for i in (1, 2, 3, 4)]
        for i in (10, 15, 20):
            gen_d.append(sv.verify(sigs.load_cedar(w, i), refs).distance)
            forg_d.append(sv.verify(sigs.load_cedar(w, i, forged=True), refs).distance)
    assert np.mean(forg_d) > np.mean(gen_d)


@cedar
@needs(SIAMESE_WEIGHTS)
def test_comparison_image(genuine_sample):
    refs = [sigs.load_cedar(50, i) for i in (1, 2, 3)]
    r = sv.verify(sigs.load_cedar(50, 9), refs)
    img = sv.comparison_image(sigs.load_cedar(50, 9), refs, r)
    assert img.ndim == 3 and img.shape[1] > img.shape[0]
