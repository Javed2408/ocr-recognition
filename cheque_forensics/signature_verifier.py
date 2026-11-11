"""Siamese-CNN signature verification against registered genuine samples.

The decision is three-way, with thresholds calibrated on validation writers
(training/train_siamese.py, models/signature_calibration.json):
  * likely_match        - distance below the threshold where skilled-forgery FAR <= 5 %
  * suspicious_mismatch - distance above the threshold where genuine FRR <= 5 %
  * inconclusive        - in between, or when the input is unusable
A mismatch is a reason for manual verification, never proof of forgery:
people's signatures vary naturally, and the model was trained on CEDAR
(Western-script signatures), not on Indian account holders.
"""
import glob
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field as dc_field
from functools import lru_cache

import cv2
import numpy as np

from .config import REGISTRY_DIR, SIAMESE_WEIGHTS, SIGNATURE_CALIBRATION
from .signature_extractor import preprocess_signature

OUTCOME_TEXT = {
    "likely_match": "Signature is consistent with the registered samples.",
    "suspicious_mismatch": "Signature differs from the registered samples beyond calibrated natural variation; "
                           "manual verification required.",
    "inconclusive": "Similarity falls between the calibrated accept and reject thresholds; manual verification "
                    "recommended.",
    "unavailable": "Signature verification could not be performed.",
}


@dataclass
class SignatureResult:
    outcome: str                        # likely_match | suspicious_mismatch | inconclusive | unavailable
    similarity: float = None            # 0-1 display score (1 - distance / 2)
    distance: float = None              # aggregated score compared with the thresholds
    forgery_probability: float = None   # logistic calibration on validation writers
    thresholds: dict = None
    per_reference: list = dc_field(default_factory=list)
    reference_spread: float = None
    explanation: str = ""
    notes: list = dc_field(default_factory=list)

    def to_dict(self):
        r = lambda v: None if v is None else round(float(v), 4)
        return {"outcome": self.outcome, "similarity": r(self.similarity), "distance": r(self.distance),
                "forgery_probability": r(self.forgery_probability), "thresholds": self.thresholds,
                "per_reference_distance": [r(d) for d in self.per_reference], "reference_spread": r(self.reference_spread),
                "explanation": self.explanation, "notes": self.notes}


@lru_cache(maxsize=1)
def _load():
    from .device import load_checkpoint
    from .nets import SiameseNetwork
    model = SiameseNetwork()
    load_checkpoint(model, SIAMESE_WEIGHTS)
    with open(SIGNATURE_CALIBRATION, encoding="utf-8") as f:
        cal = json.load(f)
    return model, cal


def available():
    try:
        _load()
        return True
    except (FileNotFoundError, OSError):
        return False


def embed(images):
    import torch
    from .device import get_device
    model, _ = _load()
    batch = np.stack([preprocess_signature(im) for im in images])[:, None].astype(np.float32)
    with torch.no_grad():
        return model.embedder(torch.from_numpy(batch).to(get_device())).cpu().numpy()


def verify(query, references):
    """Compare a query signature (gray/BGR image) with registered genuine references."""
    if not references:
        return SignatureResult("unavailable", explanation="No registered reference signatures were provided.")
    if not available():
        return SignatureResult("unavailable", explanation="Siamese model weights or calibration file are missing "
                                                          "(run training/train_siamese.py).")
    _, cal = _load()
    emb = embed([query] + list(references))
    q, refs = emb[0], emb[1:]
    dists = np.linalg.norm(refs - q, axis=1)
    mean_d = float(dists.mean())
    spread = None
    if len(refs) >= 2:
        spread = float(np.mean([np.linalg.norm(refs[i] - refs[j]) for i in range(len(refs)) for j in range(i + 1, len(refs))]))
    score = mean_d
    if cal.get("aggregation") == "spread_normalised_distance":
        score = mean_d / max(spread, 1e-3) if spread else mean_d
    acc, rej = cal["accept"], cal["reject"]
    if score <= acc:
        outcome = "likely_match"
    elif score >= rej:
        outcome = "suspicious_mismatch"
    else:
        outcome = "inconclusive"
    lg = cal["logistic"]
    prob = float(1 / (1 + np.exp(-(lg["coef"] * score + lg["intercept"]))))
    notes = []
    if len(references) < cal.get("n_refs_calibrated", 4):
        notes.append(f"only {len(references)} reference(s); thresholds were calibrated with "
                     f"{cal.get('n_refs_calibrated', 4)}, so the decision is less reliable")
    if spread is not None and cal.get("aggregation") != "spread_normalised_distance" and spread > cal.get("genuine_spread_p90", 9):
        notes.append("registered samples differ a lot from each other; natural variation is high")
    notes.append("Siamese model trained on CEDAR (Western-script signatures); performance on Indian signers is unmeasured")
    expl = (f"{OUTCOME_TEXT[outcome]} Score {score:.3f} vs accept <= {acc:.3f} / reject >= {rej:.3f} "
            f"(similarity {1 - mean_d / 2:.2f}; mean distance to {len(references)} references).")
    return SignatureResult(outcome, 1 - mean_d / 2, score, prob, {"accept": acc, "reject": rej, "eer": cal["eer"],
                                                                  "aggregation": cal.get("aggregation")},
                           dists.tolist(), spread, expl, notes)


def comparison_image(query, references, result, height=110):
    """Side-by-side panel: registered references | cheque signature, with the outcome colour."""
    def tile(img, border):
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = (255 - preprocess_signature(g, (height - 10, int((height - 10) * 2))) * 255).astype(np.uint8)
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        return cv2.copyMakeBorder(x, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=border)
    colour = {"likely_match": (60, 160, 60), "suspicious_mismatch": (40, 40, 210),
              "inconclusive": (0, 165, 230)}.get(result.outcome, (128, 128, 128))
    refs = [tile(r, (180, 180, 180)) for r in references[:4]]
    q = tile(query, colour)
    gap = np.full((q.shape[0], 16, 3), 255, np.uint8)
    row = np.hstack(sum([[r, gap] for r in refs], []) + [np.full((q.shape[0], 30, 3), 255, np.uint8), q])
    header = np.full((26, row.shape[1], 3), 255, np.uint8)
    cv2.putText(header, "Registered reference signatures", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.putText(header, "Cheque signature", (row.shape[1] - q.shape[1], 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return np.vstack([header, row])


# ---------------------------------------------------------------------------
# Registry of genuine reference signatures (explicit enrolment, deletable)
# ---------------------------------------------------------------------------
class SignatureRegistry:
    """Reference signatures per account, stored under data/signature_registry/<sha256 prefix>.

    Account numbers are never written to disk in clear: the directory name is a
    salted hash. Enrolment is explicit and `delete` removes all samples.
    """

    def __init__(self, root=REGISTRY_DIR, salt=None):
        self.root = root
        self.salt = salt or os.environ.get("CHEQUE_REGISTRY_SALT", "cheque-forensics")

    def _key(self, account):
        digits = "".join(ch for ch in str(account) if ch.isalnum())
        return hashlib.sha256((self.salt + digits).encode()).hexdigest()[:24]

    def enroll(self, account, images):
        d = os.path.join(self.root, self._key(account))
        os.makedirs(d, exist_ok=True)
        start = len(glob.glob(os.path.join(d, "ref_*.png")))
        for i, img in enumerate(images):
            cv2.imwrite(os.path.join(d, f"ref_{start + i:03d}.png"), img)
        return start + len(images)

    def get(self, account):
        files = sorted(glob.glob(os.path.join(self.root, self._key(account), "ref_*.png")))
        return [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]

    def delete(self, account):
        d = os.path.join(self.root, self._key(account))
        if os.path.isdir(d):
            shutil.rmtree(d)
            return True
        return False
