"""Explainable fraud-risk assessment.

Score (0-100): 100 x P(cheque altered or signature forged | evidence) from a
logistic regression fitted on labelled synthetic validation cheques
(training/calibrate_risk.py). Weights are learned, not chosen by hand; each
factor's contribution is reported in log-odds relative to an average
validation cheque.

Categories (thresholds from the same validation data):
  LOW RISK         score < t_low (chosen so <= 5% of altered validation cheques fall below it),
                   no FAIL check and no signature mismatch
  HIGH RISK        score >= t_high (chosen for >= 90% precision on validation) AND at least two
                   significant indicators from different checks
  REQUIRES REVIEW  everything else, including any image-quality failure, unreadable critical
                   fields, an inconclusive signature or low OCR confidence
Without a calibration file no score is shown and only the rules apply.
The score reflects synthetic validation data; it is not a calibrated real-world fraud probability.
"""
import json
from dataclasses import dataclass, field as dc_field
from functools import lru_cache

import numpy as np

from .config import RISK_CALIBRATION
from .schema import SEVERITY_ORDER

FEATURES = {
    "amount_mismatch": "Amount in figures disagrees with amount in words",
    "signature_forgery_prob": "Signature dissimilarity to registered samples",
    "amount_figures_alteration": "Alteration evidence in the amount (figures)",
    "amount_words_alteration": "Alteration evidence in the amount (words)",
    "date_alteration": "Alteration evidence in the date",
    "payee_alteration": "Alteration evidence in the payee name",
    "date_invalid": "Date invalid, stale or post-dated",
    "image_forensics": "Corroborated image-forensic anomalies",
    "duplicated_glyph": "Pixel-identical (copied) characters",
    "original_difference": "Pixel changes versus the trusted original",
    "unreadable_fields": "Missing or unreadable critical fields",
    "low_ocr_confidence": "OCR uncertainty",
    "image_quality": "Image quality problems",
}
CAP = 3.0


def _fused(analyses, name):
    fa = analyses.get(name)
    if fa is None:
        return 0.0
    v = fa.scores.get("fused")
    if v is None:
        return 0.0
    thr = fa.thresholds.get("fused", 1.0) or 1.0
    return float(min(v / thr, CAP))


def risk_features(checks, analyses, signature, forensics):
    """Numeric evidence vector (same definition at calibration and inference)."""
    c = {k.id: k for k in checks}
    amount = c["amount_consistency"]
    f = {
        "amount_mismatch": 1.0 if amount.status == "FAIL" else (0.5 if amount.status == "WARN" and
                                                                amount.evidence.get("figures_value") is not None
                                                                and amount.evidence.get("words_value") is not None else 0.0),
        "signature_forgery_prob": float(signature.forgery_probability) if signature is not None and
        signature.forgery_probability is not None else 0.5,
        "amount_figures_alteration": _fused(analyses, "amount_numeric"),
        "amount_words_alteration": _fused(analyses, "amount_words"),
        "date_alteration": _fused(analyses, "date"),
        "payee_alteration": _fused(analyses, "payee"),
        "date_invalid": 1.0 if any(w in c["date_anomalies"].explanation for w in ("stale", "post-dated", "not a valid"))
        else 0.0,
        "image_forensics": 0.0, "duplicated_glyph": 0.0, "original_difference": 0.0,
        "unreadable_fields": len(c["missing_fields"].evidence.get("missing_or_unreadable", [])) / 5.0,
        "low_ocr_confidence": 0.0,
        "image_quality": {"PASS": 0.0, "WARN": 0.5, "FAIL": 1.0}.get(c["image_quality"].status, 0.0),
    }
    confs = c["ocr_confidence"].evidence.get("confidence", {})
    crit = [v for k, v in confs.items() if k in ("amount_numeric", "amount_words", "date", "payee")]
    f["low_ocr_confidence"] = float(1 - np.mean(crit)) if crit else 1.0
    if forensics is not None:
        f["image_forensics"] = float(min(forensics.stats.get("corroborated_regions", 0), CAP))
        f["duplicated_glyph"] = float(any(i.name == "duplicated_glyph" for i in forensics.indicators))
        f["original_difference"] = float(any(i.name == "difference_from_original" and i.severity == "high"
                                             for i in forensics.indicators))
    return f


@lru_cache(maxsize=1)
def load_calibration():
    try:
        with open(RISK_CALIBRATION, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, OSError):
        return None


@dataclass
class RiskAssessment:
    category: str                       # LOW RISK | REQUIRES REVIEW | HIGH RISK
    score: float = None                 # 0-100 or None when uncalibrated
    calibrated: bool = False
    factors: list = dc_field(default_factory=list)       # [{name, label, value, contribution}]
    reasons: list = dc_field(default_factory=list)       # why this category
    significant: list = dc_field(default_factory=list)   # significant indicators (check ids)
    thresholds: dict = dc_field(default_factory=dict)
    disclaimer: str = ("This is a risk indicator based on image analysis, not proof that the cheque is genuine "
                       "or forged. Decisions need verification of the physical cheque and the account holder.")

    def to_dict(self):
        return {"category": self.category, "score": None if self.score is None else round(self.score, 1),
                "calibrated": self.calibrated, "factors": self.factors, "reasons": self.reasons,
                "significant_indicators": self.significant, "thresholds": self.thresholds,
                "disclaimer": self.disclaimer}


def assess(checks, analyses, signature, forensics):
    feats = risk_features(checks, analyses, signature, forensics)
    cal = load_calibration()
    significant = [c.id for c in checks if c.status == "FAIL" and SEVERITY_ORDER[c.severity] >= SEVERITY_ORDER["medium"]]
    review_triggers = [c for c in checks if c.status in ("FAIL", "WARN") and c.id != "cross_field_ink"]
    res = RiskAssessment(category="REQUIRES REVIEW", significant=significant)
    if cal:
        names = cal["features"]
        x = np.array([feats[n] for n in names])
        z = (x - np.array(cal["mean"])) / np.array(cal["std"])
        coef = np.array(cal["coef"])
        logit = float(cal["intercept"] + coef @ z)
        res.score = float(100 / (1 + np.exp(-logit)))
        res.calibrated = True
        res.thresholds = {"low": cal["t_low"], "high": cal["t_high"]}
        contrib = coef * z
        res.factors = sorted([{"name": n, "label": FEATURES[n], "value": round(float(v), 3),
                               "weight": round(float(w), 3), "contribution": round(float(cb), 3)}
                              for n, v, w, cb in zip(names, x, coef, contrib)], key=lambda d: -abs(d["contribution"]))
    else:
        res.factors = [{"name": n, "label": FEATURES[n], "value": round(float(v), 3), "weight": None,
                        "contribution": None} for n, v in feats.items()]
        res.reasons.append("Risk model not calibrated (models/risk_calibration.json missing): category from rules only, "
                           "no numeric score.")
    sig_bad = signature is not None and signature.outcome == "suspicious_mismatch"
    out_of_domain = any(c.id == "cheque_format" and c.status == "WARN" for c in checks)
    if out_of_domain:
        res.category = "REQUIRES REVIEW"
        res.reasons.append("The image does not match the CTS-2010 cheque format the models were trained on, so the "
                           "automated indicators are unreliable and the cheque needs manual review.")
        res.reasons += [f"{c.name}: {c.status}." for c in review_triggers if c.id != "cheque_format"][:4]
        return res
    if res.calibrated and res.score >= cal["t_high"] and len(significant) >= 2:
        res.category = "HIGH RISK"
        res.reasons.append(f"Score {res.score:.0f} is at or above the high-risk threshold {cal['t_high']:.0f} and "
                           f"{len(significant)} independent checks failed: {', '.join(significant)}.")
    elif not res.calibrated and len(significant) >= 3:
        res.category = "HIGH RISK"
        res.reasons.append(f"{len(significant)} independent checks failed: {', '.join(significant)}.")
    elif res.calibrated and res.score < cal["t_low"] and not significant and not sig_bad and \
            not any(c.id in ("image_quality", "missing_fields") and c.status == "FAIL" for c in checks):
        res.category = "LOW RISK"
        res.reasons.append(f"Score {res.score:.0f} is below the low-risk threshold {cal['t_low']:.0f} and no check failed.")
        warn = [c.name for c in review_triggers]
        if warn:
            res.reasons.append("Minor warnings to note: " + ", ".join(warn) + ".")
    else:
        if res.calibrated and res.score >= cal["t_high"]:
            res.reasons.append(f"Score {res.score:.0f} is high, but only {len(significant)} significant indicator(s); "
                               "high risk requires at least two independent ones.")
        elif res.calibrated:
            res.reasons.append(f"Score {res.score:.0f} lies between the thresholds "
                               f"({cal['t_low']:.0f}-{cal['t_high']:.0f}) or other checks need attention.")
        for c in review_triggers[:5]:
            res.reasons.append(f"{c.name}: {c.status}.")
        if not review_triggers and not res.calibrated:
            res.reasons.append("Without a calibrated score, low risk cannot be confirmed automatically.")
    return res
