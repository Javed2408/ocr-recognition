"""Calibrate document-forensics thresholds and the risk model on VALIDATION cheques, then evaluate the
whole system on TEST cheques (Phase 6 + Phase 8).

1. Per-map thresholds for document forensics: 95th percentile of each map's strongest local region on
   unaltered validation cheques (5% false alarms per map).
2. Run the full pipeline on labelled validation cheques (half altered or with forged signatures; a third
   with a trusted original) and fit a logistic regression on the risk features -> models/risk_calibration.json.
   t_low: <= 5% of altered validation cheques score below it; t_high: >= 90% precision above it.
3. Run the full pipeline on test cheques and report risk, category, per-check, OCR, signature and timing metrics.

    python -m training.calibrate_risk --n-val 240 --n-test 240
"""
import argparse
import datetime as dt
import json
import os
import time

import cv2
import numpy as np

from training.common import Timer, parallel_map, save_metrics
from cheque_forensics.config import RISK_CALIBRATION, TAMPER_CALIBRATION

SEED = 31
TODAY = dt.date(2026, 11, 3)     # synthetic cheques are dated within 80 days before 2026-11-01
_ANALYZER = None


def _file_bytes(sample):
    q = sample["scan"].get("second_jpeg_quality") or sample["scan"].get("jpeg_quality")
    if q:
        return cv2.imencode(".jpg", sample["image"], [cv2.IMWRITE_JPEG_QUALITY, int(q)])[1].tobytes()
    return cv2.imencode(".png", sample["image"])[1].tobytes()


def _doc_stats(index):
    import torch
    torch.set_num_threads(1)
    from cheque_forensics.document_forensics import analyze_document
    from cheque_forensics.synth.tampering import make_sample
    s = make_sample(index, "val", tamper_prob=0.0, seed=SEED + 7)
    r = analyze_document(s["image"], _file_bytes(s))
    return r.stats["map_max"]


def _run(args):
    index, split = args
    global _ANALYZER
    import torch
    torch.set_num_threads(2)
    from cheque_forensics.pipeline import ChequeAnalyzer
    from cheque_forensics.risk_assessor import risk_features
    from cheque_forensics.synth import signatures as sigs
    from cheque_forensics.synth.tampering import amount_value, make_sample
    if _ANALYZER is None:
        _ANALYZER = ChequeAnalyzer()
    s = make_sample(index, split, tamper_prob=0.5, seed=SEED, with_original=(index % 3 == 0))
    refs = [sigs.load_cedar(w, i) for w, i in s["references"]] if s["references"] else []
    t = time.perf_counter()
    r = _ANALYZER.analyze(s["image"], _file_bytes(s), refs, s.get("original"), today=TODAY)
    elapsed = time.perf_counter() - t
    gt = s["content"]
    f = r.fields
    words_gt = None
    from cheque_forensics.amount_parser import parse_amount_words, parse_cheque_date
    words_gt = parse_amount_words(gt.words_text)
    events = {e.field for e in s["events"]}
    return {
        "label": s["label"], "kinds": [e.kind for e in s["events"]] + (["forged_signature"] if s["forged_signature"] else []),
        "altered_fields": sorted(events), "forged_signature": s["forged_signature"], "has_original": "original" in s,
        "features": risk_features(r.checks, r.analyses, r.signature, r.forensics),
        "checks": {c.id: c.status for c in r.checks},
        "category": r.risk.category, "score": r.risk.score,
        "signature_outcome": r.signature.outcome if r.signature else None,
        "field_flags": {k: bool(a.suspicious) for k, a in r.analyses.items()},
        "ocr": {
            "amount_figures_ok": f["amount_numeric"].value is not None and amount_value(gt) is not None and
            abs(f["amount_numeric"].value - amount_value(gt)) < 0.5,
            "amount_words_ok": f["amount_words"].value is not None and words_gt is not None and
            abs(f["amount_words"].value - words_gt) < 0.5,
            "date_ok": f["date"].value is not None and f["date"].value == gt.date,
            "ifsc_ok": f["ifsc"].value == gt.ifsc, "cheque_number_ok": f["cheque_number"].value == gt.cheque_number,
            "engines": {k: f[k].ocr_engine for k in ("amount_numeric", "amount_words", "date", "ifsc")},
        },
        "gt_consistent_amounts": amount_value(gt) is not None and words_gt is not None and abs(amount_value(gt) - words_gt) < 0.5,
        "seconds": elapsed, "timings": r.timings,
    }


def fit(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from cheque_forensics.risk_assessor import FEATURES
    names = list(FEATURES)
    X = np.array([[r["features"][n] for n in names] for r in rows])
    y = np.array([r["label"] for r in rows])
    mean, std = X.mean(0), X.std(0) + 1e-6
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit((X - mean) / std, y)
    p = lr.predict_proba((X - mean) / std)[:, 1] * 100
    ts = np.unique(np.round(p, 2))
    pos = p[y == 1]
    t_low = max([t for t in ts if np.mean(pos >= t) >= 0.95], default=float(ts.min()))
    t_high = min([t for t in ts if (p >= t).sum() >= 10 and np.mean(y[p >= t]) >= 0.90], default=99.0)
    t_high = max(t_high, t_low + 1)
    uni = {}
    for i, n in enumerate(names):
        uni[n] = float(roc_auc_score(y, X[:, i])) if X[:, i].std() > 0 else 0.5
    return {"features": names, "mean": mean.tolist(), "std": std.tolist(), "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]), "t_low": float(t_low), "t_high": float(t_high),
            "univariate_auc": uni, "val_auc": float(roc_auc_score(y, p)), "n_val": len(rows),
            "positive_rate": float(y.mean()),
            "note": "fitted on synthetic validation cheques; not a real-world fraud probability"}


def evaluate(rows, cal):
    from sklearn.metrics import roc_auc_score
    y = np.array([r["label"] for r in rows])
    s = np.array([r["score"] if r["score"] is not None else np.nan for r in rows])
    cats = np.array([r["category"] for r in rows])
    out = {"n": len(rows), "n_altered_or_forged": int(y.sum())}
    if not np.isnan(s).any():
        out["risk_score_roc_auc"] = float(roc_auc_score(y, s))
    out["category_counts"] = {lab: {c: int(((cats == c) & (y == v)).sum()) for c in ("LOW RISK", "REQUIRES REVIEW", "HIGH RISK")}
                              for lab, v in (("genuine", 0), ("altered", 1))}
    high, low = cats == "HIGH RISK", cats == "LOW RISK"
    out["high_risk_precision"] = float(y[high].mean()) if high.any() else None
    out["high_risk_recall"] = float(high[y == 1].mean())
    out["altered_classified_low_rate"] = float(low[y == 1].mean())
    out["genuine_classified_low_rate"] = float(low[y == 0].mean())
    out["flagged_for_review_or_high_recall"] = float((~low)[y == 1].mean())
    kinds = {}
    for r in rows:
        for k in r["kinds"]:
            kinds.setdefault(k, []).append(r["category"] != "LOW RISK")
    out["detection_by_kind(not LOW)"] = {k: {"rate": float(np.mean(v)), "n": len(v)} for k, v in sorted(kinds.items())}
    # amount consistency check: does it fire when the ground truth amounts disagree?
    inc = [r for r in rows if not r["gt_consistent_amounts"]]
    con = [r for r in rows if r["gt_consistent_amounts"]]
    out["amount_consistency_check"] = {
        "n_inconsistent": len(inc), "fail_rate_when_inconsistent": float(np.mean([r["checks"]["amount_consistency"] == "FAIL" for r in inc])) if inc else None,
        "flag_rate_when_inconsistent(FAIL or WARN)": float(np.mean([r["checks"]["amount_consistency"] in ("FAIL", "WARN") for r in inc])) if inc else None,
        "false_fail_rate_when_consistent": float(np.mean([r["checks"]["amount_consistency"] == "FAIL" for r in con])) if con else None}
    out["ocr_accuracy"] = {k: float(np.mean([r["ocr"][k] for r in rows])) for k in
                           ("amount_figures_ok", "amount_words_ok", "date_ok", "ifsc_ok", "cheque_number_ok")}
    sig = [r for r in rows if r["signature_outcome"] not in (None, "unavailable")]
    if sig:
        forged = [r for r in sig if r["forged_signature"]]
        genuine = [r for r in sig if not r["forged_signature"]]
        out["signature_on_cheques"] = {
            "n_genuine": len(genuine), "n_forged": len(forged),
            "forged_flagged_mismatch": float(np.mean([r["signature_outcome"] == "suspicious_mismatch" for r in forged])) if forged else None,
            "forged_not_accepted(mismatch or inconclusive)": float(np.mean([r["signature_outcome"] != "likely_match" for r in forged])) if forged else None,
            "genuine_accepted": float(np.mean([r["signature_outcome"] == "likely_match" for r in genuine])) if genuine else None,
            "genuine_flagged_mismatch": float(np.mean([r["signature_outcome"] == "suspicious_mismatch" for r in genuine])) if genuine else None}
    orig = [r for r in rows if r["has_original"]]
    if orig:
        yo = np.array([bool(r["label"]) and not (r["forged_signature"] and len(r["kinds"]) == 1) for r in orig],
                      dtype=bool)   # pixel-altered cheques (a forged signature alone is not a pixel change)
        fo = np.array([r["features"]["original_difference"] > 0 for r in orig])
        out["trusted_original_comparison"] = {"n": len(orig), "recall_pixel_alterations": float(fo[yo].mean()) if yo.any() else None,
                                              "false_alarm_rate": float(fo[~yo].mean()) if (~yo).any() else None}
    secs = np.array([r["seconds"] for r in rows])
    out["processing_seconds_per_cheque"] = {"mean": float(secs.mean()), "median": float(np.median(secs)),
                                            "p95": float(np.percentile(secs, 95))}
    stages = {}
    for r in rows:
        for k, v in r["timings"].items():
            stages.setdefault(k, []).append(v)
    out["stage_seconds_median"] = {k: float(np.median(v)) for k, v in stages.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-doc", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=240)
    ap.add_argument("--n-test", type=int, default=240)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    t = Timer()
    stats = parallel_map(_doc_stats, list(range(args.n_doc)), workers=args.workers + 2)
    doc_thr = {k: max(0.3, float(np.percentile([s[k] for s in stats], 95))) for k in stats[0]}
    with open(TAMPER_CALIBRATION, encoding="utf-8") as fh:
        tcal = json.load(fh)
    tcal["document_maps"] = doc_thr
    with open(TAMPER_CALIBRATION, "w", encoding="utf-8") as fh:
        json.dump(tcal, fh, indent=2)
    print(f"document-map thresholds {doc_thr} ({t()})", flush=True)
    if os.path.exists(RISK_CALIBRATION):
        os.remove(RISK_CALIBRATION)
    val = parallel_map(_run, [(i, "val") for i in range(args.n_val)], workers=args.workers, chunksize=4)
    cal = fit(val)
    with open(RISK_CALIBRATION, "w", encoding="utf-8") as fh:
        json.dump(cal, fh, indent=2)
    print(f"risk model fitted: val AUC {cal['val_auc']:.3f}, t_low {cal['t_low']:.1f}, t_high {cal['t_high']:.1f} ({t()})", flush=True)
    test = parallel_map(_run, [(i, "test") for i in range(args.n_test)], workers=args.workers, chunksize=4)
    metrics = {"split": "test: templates, handwriting fonts and CEDAR signers unseen in training/calibration",
               "risk_calibration": {k: cal[k] for k in ("t_low", "t_high", "val_auc", "univariate_auc", "n_val")},
               "learned_weights": dict(zip(cal["features"], [round(c, 3) for c in cal["coef"]])),
               "document_map_thresholds": doc_thr, "test": evaluate(test, cal)}
    save_metrics("system_evaluation", metrics)
    print(json.dumps(metrics["test"], indent=1))


if __name__ == "__main__":
    main()
