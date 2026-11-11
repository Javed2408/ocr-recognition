"""Phase 1 evaluation on the synthetic TEST split (templates, fonts and signers unseen in training).

    python -m training.evaluate_fields --n 120
"""
import argparse

import cv2
import numpy as np

from training.common import Timer, save_metrics
from cheque_forensics.config import FIELDS


def photo_of(chq, rng):
    """Composite a cheque scan into a perspective photo. Returns (photo, true corners)."""
    from cheque_forensics.synth.backgrounds import desk_background
    h, w = chq.shape[:2]
    H0 = int(rng.integers(800, 1100))
    W0 = int(H0 * rng.uniform(1.3, 1.7))
    bg = desk_background(H0, W0, rng)
    sw = rng.uniform(0.6, 0.85) * W0
    sh = sw * h / w
    cx, cy = W0 / 2 + rng.uniform(-60, 60), H0 / 2 + rng.uniform(-40, 40)
    dst = np.float32([[cx - sw / 2, cy - sh / 2], [cx + sw / 2, cy - sh / 2], [cx + sw / 2, cy + sh / 2],
                      [cx - sw / 2, cy + sh / 2]]) + rng.uniform(-40, 40, (4, 2)).astype(np.float32)
    a = np.deg2rad(rng.uniform(-15, 15))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = ((dst - [cx, cy]) @ R.T + [cx, cy]).astype(np.float32)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    m = cv2.warpPerspective(np.ones((h, w), np.uint8), M, (W0, H0))
    return np.where(m[..., None] > 0, cv2.warpPerspective(chq, M, (W0, H0)), bg), dst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args(argv)
    from cheque_forensics.cheque_detector import detect_cheque
    from cheque_forensics.cheque_preprocessor import preprocess
    from cheque_forensics.field_extractor import layout_score, locate_fields, segmentation_available
    from cheque_forensics.synth.tampering import make_sample

    t = Timer()
    rng = np.random.default_rng(2024)
    corner = {"classical": [], "classical+unet": []}
    iou = {"segmentation+structure": {f: [] for f in FIELDS}, "prior+structure": {f: [] for f in FIELDS}}
    orient = []
    for i in range(args.n):
        s = make_sample(i, "test", tamper_prob=0.3, seed=99)
        chq = s["image"]
        photo, dst = photo_of(chq, rng)
        tol = 0.02 * np.linalg.norm(dst[1] - dst[0])
        for name, use in (("classical", False), ("classical+unet", True)):
            det = detect_cheque(photo, use_model=use)
            err = float(np.abs(det.quad - dst).max()) if det.method != "full_image" else float("inf")
            corner[name].append((err, err <= tol))
        pre = preprocess(chq, use_models=True, orientation_scorer=layout_score if segmentation_available() else None)
        for name, use in (("segmentation+structure", True), ("prior+structure", False)):
            located, _ = locate_fields(pre, use_model=use)
            for f in FIELDS:
                iou[name][f].append(located[f][0].iou(s["fields"][f]))
        if segmentation_available():
            flipped = cv2.rotate(chq, cv2.ROTATE_180)
            orient.append(layout_score(chq) > layout_score(flipped) + 0.02)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{args.n} ({t()})", flush=True)

    metrics = {"n": args.n, "split": "test (unseen templates, fonts, signers)", "boundary_detection": {}, "field_iou": {}}
    for name, vals in corner.items():
        errs = np.array([e for e, _ in vals])
        metrics["boundary_detection"][name] = {
            "success_rate(corner error <= 2% of width)": float(np.mean([ok for _, ok in vals])),
            "median_corner_error_px": float(np.median(errs[np.isfinite(errs)])) if np.isfinite(errs).any() else None,
        }
    for name, per in iou.items():
        metrics["field_iou"][name] = {
            "mean_iou": {f: float(np.mean(v)) for f, v in per.items()},
            "localization_accuracy(iou>=0.5)": {f: float(np.mean(np.array(v) >= 0.5)) for f, v in per.items()},
            "overall_mean_iou": float(np.mean([np.mean(v) for v in per.values()])),
        }
    if orient:
        metrics["orientation_accuracy_upright_vs_180"] = float(np.mean(orient))
    save_metrics("phase1_fields", metrics)
    import json
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
