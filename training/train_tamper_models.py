"""Train the tampering CNN, the anomaly autoencoder and the tampering U-Net; calibrate and evaluate.

Data: synthetic cheques with controlled alterations (cheque_forensics/synth/tampering.py).
Train/val/test use disjoint cheque templates, handwriting font families and CEDAR signers.

1. Build 64x256 windows from the handwritten fields (jittered ground-truth boxes).
2. Train: classifier (window altered?), autoencoder (genuine windows only), U-Net (altered pixels).
3. Calibrate on VALIDATION cheques with the full field analysis (analyze_field): each detector's
   threshold is the 95th percentile of its score on unaltered fields (5% false-alarm rate).
4. Report on TEST cheques: ROC-AUC, precision, recall, F1 per detector and combined; localisation IoU.

    python -m training.train_tamper_models --n-train 1500 --n-val 300 --n-test 300
"""
import argparse
import json

import cv2
import numpy as np

from training.common import Timer, parallel_map, save_metrics, seed_everything
from cheque_forensics.config import (HANDWRITTEN_FIELDS, TAMPER_AUTOENCODER_WEIGHTS, TAMPER_CALIBRATION,
                                     TAMPER_CLASSIFIER_WEIGHTS, TAMPER_UNET_WEIGHTS)
from cheque_forensics.schema import BBox

SEED = 21
MIN_TAMPER_PX = 15


def _windows_for(args):
    index, split = args
    from cheque_forensics.synth.tampering import make_sample
    from cheque_forensics.tamper_models import PH, field_input, make_windows
    s = make_sample(index, split, tamper_prob=0.6, seed=SEED)
    rng = np.random.default_rng(index * 31 + (0 if split == "train" else 999_999))
    H, W = s["image"].shape[:2]
    xs, ms, labels, genuine_sample = [], [], [], s["label"] == 0
    for f in HANDWRITTEN_FIELDS:
        b = s["fields"][f]
        jx, jy = (rng.uniform(-0.04, 0.04, 2) * [b.w, b.h]).astype(int)
        b = BBox(b.x + jx, b.y + jy, int(b.w * rng.uniform(0.96, 1.06)), int(b.h * rng.uniform(0.95, 1.1))).clip(W, H)
        crop, mcrop = b.crop(s["image"]), b.crop(s["mask"])
        x = field_input(crop)
        m = cv2.resize(mcrop, (x.shape[2], PH), interpolation=cv2.INTER_NEAREST)
        starts, wins = make_windows(x)
        _, mwins = make_windows(np.repeat(m[None].astype(np.float32), 4, 0))
        mwins = mwins[:, 0]
        pos = [k for k in range(len(wins)) if mwins[k].sum() >= MIN_TAMPER_PX]
        neg = [k for k in range(len(wins)) if mwins[k].sum() == 0]
        pick = pos[:2] + list(rng.choice(neg, min(len(neg), 1 if pos else 2), replace=False))
        for k in pick:
            xs.append((wins[k] * 255).astype(np.uint8))
            ms.append((mwins[k] > 0).astype(np.uint8))
            labels.append(int(mwins[k].sum() >= MIN_TAMPER_PX))
    return xs, ms, labels, [genuine_sample] * len(labels)


def build(split, n):
    parts = parallel_map(_windows_for, [(i, split) for i in range(n)], chunksize=8)
    x = np.stack([w for p in parts for w in p[0]])
    m = np.stack([w for p in parts for w in p[1]])
    y = np.array([w for p in parts for w in p[2]], np.float32)
    g = np.array([w for p in parts for w in p[3]], bool)
    return x, m, y, g


def augment(xb, rng):
    xb = xb.copy()
    xb[:, :3] = np.clip(xb[:, :3] * rng.uniform(0.85, 1.12, (len(xb), 1, 1, 1)) +
                        rng.uniform(-0.06, 0.06, (len(xb), 3, 1, 1)), 0, 1)
    return xb


def train_models(xt, mt, yt, gt, xv, mv, yv, epochs, batch=32):
    import torch
    import torch.nn.functional as F
    from cheque_forensics.device import get_device, save_checkpoint
    from cheque_forensics.nets import ConvAutoencoder, TamperClassifier, UNet
    from cheque_forensics.tamper_models import UNET_BASE, UNET_DEPTH
    dev = get_device()
    rng = np.random.default_rng(0)
    t = Timer()
    to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).to(dev)
    out = {}

    clf = TamperClassifier(4).to(dev)
    unet = UNet(4, 1, base=UNET_BASE, depth=UNET_DEPTH).to(dev)
    params = list(clf.parameters()) + list(unet.parameters())
    opt = torch.optim.AdamW(params, lr=1.5e-3, weight_decay=1e-4)
    steps = epochs * (len(xt) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=steps)
    pos_w = torch.tensor((1 - yt.mean()) / max(yt.mean(), 1e-3)).to(dev)
    best = (-1, None)
    for ep in range(epochs):
        clf.train(); unet.train()
        order = rng.permutation(len(xt))
        losses = []
        for k in range(0, len(order) - batch + 1, batch):
            idx = order[k:k + batch]
            xb = torch.from_numpy(augment(xt[idx].astype(np.float32) / 255.0, rng)).to(dev)
            yb = torch.from_numpy(yt[idx]).to(dev)
            mb = torch.from_numpy(mt[idx].astype(np.float32)).unsqueeze(1).to(dev)
            lc = F.binary_cross_entropy_with_logits(clf(xb), yb, pos_weight=pos_w)
            mo = unet(xb)
            pm = torch.sigmoid(mo)
            lu = F.binary_cross_entropy_with_logits(mo, mb, pos_weight=torch.tensor(5.0, device=dev)) + \
                1 - (2 * (pm * mb).sum() + 1) / (pm.sum() + mb.sum() + 1)
            loss = lc + lu
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        clf.eval(); unet.eval()
        with torch.no_grad():
            pv, iv = [], []
            for k in range(0, len(xv), 128):
                xb = to_t(xv[k:k + 128])
                pv.append(torch.sigmoid(clf(xb)).cpu().numpy())
                iv.append((torch.sigmoid(unet(xb))[:, 0] > 0.5).cpu().numpy())
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yv, np.concatenate(pv))
        pred = np.concatenate(iv)
        inter = (pred & (mv > 0)).sum()
        iou = inter / max((pred | (mv > 0)).sum(), 1)
        print(f"[clf+unet] epoch {ep + 1}/{epochs} loss {np.mean(losses):.4f} val window AUC {auc:.4f} "
              f"pixel IoU {iou:.4f} ({t()})", flush=True)
        if auc + iou > best[0]:
            best = (auc + iou, ({k: v.cpu().clone() for k, v in clf.state_dict().items()},
                                {k: v.cpu().clone() for k, v in unet.state_dict().items()}), auc, iou)
    clf.load_state_dict(best[1][0]); unet.load_state_dict(best[1][1])
    save_checkpoint(clf, TAMPER_CLASSIFIER_WEIGHTS, val_window_auc=best[2])
    save_checkpoint(unet, TAMPER_UNET_WEIGHTS, val_pixel_iou=best[3])
    out["window_val_auc"], out["window_val_pixel_iou"] = float(best[2]), float(best[3])

    ae = ConvAutoencoder(4).to(dev)
    genuine = xt[gt & (yt == 0)]
    opt = torch.optim.AdamW(ae.parameters(), lr=2e-3)
    for ep in range(max(4, epochs // 2)):
        ae.train()
        order = rng.permutation(len(genuine))
        losses = []
        for k in range(0, len(order) - batch + 1, batch):
            xb = torch.from_numpy(augment(genuine[order[k:k + batch]].astype(np.float32) / 255.0, rng)).to(dev)
            loss = F.l1_loss(ae(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[autoencoder] epoch {ep + 1} L1 {np.mean(losses):.4f} ({t()})", flush=True)
    save_checkpoint(ae, TAMPER_AUTOENCODER_WEIGHTS, n_genuine_windows=len(genuine))
    return out


def _field_scores(args):
    index, split = args
    import torch
    torch.set_num_threads(2)
    from cheque_forensics.field_alteration_detector import analyze_field
    from cheque_forensics.synth.tampering import make_sample
    s = make_sample(index, split, tamper_prob=0.5, seed=SEED + 1)
    rows = []
    for f in HANDWRITTEN_FIELDS:
        b = s["fields"][f]
        r = analyze_field(s["image"], f, b, use_dl=True)
        gt_px = int(b.crop(s["mask"]).sum())
        kinds = [e.kind for e in s["events"] if e.field == f]
        gt_boxes = [e.bbox for e in s["events"] if e.field == f]
        unet_iou = None
        if r.heatmap is not None and gt_px >= MIN_TAMPER_PX:
            from cheque_forensics.tamper_models import analyze_crop
            dl = analyze_crop(r.crop_box.crop(s["image"]), want_cam=False)
            if "unet_map" in dl:
                pred = dl["unet_map"] >= 0.5
                gt = r.crop_box.crop(s["mask"]) > 0
                unet_iou = float((pred & gt).sum() / max((pred | gt).sum(), 1))
        hits = None
        if gt_boxes:
            flagged = [i.bbox for i in r.indicators if i.bbox is not None and i.severity in ("medium", "high")]
            hits = any(fb.iou(g) > 0.1 or g.overlap_fraction(fb) > 0.5 or fb.overlap_fraction(g) > 0.5
                       for fb in flagged for g in gt_boxes)
        rows.append({"field": f, "label": int(gt_px >= MIN_TAMPER_PX), "kinds": kinds,
                     "scores": {k: float(v) for k, v in r.scores.items()}, "unet_iou": unet_iou, "hit": hits,
                     "flagged": r.suspicious})
    return rows


def field_level(split, n):
    parts = parallel_map(_field_scores, [(i, split) for i in range(n)], workers=8, chunksize=4)
    return [r for p in parts for r in p]


def calibrate(rows):
    from cheque_forensics.field_alteration_detector import fused_score
    cal = {}
    for det in ("classical", "spacing", "cnn", "ae", "unet"):
        cal[det] = {}
        for f in list(HANDWRITTEN_FIELDS) + ["all"]:
            g = [r["scores"][det] for r in rows if r["label"] == 0 and det in r["scores"] and (f == "all" or r["field"] == f)]
            if len(g) >= 10:   # floor: a detector silent on genuine fields still needs a positive threshold
                cal[det][f] = max(float(np.percentile(g, 95)), 1e-3)
            elif f != "all":
                cal[det][f] = None
    # fused score: strongest detector relative to its threshold, calibrated to the same 5% false-alarm rate
    cal["fused"] = {}
    for f in HANDWRITTEN_FIELDS:
        th = {d: cal[d].get(f) for d in ("classical", "spacing", "cnn", "ae", "unet")}
        th = {d: v for d, v in th.items() if v}
        for r in rows:
            if r["field"] == f:
                r["scores"]["fused"] = fused_score(r["scores"], th)
        g = [r["scores"]["fused"] for r in rows if r["field"] == f and r["label"] == 0]
        cal["fused"][f] = float(np.percentile(g, 95)) if len(g) >= 10 else 1.0
    cal["target_false_alarm_rate"] = 0.05
    return cal


def report(rows, cal):
    """Detector and fused-field metrics at the calibrated thresholds."""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    out = {}
    for f in list(HANDWRITTEN_FIELDS) + ["all"]:
        sel = [r for r in rows if f == "all" or r["field"] == f]
        y = np.array([r["label"] for r in sel])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        res = {"n_fields": len(sel), "n_altered": int(y.sum())}
        for det in ("classical", "spacing", "cnn", "ae", "unet", "fused"):
            s = np.array([r["scores"].get(det, np.nan) for r in sel])
            thr = cal.get(det, {}).get(f)
            if np.isnan(s).all() or (thr is None and f != "all"):
                continue
            ok = ~np.isnan(s)
            if f == "all":   # per-field thresholds
                pred = np.array([r["scores"].get(det, -1) >= (cal[det].get(r["field"]) or np.inf) for r in sel])
            else:
                pred = s >= thr
            res[det] = {"roc_auc": float(roc_auc_score(y[ok], s[ok])), "precision": float(precision_score(y, pred, zero_division=0)),
                        "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0))}
        pred = np.array([r["scores"].get("fused", 0) >= cal["fused"].get(r["field"], 1.0) for r in sel])
        res["field_flagged_suspicious"] = {"precision": float(precision_score(y, pred, zero_division=0)),
                                         "recall": float(recall_score(y, pred, zero_division=0)),
                                         "f1": float(f1_score(y, pred, zero_division=0)),
                                         "false_alarm_rate": float(pred[y == 0].mean())}
        ious = [r["unet_iou"] for r in sel if r["unet_iou"] is not None]
        hits = [r["hit"] for r in sel if r["hit"] is not None]
        res["localisation"] = {"unet_pixel_iou_mean": float(np.mean(ious)) if ious else None,
                               "indicator_hit_rate": float(np.mean(hits)) if hits else None}
        out[f] = res
    kinds = {}
    for r in rows:
        for k in r["kinds"]:
            kinds.setdefault(k, []).append(r["scores"].get("fused", 0) >= cal["fused"].get(r["field"], 1.0))
    out["recall_by_alteration_kind"] = {k: {"recall": float(np.mean(v)), "n": len(v)} for k, v in sorted(kinds.items())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args(argv)
    seed_everything(0)
    t = Timer()
    train_info = {}
    if not args.skip_train:
        xt, mt, yt, gt = build("train", args.n_train)
        xv, mv, yv, _ = build("val", args.n_val)
        print(f"windows: train {xt.shape} ({yt.mean():.2f} altered), val {xv.shape} ({t()})", flush=True)
        train_info = train_models(xt, mt, yt, gt, xv, mv, yv, args.epochs)
        train_info.update({"n_train_windows": len(xt), "n_val_windows": len(xv)})
    from cheque_forensics import tamper_models
    tamper_models._models.cache_clear()
    tamper_models.calibration.cache_clear()
    import os
    if os.path.exists(TAMPER_CALIBRATION):
        os.remove(TAMPER_CALIBRATION)
    val_rows = field_level("val", args.n_val)
    cal = calibrate(val_rows)
    with open(TAMPER_CALIBRATION, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    tamper_models.calibration.cache_clear()
    print(f"calibrated on {len(val_rows)} validation fields ({t()})", flush=True)
    test_rows = field_level("test", args.n_test)
    metrics = {"training": train_info, "calibration": cal, "validation": report(val_rows, cal),
               "test": report(test_rows, cal),
               "data": "synthetic cheques; template/font/signer-disjoint splits; field-level labels from pixel masks"}
    save_metrics("tamper_detection", metrics)
    print(json.dumps(metrics["test"], indent=1))


if __name__ == "__main__":
    main()
