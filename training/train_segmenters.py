"""Train the cheque-boundary U-Net and the field-segmentation U-Net on synthetic cheques.

Splits follow the synthetic template split (train/val/test layouts never overlap).

    python -m training.train_segmenters --n-train 2000 --n-val 300 --epochs 8
"""
import argparse
import os

import cv2
import numpy as np

from training.common import PROJECT_DIR, Timer, parallel_map, save_metrics, seed_everything  # noqa: F401
from cheque_forensics.config import (CHEQUE_SEG_SIZE, CHEQUE_SEG_WEIGHTS, FIELD_SEG_SIZE, FIELD_SEG_WEIGHTS,
                                     FIELDS, SEG_BASE_CHANNELS, SYNTH_DIR)
from cheque_forensics.schema import BBox

# Larger zones are painted first so small fields (cheque number inside MICR) win overlaps.
PAINT_ORDER = ["signature", "amount_words", "payee", "micr", "account_number", "ifsc", "date",
               "amount_numeric", "cheque_number"]
CLASS_ID = {name: i + 1 for i, name in enumerate(FIELDS)}


def jitter_crop(image, fields, rng, max_frac=0.025):
    """Simulate an imperfect boundary crop: random corner jitter + re-warp; boxes follow."""
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    jit = rng.uniform(-max_frac, max_frac, (4, 2)) * [w, h]
    M = cv2.getPerspectiveTransform((src + jit).astype(np.float32), src)
    out = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    new = {}
    for name, b in fields.items():
        pts = np.float32([[b.x, b.y], [b.x2, b.y], [b.x2, b.y2], [b.x, b.y2]]).reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        new[name] = BBox.from_xyxy(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()).clip(w, h)
    return out, new


def field_item(args):
    index, split = args
    from cheque_forensics.synth.tampering import make_sample
    s = make_sample(index, split, tamper_prob=0.3, seed=11)
    rng = np.random.default_rng(index + (0 if split == "train" else 10_000_000))
    img, fields = jitter_crop(s["image"], s["fields"], rng)
    H, W = FIELD_SEG_SIZE
    sy, sx = H / img.shape[0], W / img.shape[1]
    x = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    mask = np.zeros((H, W), np.uint8)
    for name in PAINT_ORDER:
        b = fields[name].scale(sx, sy)
        mask[max(0, b.y):b.y2, max(0, b.x):b.x2] = CLASS_ID[name]
    return x, mask


def cheque_item(args):
    index, split = args
    from cheque_forensics.synth.backgrounds import desk_background
    from cheque_forensics.synth.tampering import make_sample
    s = make_sample(index, split, tamper_prob=0.2, seed=12)
    rng = np.random.default_rng(index + (0 if split == "train" else 10_000_000) + 777)
    chq = s["image"]
    h, w = chq.shape[:2]
    mode = rng.random()
    if mode < 0.15:     # already-cropped scan: whole frame is cheque
        img, mask = chq, np.ones((h, w), np.uint8)
    else:
        H0 = int(rng.integers(700, 1200))
        W0 = int(H0 * rng.uniform(1.0, 1.8))
        bg = desk_background(H0, W0, rng) if mode > 0.3 else np.full((H0, W0, 3), rng.integers(200, 256), np.uint8)
        sw = rng.uniform(0.5, 0.92) * W0
        sh = sw * h / w
        cx, cy = W0 / 2 + rng.uniform(-0.1, 0.1) * W0, H0 / 2 + rng.uniform(-0.1, 0.1) * H0
        dst = np.float32([[cx - sw / 2, cy - sh / 2], [cx + sw / 2, cy - sh / 2], [cx + sw / 2, cy + sh / 2],
                          [cx - sw / 2, cy + sh / 2]]) + rng.uniform(-0.06, 0.06, (4, 2)).astype(np.float32) * sw
        a = np.deg2rad(rng.uniform(-25, 25))
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        dst = ((dst - [cx, cy]) @ R.T + [cx, cy]).astype(np.float32)
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
        mask = cv2.warpPerspective(np.ones((h, w), np.uint8), M, (W0, H0))
        img = np.where(mask[..., None] > 0, cv2.warpPerspective(chq, M, (W0, H0)), bg)
        if rng.random() < 0.5:   # shadow / uneven light
            g = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.9, 1.1), W0, dtype=np.float32)[None, :, None]
            img = np.clip(img * g, 0, 255).astype(np.uint8)
    H, W = CHEQUE_SEG_SIZE
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


def build(kind, n_train, n_val):
    path = os.path.join(SYNTH_DIR, f"{kind}_seg_{n_train}_{n_val}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["xt"], d["mt"], d["xv"], d["mv"]
    fn = field_item if kind == "field" else cheque_item
    t = Timer()
    train = parallel_map(fn, [(i, "train") for i in range(n_train)])
    val = parallel_map(fn, [(i, "val") for i in range(n_val)])
    xt, mt = np.stack([a for a, _ in train]), np.stack([b for _, b in train])
    xv, mv = np.stack([a for a, _ in val]), np.stack([b for _, b in val])
    os.makedirs(SYNTH_DIR, exist_ok=True)
    np.savez_compressed(path, xt=xt, mt=mt, xv=xv, mv=mv)
    print(f"built {kind} dataset {xt.shape} / {xv.shape} in {t()}")
    return xt, mt, xv, mv


def augment_batch(x, rng):
    """Photometric augmentation on a uint8 NHWC batch."""
    x = x.astype(np.float32)
    x = x * rng.uniform(0.8, 1.15, (len(x), 1, 1, 1)) + rng.uniform(-20, 20, (len(x), 1, 1, 1))
    x = x * rng.uniform(0.9, 1.1, (len(x), 1, 1, 3))      # colour cast
    x += rng.normal(0, rng.uniform(0, 6), x.shape)
    return np.clip(x, 0, 255) / 255.0


def train(kind, xt, mt, xv, mv, epochs, batch=16, lr=2e-3):
    import torch
    import torch.nn.functional as F
    from cheque_forensics.device import get_device, save_checkpoint
    from cheque_forensics.nets import UNet
    device = get_device()
    n_cls = len(FIELDS) + 1 if kind == "field" else 1
    model = UNet(3, n_cls, base=SEG_BASE_CHANNELS, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(xt) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    rng = np.random.default_rng(0)
    if kind == "field":
        freq = np.bincount(mt.ravel(), minlength=n_cls).astype(np.float64)
        weights = torch.tensor((freq.sum() / (freq + 1)) ** 0.5, dtype=torch.float32)
        weights = (weights / weights.mean()).to(device)
    t = Timer()
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(xt))
        losses = []
        for k in range(0, len(order) - batch + 1, batch):
            idx = order[k:k + batch]
            xb = torch.from_numpy(augment_batch(xt[idx], rng).transpose(0, 3, 1, 2)).float().to(device)
            yb = torch.from_numpy(mt[idx].astype(np.int64)).to(device)
            out = model(xb)
            if kind == "field":
                loss = F.cross_entropy(out, yb, weight=weights)
                prob = out.softmax(1)
                onehot = F.one_hot(yb, n_cls).permute(0, 3, 1, 2).float()
                inter = (prob * onehot).sum((0, 2, 3))
                dice = 1 - ((2 * inter + 1) / (prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1))[1:].mean()
                loss = loss + dice
            else:
                yf = yb.float().unsqueeze(1)
                prob = torch.sigmoid(out)
                loss = F.binary_cross_entropy_with_logits(out, yf) + \
                    1 - (2 * (prob * yf).sum() + 1) / (prob.sum() + yf.sum() + 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        score = evaluate(kind, model, xv, mv)
        print(f"[{kind}] epoch {ep + 1}/{epochs} loss {np.mean(losses):.4f} val {score['summary']:.4f} ({t()})", flush=True)
        if score["summary"] > best:
            best = score["summary"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = score
    model.load_state_dict(best_state)
    path = FIELD_SEG_WEIGHTS if kind == "field" else CHEQUE_SEG_WEIGHTS
    save_checkpoint(model, path, kind=kind, epochs=epochs, n_train=len(xt), val=best_metrics)
    return model, best_metrics


def evaluate(kind, model, xv, mv, batch=32):
    import torch
    from cheque_forensics.device import get_device
    model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(xv), batch):
            xb = torch.from_numpy(xv[k:k + batch].astype(np.float32).transpose(0, 3, 1, 2) / 255.0).to(get_device())
            out = model(xb)
            preds.append((out.argmax(1) if kind == "field" else (out[:, 0] > 0).long()).cpu().numpy())
    pred = np.concatenate(preds)
    if kind == "cheque":
        inter = ((pred == 1) & (mv == 1)).sum((1, 2))
        union = ((pred == 1) | (mv == 1)).sum((1, 2))
        iou = inter / np.maximum(union, 1)
        return {"summary": float(iou.mean()), "mask_iou_mean": float(iou.mean()), "mask_iou_min": float(iou.min())}
    per_class = {}
    for name in FIELDS:
        c = CLASS_ID[name]
        inter = ((pred == c) & (mv == c)).sum()
        union = ((pred == c) | (mv == c)).sum()
        per_class[name] = float(inter / union) if union else float("nan")
    return {"summary": float(np.nanmean(list(per_class.values()))), "pixel_iou": per_class}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--only", choices=["field", "cheque"])
    args = ap.parse_args(argv)
    seed_everything(0)
    results = {}
    for kind in ([args.only] if args.only else ["cheque", "field"]):
        xt, mt, xv, mv = build(kind, args.n_train if kind == "field" else args.n_train // 2, args.n_val)
        _, metrics = train(kind, xt, mt, xv, mv, args.epochs)
        results[kind] = metrics
        save_metrics(f"{kind}_segmenter", {"val": metrics, "n_train": len(xt), "n_val": len(xv),
                                           "epochs": args.epochs, "data": "synthetic, template-disjoint val split"})
    return results


if __name__ == "__main__":
    main()
