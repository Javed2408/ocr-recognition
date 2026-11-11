"""Train and evaluate the Siamese signature verifier on CEDAR with writer-disjoint splits.

Writers 1-35 train, 36-45 validation (threshold calibration and model
selection), 46-55 test (reported metrics). Training pairs: genuine/genuine of
the same writer (positive), genuine/skilled-forgery of the same writer and
genuine/genuine of different writers (negative).

    python -m training.train_siamese --epochs 25
"""
import argparse
import json
import os

import cv2
import numpy as np

from training.common import Timer, parallel_map, save_metrics, seed_everything
from cheque_forensics.config import SIAMESE_WEIGHTS, SIGNATURE_CALIBRATION, SYNTH_DIR
from cheque_forensics.synth import signatures as sigs

N_REFS = 4


def _load(args):
    writer, idx, forged = args
    from cheque_forensics.signature_extractor import preprocess_signature
    return (preprocess_signature(sigs.load_cedar(writer, idx, forged)) * 255).astype(np.uint8)


def load_cedar_arrays():
    path = os.path.join(SYNTH_DIR, "cedar_preprocessed.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["gen"], d["forg"]
    writers = sigs.cedar_writers()
    assert len(writers) == 55, f"expected 55 CEDAR writers, found {len(writers)}"
    items = [(w, i, f) for f in (False, True) for w in range(1, 56) for i in range(1, 25)]
    arrs = parallel_map(_load, items, chunksize=16)
    gen = np.stack(arrs[:55 * 24]).reshape(55, 24, *arrs[0].shape)
    forg = np.stack(arrs[55 * 24:]).reshape(55, 24, *arrs[0].shape)
    os.makedirs(SYNTH_DIR, exist_ok=True)
    np.savez_compressed(path, gen=gen, forg=forg)
    return gen, forg


def augment(img, rng):
    """Natural-variation augmentation: small rotation/scale/shear/shift and stroke width."""
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), rng.uniform(0.88, 1.08))
    m[0, 1] += rng.uniform(-0.12, 0.12)
    m[:, 2] += rng.uniform(-0.05, 0.05, 2) * [w, h]
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    r = rng.random()
    if r < 0.2:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))
    elif r < 0.35:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    return out


def sample_pairs(gen, forg, writers, n, rng):
    """Half positive, quarter skilled forgeries, quarter random (other writer) forgeries."""
    a, b, y = [], [], []
    for k in range(n):
        w = writers[rng.integers(len(writers))]
        i, j = rng.choice(24, 2, replace=False)
        kind = k % 4
        if kind < 2:
            a.append(gen[w - 1, i]); b.append(gen[w - 1, j]); y.append(1)
        elif kind == 2:
            a.append(gen[w - 1, i]); b.append(forg[w - 1, j]); y.append(0)
        else:
            other = writers[rng.integers(len(writers))]
            while other == w:
                other = writers[rng.integers(len(writers))]
            a.append(gen[w - 1, i]); b.append(gen[other - 1, j]); y.append(0)
    return np.stack(a), np.stack(b), np.array(y, np.float32)


def embed_all(model, arr, batch=128):
    import torch
    from cheque_forensics.device import get_device
    model.eval()
    flat = arr.reshape(-1, *arr.shape[-2:])
    outs = []
    with torch.no_grad():
        for k in range(0, len(flat), batch):
            xb = torch.from_numpy(flat[k:k + batch, None].astype(np.float32) / 255.0).to(get_device())
            outs.append(model.embedder(xb).cpu().numpy())
    return np.concatenate(outs).reshape(*arr.shape[:-2], -1)


def verification_scores(emb_gen, emb_forg, writers, rng_seed=0):
    """For each writer: N_REFS genuine references; score genuine queries, skilled and random forgeries.

    Returns dict of arrays of (mean distance, normalised distance) per query type.
    """
    rng = np.random.default_rng(rng_seed)
    out = {"genuine": [], "skilled": [], "random": []}
    for w in writers:
        refs_idx = rng.choice(24, N_REFS, replace=False)
        refs = emb_gen[w - 1, refs_idx]
        intra = [np.linalg.norm(refs[i] - refs[j]) for i in range(N_REFS) for j in range(i + 1, N_REFS)]
        spread = float(np.mean(intra))

        def score(q):
            d = float(np.linalg.norm(refs - q, axis=1).mean())
            return d, d / max(spread, 1e-3)
        for i in set(range(24)) - set(refs_idx):
            out["genuine"].append(score(emb_gen[w - 1, i]))
        for i in range(24):
            out["skilled"].append(score(emb_forg[w - 1, i]))
        others = [o for o in writers if o != w]
        for o in rng.choice(others, 10):
            out["random"].append(score(emb_gen[o - 1, rng.integers(24)]))
    return {k: np.array(v) for k, v in out.items()}


def rates(genuine, forged, t):
    """FAR: forgeries accepted (score <= t). FRR: genuine rejected (score > t)."""
    return float(np.mean(forged <= t)), float(np.mean(genuine > t))


def eer(genuine, forged):
    ts = np.unique(np.concatenate([genuine, forged]))
    best = min(ts, key=lambda t: abs(np.subtract(*rates(genuine, forged, t))))
    far, frr = rates(genuine, forged, best)
    return (far + frr) / 2, float(best)


def summarize(scores, col, thresholds):
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    g = scores["genuine"][:, col]
    out = {}
    for kind in ("skilled", "random"):
        f = scores[kind][:, col]
        y = np.r_[np.zeros(len(g)), np.ones(len(f))]          # positive class = forgery
        s = np.r_[g, f]
        e, t_e = eer(g, f)
        far, frr = rates(g, f, thresholds["eer"])
        pred = s > thresholds["eer"]
        acc_t, rej_t = thresholds["accept"], thresholds["reject"]
        inconclusive = float(np.mean((s > acc_t) & (s < rej_t)))
        decided = (s <= acc_t) | (s >= rej_t)
        out[kind] = {
            "roc_auc": float(roc_auc_score(y, s)), "eer": float(e),
            "far_at_val_threshold": far, "frr_at_val_threshold": frr,
            "precision_forgery": float(precision_score(y, pred, zero_division=0)),
            "recall_forgery": float(recall_score(y, pred, zero_division=0)),
            "three_way": {
                "inconclusive_rate": inconclusive,
                "far_decided": float(np.mean(f[(f <= acc_t) | (f >= rej_t)] <= acc_t)) if decided[len(g):].any() else None,
                "frr_decided": float(np.mean(g[(g <= acc_t) | (g >= rej_t)] >= rej_t)) if decided[:len(g)].any() else None,
            },
            "n_genuine": len(g), "n_forged": len(f),
        }
    return out


def calibrate(scores, col, target=0.05):
    """Thresholds from validation data: EER point, accept (FAR_skilled <= target), reject (FRR <= target)."""
    from sklearn.linear_model import LogisticRegression
    g, f = scores["genuine"][:, col], np.r_[scores["skilled"][:, col], scores["random"][:, col]]
    fs = scores["skilled"][:, col]
    _, t_eer = eer(g, fs)
    ts = np.sort(np.unique(np.r_[g, fs]))
    accept = max([t for t in ts if np.mean(fs <= t) <= target], default=float(ts[0]))
    reject = min([t for t in ts if np.mean(g > t) <= target], default=float(ts[-1]))
    if accept >= reject:
        accept = reject = t_eer
    lr = LogisticRegression().fit(np.r_[g, f][:, None], np.r_[np.zeros(len(g)), np.ones(len(f))])
    return {"eer": float(t_eer), "accept": float(accept), "reject": float(reject),
            "logistic": {"coef": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])},
            "genuine_spread_p90": float(np.percentile(g, 90))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--pairs", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)
    import torch
    from cheque_forensics.device import get_device, save_checkpoint
    from cheque_forensics.nets import SiameseNetwork, contrastive_loss
    seed_everything(0)
    t = Timer()
    gen, forg = load_cedar_arrays()
    print(f"CEDAR loaded {gen.shape} {forg.shape} ({t()})", flush=True)
    W = sigs.WRITER_SPLITS
    device = get_device()
    model = SiameseNetwork().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    steps = args.epochs * (args.pairs // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    rng = np.random.default_rng(0)
    best, best_state, best_col = 1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        a, b, y = sample_pairs(gen, forg, W["train"], args.pairs, rng)
        losses = []
        for k in range(0, args.pairs, args.batch):
            xa = np.stack([augment(im, rng) for im in a[k:k + args.batch]]).astype(np.float32) / 255.0
            xb = np.stack([augment(im, rng) for im in b[k:k + args.batch]]).astype(np.float32) / 255.0
            xa, xb = torch.from_numpy(xa[:, None]).to(device), torch.from_numpy(xb[:, None]).to(device)
            yt = torch.from_numpy(y[k:k + args.batch]).to(device)
            _, _, d = model(xa, xb)
            loss = contrastive_loss(d, yt, margin=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        sc = verification_scores(embed_all(model, gen), embed_all(model, forg), W["val"])
        e_raw = eer(sc["genuine"][:, 0], sc["skilled"][:, 0])[0]
        e_norm = eer(sc["genuine"][:, 1], sc["skilled"][:, 1])[0]
        print(f"epoch {ep + 1}/{args.epochs} loss {np.mean(losses):.4f} val skilled EER raw {e_raw:.4f} "
              f"norm {e_norm:.4f} ({t()})", flush=True)
        if min(e_raw, e_norm) < best:
            best, best_col = min(e_raw, e_norm), int(e_norm < e_raw)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    eg, ef = embed_all(model, gen), embed_all(model, forg)
    val_scores = verification_scores(eg, ef, W["val"])
    cal = calibrate(val_scores, best_col)
    cal.update({"aggregation": ["mean_distance", "spread_normalised_distance"][best_col], "n_refs_calibrated": N_REFS,
                "target_rate": 0.05})
    test_scores = verification_scores(eg, ef, W["test"], rng_seed=1)
    val_metrics = summarize(val_scores, best_col, cal)
    test_metrics = summarize(test_scores, best_col, cal)
    save_checkpoint(model, SIAMESE_WEIGHTS, epochs=args.epochs, val_skilled_eer=best)
    with open(SIGNATURE_CALIBRATION, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    save_metrics("signature_verification", {
        "dataset": "CEDAR (55 writers x 24 genuine + 24 skilled forgeries), writer-disjoint splits",
        "splits": {k: [v[0], v[-1]] for k, v in W.items()}, "calibration": cal,
        "validation": val_metrics, "test": test_metrics, "epochs": args.epochs,
        "protocol": f"{N_REFS} genuine references per writer; score = {cal['aggregation']} to references",
    })
    print(json.dumps({"val": val_metrics, "test": test_metrics, "calibration": cal}, indent=1))


if __name__ == "__main__":
    main()
