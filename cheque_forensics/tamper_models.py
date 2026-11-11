"""Inference for the deep tampering models on handwritten field crops.

  * TamperClassifier - CNN: probability that a field window was altered; Grad-CAM explains it
  * ConvAutoencoder  - trained on genuine windows only; reconstruction error = anomaly map
  * UNet             - per-pixel tampering mask for localisation

Field crops are resized to 64 px high and cut into 64 x 256 windows (stride 128).
Inputs have 4 channels: BGR / 255 and a high-pass residual that exposes
stroke edges and paper-texture changes.
"""
import json
from functools import lru_cache

import cv2
import numpy as np

from .config import (TAMPER_AUTOENCODER_WEIGHTS, TAMPER_CALIBRATION, TAMPER_CLASSIFIER_WEIGHTS, TAMPER_PATCH,
                     TAMPER_UNET_WEIGHTS)

PH, PW = TAMPER_PATCH
STRIDE = PW // 2
UNET_BASE, UNET_DEPTH = 12, 3


def field_input(crop):
    """BGR crop -> float32 (4, 64, W) model input."""
    h, w = crop.shape[:2]
    nw = max(PW // 4, int(round(w * PH / max(h, 1))))
    x = cv2.resize(crop, (nw, PH), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.clip((gray - cv2.GaussianBlur(gray, (0, 0), 1.5)) / 32.0 + 0.5, 0, 1)
    return np.concatenate([x.transpose(2, 0, 1) / 255.0, hp[None]], 0).astype(np.float32)


def make_windows(x):
    """(4, 64, W) -> (starts, windows (N, 4, 64, 256)), right-padding by edge replication."""
    _, _, w = x.shape
    if w < PW:
        x = np.pad(x, ((0, 0), (0, 0), (0, PW - w)), mode="edge")
        w = PW
    starts = list(range(0, w - PW + 1, STRIDE))
    if starts[-1] != w - PW:
        starts.append(w - PW)
    return starts, np.stack([x[:, :, s:s + PW] for s in starts])


def stitch(maps, starts, width):
    """Average overlapping per-window maps (N, 64, 256) into one (64, width) map."""
    acc = np.zeros((PH, max(width, PW)), np.float32)
    cnt = np.zeros_like(acc)
    for s, m in zip(starts, maps):
        acc[:, s:s + PW] += m
        cnt[:, s:s + PW] += 1
    return (acc / np.maximum(cnt, 1))[:, :width]


@lru_cache(maxsize=1)
def _models():
    from .device import load_checkpoint
    from .nets import ConvAutoencoder, TamperClassifier, UNet
    out = {}
    for name, model, path in (("classifier", TamperClassifier(4), TAMPER_CLASSIFIER_WEIGHTS),
                              ("autoencoder", ConvAutoencoder(4), TAMPER_AUTOENCODER_WEIGHTS),
                              ("unet", UNet(4, 1, base=UNET_BASE, depth=UNET_DEPTH), TAMPER_UNET_WEIGHTS)):
        try:
            load_checkpoint(model, path)
            out[name] = model
        except FileNotFoundError:
            pass
    return out


@lru_cache(maxsize=1)
def calibration():
    try:
        with open(TAMPER_CALIBRATION, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError):
        return {}


def available():
    return sorted(_models())


def ae_error_map(x, recon):
    """Per-pixel reconstruction error emphasising colour and texture channels."""
    return np.abs(x - recon).mean(0)


def analyze_crop(crop, want_cam=True):
    """Run the available models on one field crop.

    Returns dict with (all maps resized to the crop size, values 0-1 except ae_map):
      classifier_prob, window_probs, cam (Grad-CAM), ae_map, ae_score, unet_map, unet_max
    Keys are missing for untrained models.
    """
    import torch
    from .device import get_device
    models = _models()
    if not models or crop is None or crop.size == 0:
        return {}
    dev = get_device()
    x = field_input(crop)
    width = x.shape[2]
    starts, wins = make_windows(x)
    t = torch.from_numpy(wins).to(dev)
    out = {}
    H, W = crop.shape[:2]
    up = lambda m: cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    if "classifier" in models:
        clf = models["classifier"]
        if want_cam:
            t_req = t.clone().requires_grad_(False)
            logits = clf(t_req, keep_activations=True)
            clf.zero_grad()
            logits.sum().backward()
            acts, grads = clf.activations.detach(), clf.gradients.detach()
            weights = grads.mean((2, 3), keepdim=True)
            cam = torch.relu((weights * acts).sum(1)).cpu().numpy()
            cams = np.stack([cv2.resize(c, (PW, PH)) for c in cam])
            cams /= max(cams.max(), 1e-6)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            cams *= probs[:, None, None]       # windows the model considers genuine contribute little
            out["cam"] = up(stitch(cams, starts, width))
        else:
            with torch.no_grad():
                probs = torch.sigmoid(clf(t)).cpu().numpy()
        out["window_probs"] = probs.tolist()
        out["classifier_prob"] = float(probs.max())
    with torch.no_grad():
        if "autoencoder" in models:
            recon = models["autoencoder"](t).cpu().numpy()
            errs = np.stack([ae_error_map(w, r) for w, r in zip(wins, recon)])
            emap = cv2.GaussianBlur(stitch(errs, starts, width), (0, 0), 2)
            out["ae_map"] = up(emap)
            out["ae_score"] = float(np.percentile(emap, 99.5))
        if "unet" in models:
            m = torch.sigmoid(models["unet"](t))[:, 0].cpu().numpy()
            umap = stitch(m, starts, width)
            out["unet_map"] = up(umap)
            out["unet_max"] = float(np.percentile(umap, 99.8))
    return out
