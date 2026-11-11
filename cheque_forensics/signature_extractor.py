"""Signature extraction from the cheque's signature zone and signature preprocessing.

Extraction separates pen ink from the tinted security background using
darkness and chroma relative to a local background estimate, removes the
printed signature line, drops speckles and crops to the ink. Preprocessing
(shared by training and inference) normalises the background, removes noise,
crops and resizes with the aspect ratio preserved.
"""
from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

from .config import SIGNATURE_SIZE
from .schema import BBox

MIN_SIGNATURE_INK_PX = 150


@dataclass
class SignatureExtraction:
    image: np.ndarray          # gray uint8 signature on white, cropped to the ink
    mask: np.ndarray           # uint8 ink mask of the zone crop
    bbox: BBox                 # ink bounding box inside the zone crop
    ink_pixels: int
    present: bool
    notes: list = dc_field(default_factory=list)


def ink_strength(color):
    """Per-pixel ink evidence (0-255): darker than, or more saturated than, the local background."""
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    k = min(31, max(15, (min(color.shape[:2]) // 4) | 1))   # strokes are < 10 px; a larger kernel only costs time
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, se)
    bg_a = cv2.GaussianBlur(cv2.morphologyEx(a, cv2.MORPH_OPEN, se), (0, 0), 3)
    bg_b = cv2.GaussianBlur(cv2.morphologyEx(b, cv2.MORPH_CLOSE, se), (0, 0), 3)
    dark = np.clip(bg_L - L, 0, None)
    chroma = np.hypot(a - bg_a, b - bg_b)
    return np.clip(np.maximum(dark * 1.4, chroma * 1.6), 0, 255).astype(np.uint8)


def extract_signature(zone_bgr):
    """Find the handwritten signature inside the signature-zone crop."""
    if zone_bgr is None or zone_bgr.size == 0:
        return SignatureExtraction(np.full((10, 10), 255, np.uint8), np.zeros((10, 10), np.uint8),
                                   BBox(0, 0, 1, 1), 0, False, ["empty signature zone"])
    strength = ink_strength(zone_bgr)
    thr, _ = cv2.threshold(strength, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (strength > max(thr, 38)).astype(np.uint8) * 255
    # printed signature line and box borders
    for k in ((max(25, zone_bgr.shape[1] // 3), 1), (1, max(25, int(zone_bgr.shape[0] * 0.8)))):
        mask &= ~cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        keep[i] = area >= 12 and max(w, h) >= 6
    mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    ink = int(np.count_nonzero(mask))
    if ink < MIN_SIGNATURE_INK_PX:
        return SignatureExtraction(np.full((10, 10), 255, np.uint8), mask, BBox(0, 0, 1, 1), ink, False,
                                   ["no signature ink found in the signature zone"])
    ys, xs = np.nonzero(mask)
    # drop isolated far-away blobs (stray printed text): keep the dense core
    x1, x2 = np.percentile(xs, [0.5, 99.5]).astype(int)
    y1, y2 = np.percentile(ys, [0.5, 99.5]).astype(int)
    pad = 4
    box = BBox.from_xyxy(max(0, x1 - pad), max(0, y1 - pad), min(mask.shape[1], x2 + pad + 1),
                         min(mask.shape[0], y2 + pad + 1))
    s = box.crop(strength).astype(np.float32) * (box.crop(mask) > 0)
    sig = (255 - np.clip(s / max(np.percentile(s[s > 0], 95), 1) * 255, 0, 255)).astype(np.uint8)
    notes = []
    if box.w < 30 or box.h < 12:
        notes.append("signature is very small; comparison is unreliable")
    return SignatureExtraction(sig, mask, box, ink, True, notes)


def preprocess_signature(gray, size=SIGNATURE_SIZE):
    """Gray signature (dark ink on light paper) -> float32 (H, W), ink = 1, background = 0.

    Steps: grayscale, background normalisation, median denoising, Otsu-based
    background suppression (ink intensity kept), crop to ink, aspect-preserving
    resize, centred padding.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    k = max(15, (min(g.shape) // 3) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ink = np.clip(bg - g, 0, 255).astype(np.uint8)
    ink = cv2.medianBlur(ink, 3)
    thr, _ = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = np.where(ink > max(thr * 0.6, 12), ink, 0).astype(np.float32)
    ys, xs = np.nonzero(ink)
    H, W = size
    if len(xs) == 0:
        return np.zeros((H, W), np.float32)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ink /= max(np.percentile(ink[ink > 0], 95), 1.0)
    scale = min((H - 4) / ink.shape[0], (W - 4) / ink.shape[1])
    nh, nw = max(1, int(ink.shape[0] * scale)), max(1, int(ink.shape[1] * scale))
    small = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W), np.float32)
    top, left = (H - nh) // 2, (W - nw) // 2
    out[top:top + nh, left:left + nw] = np.clip(small, 0, 1)
    return out
