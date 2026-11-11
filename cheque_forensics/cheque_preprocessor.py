"""Cheque image preprocessing.

Pipeline: input quality checks (aiproject.validate_input_image) -> boundary
detection and perspective correction -> 180-degree orientation check ->
deskew -> grayscale -> denoising -> illumination flattening -> CLAHE ->
adaptive threshold -> normalisation. Every intermediate image is kept so the
UI can show the steps, and the homography back to the uploaded image is kept
so field boxes can be drawn on the original.
"""
from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

import aiproject
from .cheque_detector import ChequeDetection, detect_cheque, warp_to_canonical
from .config import CANONICAL_WIDTH


@dataclass
class PreprocessedCheque:
    original: np.ndarray          # uploaded image (BGR)
    color: np.ndarray             # cropped, perspective-corrected, deskewed (BGR, canonical width)
    gray: np.ndarray
    denoised: np.ndarray
    flattened: np.ndarray         # illumination-normalised gray
    enhanced: np.ndarray          # CLAHE
    binary: np.ndarray            # adaptive threshold, ink = 255
    normalized: np.ndarray        # float32 0-1, zero-centred lighting
    homography: np.ndarray        # 3x3, original -> canonical
    detection: ChequeDetection
    rotation: float               # degrees of deskew applied
    flipped: bool                 # rotated by 180 degrees
    quality: list                 # aiproject.Check items
    quality_metrics: dict = dc_field(default_factory=dict)

    def to_original(self, points):
        """Map canonical-image points (N x 2) back to the uploaded image."""
        inv = np.linalg.inv(self.homography)
        pts = cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), inv)
        return pts.reshape(-1, 2)

    def box_to_original(self, bbox):
        pts = [[bbox.x, bbox.y], [bbox.x2, bbox.y], [bbox.x2, bbox.y2], [bbox.x, bbox.y2]]
        return self.to_original(pts).round(1).tolist()


def denoise(gray):
    """Edge-preserving denoising: median for salt-and-pepper, then non-local means."""
    g = cv2.medianBlur(gray, 3)
    return cv2.fastNlMeansDenoising(g, None, h=6, templateWindowSize=7, searchWindowSize=21)


def flatten_illumination(gray):
    """Divide by a large-scale background estimate to remove shadows and uneven lighting."""
    k = max(15, (gray.shape[1] // 40) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.GaussianBlur(background, (0, 0), k / 2)
    flat = cv2.divide(gray, background, scale=235)
    return flat


def clahe(gray, clip=2.0, tiles=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)


def adaptive_binary(gray, block=31, c=12, min_area=6):
    """Gaussian adaptive threshold, ink white on black, with speckle removal."""
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < min_area)[0]
    if len(small) > 1:
        binary[np.isin(labels, small[small > 0])] = 0
    return binary


def _rotate(image, angle, border=cv2.BORDER_REPLICATE):
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=border), np.vstack([m, [0, 0, 1]])


def deskew(color):
    """Straighten small residual rotation using long horizontal lines (aiproject.estimate_skew)."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    binary, _ = aiproject.binarize(gray)
    angle = aiproject.estimate_skew(binary)
    if angle is None or abs(angle) < 0.25 or abs(angle) > 20:
        return color, np.eye(3), 0.0
    rotated, m = _rotate(color, angle)
    return rotated, m, float(angle)


def fix_orientation(color, scorer=None):
    """Return (image, flipped, checked) choosing between 0 and 180 degrees.

    `scorer(image) -> float` should be higher for upright cheques (the field
    segmenter's layout confidence). Without a scorer the image is left as is
    and `checked` is False, because simple ink-profile heuristics proved
    unreliable on cheques with dark letterheads.
    """
    if scorer is None:
        return color, False, False
    flipped = cv2.rotate(color, cv2.ROTATE_180)
    if scorer(flipped) > scorer(color) + 0.02:
        return flipped, True, True
    return color, False, True


def quality_metrics(gray):
    return {
        "brightness": float(gray.mean()), "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "resolution": [int(gray.shape[1]), int(gray.shape[0])],
    }


def preprocess(image, use_models=True, orientation_scorer=None):
    """Run the full preprocessing pipeline on a BGR image."""
    if image is None or image.ndim not in (2, 3):
        raise ValueError("expected a BGR or grayscale image array")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = image[..., :3]
    gray0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary0, inverted = aiproject.binarize(gray0)
    quality = aiproject.validate_input_image(image, gray0, binary0, inverted)
    metrics = quality_metrics(gray0)

    detection = detect_cheque(image, use_model=use_models)
    color, H = warp_to_canonical(image, detection.quad, CANONICAL_WIDTH)
    color, flipped, orientation_checked = fix_orientation(color, orientation_scorer)
    if flipped:
        h, w = color.shape[:2]
        H = np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], np.float64) @ H
    color, rot, angle = deskew(color)
    H = rot @ H

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    den = denoise(gray)
    flat = flatten_illumination(den)
    enh = clahe(flat)
    binary = adaptive_binary(enh)
    norm = flat.astype(np.float32) / 255.0
    return PreprocessedCheque(
        original=image, color=color, gray=gray, denoised=den, flattened=flat, enhanced=enh, binary=binary,
        normalized=norm, homography=H, detection=detection, rotation=angle, flipped=flipped,
        quality=quality, quality_metrics=dict(metrics, orientation_checked=orientation_checked),
    )
