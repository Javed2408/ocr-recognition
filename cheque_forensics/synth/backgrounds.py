"""Cheque paper with security printing, and desk/photo backgrounds."""
import cv2
import numpy as np

# Pastel BGR paper tints used on CTS-2010 cheques
PAPER_TINTS = [(245, 235, 220), (232, 244, 238), (238, 236, 250), (225, 240, 250), (240, 244, 228),
               (236, 236, 236), (230, 236, 248), (248, 240, 232)]


def _smooth_noise(rng, h, w, scale, amp):
    small = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def security_paper(h, w, template_rng, rng):
    """Tinted paper with guilloche lines, micro-text style dashes and grain.

    template_rng fixes the design (same for every leaf of a template); rng adds
    per-leaf variation (grain, lighting).
    """
    tint = np.array(PAPER_TINTS[template_rng.integers(len(PAPER_TINTS))], np.float32)
    paper = np.ones((h, w, 3), np.float32) * tint
    pattern_color = tint * template_rng.uniform(0.80, 0.92)
    pattern = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # guilloche: interleaved sinusoidal line families
    for _ in range(template_rng.integers(2, 4)):
        period = template_rng.uniform(60, 220)
        amp = template_rng.uniform(4, 18)
        spacing = template_rng.uniform(7, 14)
        phase = template_rng.uniform(0, 2 * np.pi)
        v = (yy + amp * np.sin(2 * np.pi * xx / period + phase)) / spacing
        pattern = np.maximum(pattern, np.clip(1 - np.abs(v - np.round(v)) * template_rng.uniform(4, 7), 0, 1))
    # radial rosette behind the centre (common anti-copy motif)
    cx, cy = w * template_rng.uniform(0.3, 0.6), h * template_rng.uniform(0.4, 0.6)
    r = np.hypot(xx - cx, yy - cy)
    theta = np.arctan2(yy - cy, xx - cx)
    rosette = (np.sin(r / template_rng.uniform(3, 6) + 6 * np.sin(theta * template_rng.integers(5, 12))) > 0.92)
    pattern = np.maximum(pattern, rosette.astype(np.float32) * (r < h * 0.45) * 0.8)
    strength = template_rng.uniform(0.35, 0.7)
    paper = paper * (1 - pattern[..., None] * strength) + pattern_color * pattern[..., None] * strength
    paper += _smooth_noise(rng, h, w, 40, 3.0)[..., None]                 # uneven paper
    paper += rng.normal(0, 1.8, (h, w, 1)).astype(np.float32)              # grain
    return np.clip(paper, 0, 255).astype(np.uint8)


def desk_background(h, w, rng):
    """Plausible photo surroundings: table, cloth or dark mat, with lighting gradient."""
    kind = rng.integers(4)
    base = rng.uniform(20, 200, 3).astype(np.float32)
    img = np.ones((h, w, 3), np.float32) * base
    if kind == 0:   # wood grain
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grain = np.sin((yy + _smooth_noise(rng, h, w, 30, 25)) / rng.uniform(3, 9))
        img += grain[..., None] * rng.uniform(8, 25)
    elif kind == 1:   # fabric
        img += _smooth_noise(rng, h, w, 3, 12)[..., None]
    elif kind == 2:   # clutter rectangles
        for _ in range(rng.integers(3, 9)):
            x0, y0 = rng.integers(0, w), rng.integers(0, h)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + rng.integers(20, w // 2)), int(y0 + rng.integers(20, h // 2))),
                          tuple(float(c) for c in rng.uniform(0, 255, 3)), -1)
    img += _smooth_noise(rng, h, w, max(h, w) // 3, 30)[..., None]     # lighting gradient
    img += rng.normal(0, 3, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)
