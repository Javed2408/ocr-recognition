"""Classical image-forensic features for handwritten cheque fields.

All comparisons are relative to the other characters/words of the same
field (robust z-scores), so a writer's own style, pen colour or thickness is
never flagged on its own; only a part that differs from the rest of the same
field is.
"""
from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

from .schema import BBox
from .signature_extractor import ink_strength


# ---------------------------------------------------------------------------
# Ink and characters
# ---------------------------------------------------------------------------
def remove_rules(mask, crop_shape):
    """Remove printed ruled lines and box borders from an ink mask."""
    h, w = crop_shape[:2]
    out = mask.copy()
    for k in ((max(20, w // 4), 1), (1, max(15, int(h * 0.75)))):
        out &= ~cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    return out


def ink_mask(crop):
    """uint8 mask (255 = handwriting ink) of a field crop."""
    s = ink_strength(crop)
    thr, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (s > max(thr, 30)).astype(np.uint8) * 255
    mask = remove_rules(mask, crop.shape)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < 8)[0]
    if len(small) > 1:
        mask[np.isin(lab, small[small > 0])] = 0
    return mask


def segment_characters(mask, min_h_frac=0.25):
    """Character boxes (BBox) in reading order from an ink mask.

    Components that overlap horizontally by more than half are merged (a
    digit's separate strokes; an overwritten digit and the original).
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [BBox(int(x), int(y), int(w), int(h)) for x, y, w, h, a in stats[1:] if a >= 10]
    if not boxes:
        return []
    boxes.sort(key=lambda b: b.x)
    merged = [boxes[0]]
    for b in boxes[1:]:
        m = merged[-1]
        ov = min(m.x2, b.x2) - max(m.x, b.x)
        if ov > 0.5 * min(m.w, b.w):
            merged[-1] = BBox.from_xyxy(min(m.x, b.x), min(m.y, b.y), max(m.x2, b.x2), max(m.y2, b.y2))
        else:
            merged.append(b)
    med_h = np.median([b.h for b in merged])
    return [b for b in merged if b.h >= med_h * min_h_frac]


def group_words(chars, gap_factor=0.6):
    """Group character boxes of one line into words using inter-character gaps."""
    if not chars:
        return []
    med_h = float(np.median([c.h for c in chars]))
    words, cur = [], [chars[0]]
    for a, b in zip(chars, chars[1:]):
        if b.x - a.x2 > gap_factor * med_h:
            words.append(cur)
            cur = [b]
        else:
            cur.append(b)
    words.append(cur)
    return words


def split_line_boxes(mask):
    """Row bands of a (possibly two-line) field: [(y1, y2)]."""
    rows = (mask > 0).sum(1)
    on = rows > max(2, 0.01 * mask.shape[1])
    bands, start = [], None
    for y, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = y
        elif not v and start is not None:
            if y - start >= 6:
                bands.append([start, y])
            start = None
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < max(4, (merged[-1][1] - merged[-1][0]) // 3):
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    tall = [b for b in merged if b[1] - b[0] >= max(10, mask.shape[0] * 0.2)]
    return tall or [[0, mask.shape[0]]]


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------
def lbp(gray):
    """8-neighbour, radius-1 Local Binary Pattern codes (uint8)."""
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    for bit, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
        n = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((n >= c).astype(np.uint8) << bit)
    return np.pad(code, 1, mode="edge")


UNIFORM = np.array([bin(i ^ ((i << 1 | i >> 7) & 0xFF)).count("1") <= 2 for i in range(256)])
UNIFORM_INDEX = np.where(UNIFORM, np.cumsum(UNIFORM) - 1, UNIFORM.sum())   # 58 uniform patterns + 1 bin


def lbp_hist(codes, mask=None):
    v = UNIFORM_INDEX[codes[mask > 0] if mask is not None else codes.ravel()]
    h = np.bincount(v, minlength=59).astype(np.float64)
    return h / max(h.sum(), 1)


def chi2(h1, h2):
    return float(0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-9)))


# ---------------------------------------------------------------------------
# Per-region measurements
# ---------------------------------------------------------------------------
def stroke_widths(mask):
    """Stroke widths sampled on the medial ridge of the ink (2 x distance transform - 1)."""
    m = np.pad((mask > 0).astype(np.uint8), 1)   # zero border: a crop that is all ink would give infinite distances
    if m.sum() == 0:
        return np.array([])
    dt = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6) & (m > 0)
    return 2 * dt[ridge] - 1          # distance is to the nearest background pixel centre


@dataclass
class RegionFeatures:
    box: BBox
    stroke_width: float
    stroke_iqr: float
    darkness: float            # background L - ink L
    ink_lab: tuple             # median (L, a, b) of the ink
    overlap: float             # share of ink much darker than the region's own median (double inking)
    edge_density: float        # Canny edge pixels per ink pixel (double contours raise it)
    holes: int
    bg_brightness: float       # paper around the region
    bg_noise: float
    bg_lbp: np.ndarray = dc_field(repr=False, default=None)
    slant: float = 0.0


def region_features(crop, gray, lab, mask, codes, box):
    sub = box.crop(mask) > 0
    if sub.sum() < 5:
        return None
    sw = stroke_widths(box.crop(mask))
    L = box.crop(lab[..., 0]).astype(np.float32)
    a, b = box.crop(lab[..., 1]).astype(np.float32), box.crop(lab[..., 2]).astype(np.float32)
    ring_box = box.pad(max(3, box.h // 5), max(3, box.h // 5)).clip(mask.shape[1], mask.shape[0])
    ring = cv2.dilate(ring_box.crop(mask), np.ones((5, 5), np.uint8)) == 0
    ringL = ring_box.crop(lab[..., 0]).astype(np.float32)[ring]
    bgL = float(np.median(ringL)) if ringL.size else float(np.percentile(L, 90))
    inkL = L[sub]
    edges = cv2.Canny(box.crop(gray), 50, 140) > 0
    cnts, hier = cv2.findContours(box.crop(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = int(sum(1 for h in (hier[0] if hier is not None else []) if h[3] >= 0))
    ys, xs = np.nonzero(sub)
    slant = 0.0
    if len(xs) > 10:
        cov = np.cov(xs - xs.mean(), ys - ys.mean())
        slant = float(cov[0, 1] / max(cov[1, 1], 1e-6))
    return RegionFeatures(
        box=box, stroke_width=float(np.median(sw)) if sw.size else 0.0,
        stroke_iqr=float(np.subtract(*np.percentile(sw, [75, 25]))) if sw.size else 0.0,
        darkness=bgL - float(np.median(inkL)),
        ink_lab=(float(np.median(inkL)), float(np.median(a[sub])), float(np.median(b[sub]))),
        overlap=float(np.mean(inkL < np.median(inkL) - 22)),
        edge_density=float(edges.sum() / max(sub.sum(), 1)), holes=holes,
        bg_brightness=bgL, bg_noise=float(np.std(ringL)) if ringL.size else 0.0,
        bg_lbp=lbp_hist(ring_box.crop(codes), ring.astype(np.uint8)), slant=slant,
    )


def robust_z(values):
    v = np.asarray(values, np.float64)
    if len(v) < 3:
        return np.zeros_like(v)
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return (v - med) / max(mad, 1e-6 + 0.05 * abs(med))


def ink_delta_e(feats):
    """Colour difference (Lab, ignoring lightness) of each region's ink from the field median."""
    ab = np.array([[f.ink_lab[1], f.ink_lab[2]] for f in feats])
    return np.linalg.norm(ab - np.median(ab, axis=0), axis=1)


def anomaly_table(feats):
    """Per-region anomaly components (non-negative, ~1 at the flagging level) and a combined score.

    Components:
      stroke   |z| of stroke width
      ink      Lab chroma difference from the field's median ink, in units of 10 dE
      dark     z of darkness (darker ink, e.g. a second pen)
      overlap  z of double-inked share (overwritten strokes)
      edges    z of edge density (double contours)
      paper    z of background brightness + LBP texture distance (erasure)
    """
    if len(feats) < 2:
        return [], np.zeros(len(feats))
    sw = robust_z([f.stroke_width for f in feats])
    dk = robust_z([f.darkness for f in feats])
    ov = robust_z([f.overlap for f in feats])
    ed = robust_z([f.edge_density for f in feats])
    br = robust_z([f.bg_brightness for f in feats])
    hist_med = np.median(np.stack([f.bg_lbp for f in feats]), axis=0)
    tex = robust_z([chi2(f.bg_lbp, hist_med) for f in feats])
    de = ink_delta_e(feats)
    rows = []
    for i in range(len(feats)):
        comp = {
            "stroke": abs(sw[i]) / 3.5, "ink": de[i] / 10.0, "dark": max(dk[i], 0) / 3.5,
            "overlap": max(ov[i], 0) / 3.5, "edges": max(ed[i], 0) / 3.5,
            "paper": max(max(br[i], 0) / 3.5, max(tex[i], 0) / 4.0),
        }
        rows.append(comp)
    combined = np.array([max(r.values()) for r in rows]) if rows else np.zeros(0)
    return rows, combined


COMPONENT_TEXT = {
    "stroke": "stroke width differs from the rest of the field",
    "ink": "ink colour differs from the rest of the field",
    "dark": "ink is noticeably darker than the rest of the field",
    "overlap": "double-inked (overlapping) strokes, typical of overwriting",
    "edges": "doubled stroke edges, typical of overwriting",
    "paper": "paper texture/brightness around it is disturbed (possible erasure)",
}
