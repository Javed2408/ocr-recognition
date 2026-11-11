"""Cheque boundary detection, cropping and perspective correction.

Three candidate generators are scored together:
  * edge contours (Canny + morphology + polygon approximation),
  * bright-paper segmentation (Otsu on lightness, largest component),
  * a U-Net boundary segmenter trained on synthetic photos (when its weights exist).
A candidate quadrilateral is scored on area, rectangularity and how close its
aspect ratio is to the CTS-2010 leaf. When nothing convincing is found the
image is assumed to be an already-cropped scan.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import ASPECT_RANGE, CANONICAL_WIDTH, CHEQUE_SEG_SIZE, CHEQUE_SEG_WEIGHTS, CTS_ASPECT, SEG_BASE_CHANNELS

WORK_SIZE = 800     # longest side used for boundary search


@dataclass
class ChequeDetection:
    quad: np.ndarray        # 4x2 float32 corners in the input image: TL, TR, BR, BL
    method: str             # "contour", "paper", "segmentation" or "full_image"
    confidence: float
    aspect: float

    def to_dict(self):
        return {"quad": self.quad.round(1).tolist(), "method": self.method,
                "confidence": round(float(self.confidence), 3), "aspect": round(float(self.aspect), 3)}


def order_corners(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def quad_size(quad):
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return w, h


def _aspect_score(aspect):
    if not ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]:
        return 0.0
    return float(np.exp(-abs(np.log(aspect / CTS_ASPECT)) * 2.0))


def edge_support(quad, edges, samples=200):
    """Fraction of points along the quad perimeter that lie on a (dilated) edge."""
    pts = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        t = np.linspace(0, 1, samples // 4, endpoint=False)[:, None]
        pts.append(a + (b - a) * t)
    pts = np.vstack(pts).round().astype(int)
    h, w = edges.shape
    inside = (pts[:, 0] >= 2) & (pts[:, 0] < w - 2) & (pts[:, 1] >= 2) & (pts[:, 1] < h - 2)
    if not inside.any():
        return 0.0
    on_edge = edges[pts[inside, 1], pts[inside, 0]] > 0
    return float(on_edge.sum() / len(pts))   # points on the image border count as unsupported


def edge_map(small):
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(10, 0.5 * med)), int(min(255, 1.2 * med)))
    return cv2.dilate(edges, np.ones((5, 5), np.uint8))


def score_quad(quad, image_shape, contour_area=None, edges=None):
    h_img, w_img = image_shape[:2]
    w, h = quad_size(quad)
    if min(w, h) < 20:
        return 0.0
    aspect = max(w, h) / min(w, h)
    area = cv2.contourArea(quad.astype(np.float32))
    area_frac = area / float(w_img * h_img)
    if area_frac < 0.12:
        return 0.0
    rect = 1.0 if contour_area is None else min(1.0, contour_area / max(area, 1))
    if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        return 0.0
    base = 0.45 * min(area_frac / 0.5, 1.0) + 0.35 * _aspect_score(aspect) + 0.2 * rect
    if edges is None:
        return float(base)
    return float(0.6 * base + 0.4 * edge_support(quad, edges))


def _quad_from_contour(cnt):
    peri = cv2.arcLength(cnt, True)
    for eps in (0.015, 0.02, 0.03, 0.045):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(cnt)))


def contour_candidates(small):
    edges = cv2.morphologyEx(edge_map(small), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = sorted((cv2.convexHull(c) for c in cnts), key=cv2.contourArea, reverse=True)[:6]
    return [("contour", _quad_from_contour(h), cv2.contourArea(h)) for h in hulls]


def paper_candidates(small):
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    light = cv2.GaussianBlur(lab[..., 0], (7, 7), 0)
    _, mask = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    return [("paper", _quad_from_contour(cv2.convexHull(cnt)), cv2.contourArea(cnt))]


_SEG_MODEL = None


def _seg_model():
    global _SEG_MODEL
    if _SEG_MODEL is None:
        from .device import load_checkpoint
        from .nets import UNet
        model = UNet(3, 1, base=SEG_BASE_CHANNELS, depth=4)
        try:
            load_checkpoint(model, CHEQUE_SEG_WEIGHTS)
            _SEG_MODEL = model
        except FileNotFoundError:
            _SEG_MODEL = False
    return _SEG_MODEL or None


def segment_cheque(image):
    """Probability map (same size as `image`) that a pixel belongs to the cheque, or None if untrained."""
    model = _seg_model()
    if model is None:
        return None
    import torch
    from .device import get_device
    h, w = CHEQUE_SEG_SIZE
    x = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    with torch.no_grad():
        t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(get_device())
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    return cv2.resize(prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def segmentation_candidates(small):
    prob = segment_cheque(small)
    if prob is None:
        return [], None
    mask = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [], prob
    cnt = max(cnts, key=cv2.contourArea)
    return [("segmentation", _quad_from_contour(cv2.convexHull(cnt)), cv2.contourArea(cnt))], prob


def detect_cheque(image, use_model=True):
    """Find the cheque in `image` (BGR). Returns a ChequeDetection in input-image coordinates."""
    h, w = image.shape[:2]
    scale = WORK_SIZE / max(h, w) if max(h, w) > WORK_SIZE else 1.0
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    cands = contour_candidates(small) + paper_candidates(small)
    prob = None
    if use_model:
        seg, prob = segmentation_candidates(small)
        cands += seg
    best, best_score = None, 0.0
    edges = edge_map(small)
    for method, quad, area in cands:
        s = score_quad(quad, small.shape, area, edges)
        if prob is not None and s > 0:   # agreement with the learned cheque mask
            poly = np.zeros(small.shape[:2], np.uint8)
            cv2.fillConvexPoly(poly, quad.astype(np.int32), 1)
            inter = float((prob[poly > 0] > 0.5).sum())
            union = float(poly.sum() + (prob > 0.5).sum() - inter)
            s = 0.7 * s + 0.3 * (inter / union if union else 0)
        if s > best_score:
            best, best_score = (method, quad), s
    full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    if best is None or best_score < 0.35:
        return ChequeDetection(full, "full_image", 0.5 if ASPECT_RANGE[0] <= max(w, h) / min(w, h) <= ASPECT_RANGE[1] else 0.3,
                               max(w, h) / min(w, h))
    method, quad = best
    quad = quad / scale
    # A quad that hugs the image border means the input is already a cropped scan.
    if cv2.contourArea(quad) > 0.93 * w * h:
        return ChequeDetection(full, "full_image", best_score, max(w, h) / min(w, h))
    qw, qh = quad_size(quad)
    return ChequeDetection(quad.astype(np.float32), method, best_score, max(qw, qh) / min(qw, qh))


def warp_to_canonical(image, quad, width=CANONICAL_WIDTH):
    """Perspective-correct the cheque to `width` pixels wide (height from its measured aspect).

    Portrait quads are rotated so the long side is horizontal. Returns (warped, H) where H maps
    input-image points to warped points.
    """
    quad = order_corners(quad)
    qw, qh = quad_size(quad)
    if qh > qw:   # cheque photographed in portrait: rotate corner order by 90 degrees
        quad = np.roll(quad, -1, axis=0)
        qw, qh = qh, qw
    aspect = float(np.clip(qw / max(qh, 1), ASPECT_RANGE[0], ASPECT_RANGE[1]))
    height = int(round(width / aspect))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped, H
