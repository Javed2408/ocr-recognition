"""Render handwriting-like and printed text as alpha masks and composite them as ink.

Ink is composited multiplicatively, so a second pen stroke over an existing
one gets darker, as real overwriting does.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..schema import BBox
from .fonts import handwriting_families, load_font

# BGR ink colours for common Indian cheque pens (blue and black ballpoint/gel)
INKS = {
    "blue": [(150, 60, 20), (135, 45, 15), (170, 85, 40), (120, 40, 30), (160, 70, 55)],
    "black": [(40, 35, 35), (55, 50, 48), (30, 30, 40), (60, 50, 40)],
}


@dataclass
class Writer:
    family: str
    font: str
    size: int            # font size in px at canonical cheque resolution
    ink: tuple           # BGR
    pressure: float      # ink opacity 0-1
    thickness: int       # -1 thin, 0 normal, 1-2 thicker (dilation)
    slant: float         # horizontal shear
    spacing: float       # letter spacing multiplier
    jitter: float        # baseline jitter, fraction of size
    rotation: float      # per-character rotation jitter in degrees
    ink_name: str = "blue"

    def variant(self, rng, ink_shift=True):
        """Same person, different occasion or pen: small natural variation."""
        ink = self.ink
        if ink_shift:
            ink = tuple(int(np.clip(c + rng.integers(-18, 19), 0, 255)) for c in self.ink)
        return replace(self, ink=ink, size=int(self.size * rng.uniform(0.95, 1.05)),
                       pressure=float(np.clip(self.pressure + rng.uniform(-0.05, 0.05), 0.6, 1.0)))


def random_writer(rng, split, digits=False, ink_name=None, family=None):
    fams = handwriting_families(split, digits=digits)
    if family is None or family not in fams:
        family = sorted(fams)[rng.integers(len(fams))]
    font = fams[family][rng.integers(len(fams[family]))]
    ink_name = ink_name or ("blue" if rng.random() < 0.7 else "black")
    palette = INKS[ink_name]
    return Writer(
        family=family, font=font, size=int(rng.integers(30, 42)),
        ink=tuple(int(c) for c in palette[rng.integers(len(palette))]),
        pressure=float(rng.uniform(0.78, 0.97)), thickness=int(rng.choice([-1, 0, 0, 1, 1, 2])),
        slant=float(rng.uniform(-0.18, 0.28)), spacing=float(rng.uniform(0.95, 1.2)),
        jitter=float(rng.uniform(0.0, 0.06)), rotation=float(rng.uniform(0, 5)), ink_name=ink_name,
    )


def _glyph(ch, font, slant, rotation, scale, elastic_rng=None):
    """Alpha mask of one character, its baseline offset and advance width."""
    ascent, descent = font.getmetrics()
    pad = int(font.size * 0.6)
    w = int(font.getlength(ch)) + 2 * pad
    h = ascent + descent + 2 * pad
    img = Image.new("L", (max(w, 4), h), 0)
    ImageDraw.Draw(img).text((pad, pad + ascent), ch, font=font, fill=255, anchor="ls")
    a = np.asarray(img, dtype=np.uint8)
    if slant or rotation or scale != 1.0:
        cx, cy = w / 2, pad + ascent
        m = cv2.getRotationMatrix2D((cx, cy), rotation, scale)
        m[0, 1] += -slant           # shear x by y (positive slant leans right)
        m[0, 2] += slant * cy
        a = cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    if elastic_rng is not None:     # no two handwritten glyphs are identical
        a = _elastic(a, elastic_rng, font.size)
    return a, pad, pad + ascent, font.getlength(ch)


def _elastic(a, rng, size, amp=0.035):
    h, w = a.shape
    dx = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), size * 0.25) * size * amp * 6
    dy = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), size * 0.25) * size * amp * 6
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(a, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=0)


def render_text(text, writer, rng, printed=False):
    """Render `text` in the writer's hand.

    Returns (alpha float32 in [0,1], char_boxes [(char, BBox)] relative to alpha, baseline_y).
    """
    font = load_font(writer.font, writer.size)
    space = font.getlength(" ") * writer.spacing
    pieces, x = [], 0.0
    for ch in text:
        if ch == " ":
            x += space * (1.0 if printed else rng.uniform(0.8, 1.3))
            continue
        jitter = 0.0 if printed else rng.normal(0, writer.jitter * writer.size)
        rot = 0.0 if printed else rng.normal(0, writer.rotation / 2)
        scale = 1.0 if printed else rng.uniform(0.94, 1.06)
        a, ox, oy, adv = _glyph(ch, font, 0.0 if printed else writer.slant, rot, scale, None if printed else rng)
        pieces.append((ch, a, x - ox, -oy + jitter))
        x += adv * (writer.spacing if not printed else 1.0) + (0 if printed else rng.normal(0, writer.size * 0.02))
    if not pieces:
        return np.zeros((writer.size, 4), np.float32), [], writer.size
    min_x = min(px for _, _, px, _ in pieces)
    min_y = min(py for _, _, _, py in pieces)
    max_x = max(px + a.shape[1] for _, a, px, _ in pieces)
    max_y = max(py + a.shape[0] for _, a, _, py in pieces)
    W, H = int(np.ceil(max_x - min_x)) + 2, int(np.ceil(max_y - min_y)) + 2
    canvas = np.zeros((H, W), np.uint8)
    boxes = []
    for ch, a, px, py in pieces:
        ix, iy = int(round(px - min_x)), int(round(py - min_y))
        region = canvas[iy:iy + a.shape[0], ix:ix + a.shape[1]]
        np.maximum(region, a[:region.shape[0], :region.shape[1]], out=region)
        ys, xs = np.nonzero(a > 40)
        if len(xs):
            boxes.append((ch, BBox.from_xyxy(ix + xs.min(), iy + ys.min(), ix + xs.max() + 1, iy + ys.max() + 1)))
    if writer.thickness > 0:
        canvas = cv2.dilate(canvas, np.ones((writer.thickness + 1, writer.thickness + 1), np.uint8))
    elif writer.thickness < 0:
        canvas = cv2.erode(canvas, np.ones((2, 2), np.uint8))
    # crop to content
    ys, xs = np.nonzero(canvas > 20)
    if len(xs) == 0:
        return canvas.astype(np.float32) / 255.0, boxes, int(-min_y)
    x0, y0 = xs.min(), ys.min()
    canvas = canvas[y0:ys.max() + 1, x0:xs.max() + 1]
    boxes = [(ch, b.shift(-x0, -y0)) for ch, b in boxes]
    baseline = int(-min_y - y0)
    return canvas.astype(np.float32) / 255.0, boxes, baseline


def composite_ink(image, alpha, x, y, ink, pressure=0.9, rng=None, blur=0.6):
    """Multiply-blend ink of colour `ink` (BGR) into `image` (uint8 BGR, modified in place).

    Returns the BBox actually covered (clipped to the image).
    """
    H, W = image.shape[:2]
    h, w = alpha.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    if blur:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    if rng is not None:   # uneven ink flow along the stroke
        flow = cv2.GaussianBlur(rng.uniform(0.75, 1.0, a.shape).astype(np.float32), (0, 0), 1.5)
        a = a * flow
    a = np.clip(a * pressure, 0, 1)[..., None]
    region = image[y0:y1, x0:x1].astype(np.float32)
    inked = region * (np.array(ink, np.float32) / 255.0)
    image[y0:y1, x0:x1] = np.clip(region * (1 - a) + inked * a, 0, 255).astype(np.uint8)
    return BBox(x0, y0, x1 - x0, y1 - y0)


def write_text(image, text, writer, x, baseline_y, rng, printed=False, max_width=None):
    """Render and composite text with its baseline at `baseline_y`.

    Returns (covered BBox, [(char, BBox)] in image coordinates).
    """
    alpha, boxes, baseline = render_text(text, writer, rng, printed=printed)
    if max_width and alpha.shape[1] > max_width:   # squeeze long text into the space, like a real writer
        f = max_width / alpha.shape[1]
        alpha = cv2.resize(alpha, (max_width, alpha.shape[0]), interpolation=cv2.INTER_AREA)
        boxes = [(c, BBox.from_xyxy(b.x * f, b.y, b.x2 * f, b.y2)) for c, b in boxes]
    top = int(baseline_y - baseline)
    covered = composite_ink(image, alpha, int(x), top, writer.ink, 1.0 if printed else writer.pressure,
                            None if printed else rng, blur=0.4 if printed else 0.6)
    return covered, [(c, b.shift(int(x), top)) for c, b in boxes]


def ink_alpha_from_scan(gray):
    """Alpha mask of ink from a scanned grayscale signature (white background)."""
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    bg = np.percentile(g, 90)
    a = np.clip((bg - g) / max(bg - g.min(), 1.0), 0, 1)
    a[a < 0.12] = 0
    return a
