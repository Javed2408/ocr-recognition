"""General image forensics for the whole cheque.

Maps (canonical cheque size, 0 = normal, 1 = strongly unusual for this image):
  ela        Error Level Analysis, normalised by local edge energy so text edges do not dominate
  noise      inconsistency of the local noise residual on the paper
  jpeg_ghost blocks whose best-matching JPEG quality differs from the rest (double compression)
  jpeg_grid  blocks whose 8x8 compression grid is misaligned with the rest (pasted JPEG content)
  texture    Local Binary Pattern texture of the paper unlike the rest of the leaf
  edges      straight edges cutting through the background that are not printed lines
  frequency  unusual high-frequency energy (smoothing, resampling)
  difference pixel-level difference from a trusted original after registration (if provided)
Each map is relative to the image itself (robust z-scores), so it highlights
inconsistencies, not absolute values. None of them proves manipulation on its
own: compression, scanning, folds and stamps can all trigger them.
"""
import io
from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

from . import forensic_features as ff
from .schema import BBox, Indicator

BLOCK = 16
MAP_WEIGHTS = {"ela": 0.6, "noise": 0.8, "jpeg_ghost": 0.8, "jpeg_grid": 0.8, "texture": 0.6, "edges": 0.7,
               "frequency": 0.4, "difference": 1.0}
MAP_TEXT = {
    "ela": "recompression error (ELA) differs from comparable areas",
    "noise": "local sensor/paper noise level differs from the rest of the leaf",
    "jpeg_ghost": "the area matches a different JPEG quality than the rest (possible double compression)",
    "jpeg_grid": "the 8x8 JPEG grid here is misaligned with the rest of the image (possible pasted content)",
    "texture": "paper texture (Local Binary Patterns) differs from the rest of the leaf",
    "edges": "a straight edge cuts through the background where no printed line exists",
    "frequency": "unusual fine-detail (high-frequency) energy, as after smoothing or resampling",
    "difference": "pixels differ from the trusted original image after alignment",
}


@dataclass
class ForensicsResult:
    maps: dict = dc_field(default_factory=dict)
    fused: np.ndarray = None
    indicators: list = dc_field(default_factory=list)
    jpeg: dict = dc_field(default_factory=dict)
    registration: dict = None
    stats: dict = dc_field(default_factory=dict)

    def to_dict(self):
        return {"jpeg": self.jpeg, "registration": self.registration, "stats": self.stats,
                "maps": sorted(self.maps), "indicators": [i.to_dict() for i in self.indicators]}


def _jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def _blocks(arr, fn, block=BLOCK):
    h, w = arr.shape[:2]
    out = np.zeros((h // block, w // block), np.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = fn(arr[i * block:(i + 1) * block, j * block:(j + 1) * block])
    return out


def _to_unit(z, lo=2.5, hi=6.0):
    """Robust z-score -> 0..1 suspicion (0 below lo, 1 above hi)."""
    return np.clip((z - lo) / (hi - lo), 0, 1).astype(np.float32)


def _upsample(blockmap, shape):
    return cv2.resize(blockmap, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _z(blockmap, valid=None):
    v = blockmap[valid] if valid is not None else blockmap.ravel()
    if v.size < 10:
        return np.zeros_like(blockmap)
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med)) + 1e-6
    return (blockmap - med) / mad


# ---------------------------------------------------------------------------
# Individual analyses
# ---------------------------------------------------------------------------
def jpeg_info(file_bytes):
    """Is the upload a JPEG, and its approximate quality from the luminance quantisation table."""
    info = {"is_jpeg": False, "estimated_quality": None}
    if not file_bytes:
        return info
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(file_bytes))
        if im.format == "JPEG":
            info["is_jpeg"] = True
            q = getattr(im, "quantization", None)
            if q:
                lum = np.array(q[0], np.float32)
                # IJG standard luminance table mean = 57.6 at quality 50
                scale = lum.mean() / 57.625 * 100
                info["estimated_quality"] = int(np.clip(100 - scale / 2 if scale <= 100 else 5000 / scale, 1, 100))
    except Exception:   # unreadable metadata is not an error for the analysis
        pass
    return info


def ela_map(color, gray, quality=90):
    diff = np.abs(color.astype(np.int16) - _jpeg(color, quality).astype(np.int16)).mean(2).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    e = _blocks(diff, np.mean)
    g = _blocks(grad, np.mean)
    # expected ELA given edge energy: robust fit in edge-energy bins
    bins = np.quantile(g, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(g, bins[1:-1]), 0, 9)
    expected = np.zeros_like(e)
    for k in range(10):
        sel = idx == k
        if sel.any():
            expected[sel] = np.median(e[sel])
    return _z(e - expected), diff


def noise_map(gray, ink):
    resid = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    bg = cv2.dilate(ink, np.ones((5, 5), np.uint8)) == 0
    vals = _blocks(np.where(bg, np.abs(resid), np.nan), lambda b: np.nanmedian(b) if np.isfinite(b).sum() > 40 else np.nan)
    valid = np.isfinite(vals)
    vals = np.where(valid, vals, np.nanmedian(vals) if valid.any() else 0)
    z = _z(vals, valid)
    return np.abs(z), valid


def jpeg_ghost_map(color, qualities=range(55, 100, 5)):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    errs = []
    for q in qualities:
        d = (gray - cv2.cvtColor(_jpeg(color, q), cv2.COLOR_BGR2GRAY).astype(np.float32)) ** 2
        errs.append(_blocks(cv2.blur(d, (BLOCK, BLOCK)), np.mean))
    errs = np.stack(errs)
    errs /= errs.max(0, keepdims=True) + 1e-6
    best = np.argmin(errs, axis=0)
    global_best = np.bincount(best.ravel()).argmax()
    # how much better a block fits a different quality than the global one
    gain = errs[global_best] - errs.min(0)
    return _z(gain), list(qualities)[global_best]


def jpeg_grid_map(gray, region=4):
    """Per (region x 8)-pixel tile: offset of the 8x8 blocking grid; tiles disagreeing with the majority."""
    g = gray.astype(np.float32)
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    tile = 8 * region
    h, w = g.shape
    ny, nx = h // tile, w // tile
    offsets = np.zeros((ny, nx), np.int32)
    strength = np.zeros((ny, nx), np.float32)
    for i in range(ny):
        for j in range(nx):
            th = dh[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile - 1]
            tv = dv[i * tile:(i + 1) * tile - 1, j * tile:(j + 1) * tile]
            ch = np.array([th[:, k::8].mean() for k in range(8)]) if th.shape[1] >= 8 else np.zeros(8)
            cv_ = np.array([tv[k::8, :].mean() for k in range(8)]) if tv.shape[0] >= 8 else np.zeros(8)
            score = ch[:, None] + cv_[None, :]
            k = np.unravel_index(score.argmax(), score.shape)
            offsets[i, j] = k[0] * 8 + k[1]
            strength[i, j] = (score.max() - np.median(score)) / (np.median(score) + 1e-6)
    major = np.bincount(offsets.ravel()).argmax() if offsets.size else 0
    blocky = float(np.median(strength))
    mis = ((offsets != major) & (strength > max(0.15, blocky))).astype(np.float32)
    return mis, blocky, int(major)


def texture_map(gray, ink):
    codes = ff.lbp(gray)
    bg = (cv2.dilate(ink, np.ones((5, 5), np.uint8)) == 0).astype(np.uint8)
    h, w = gray.shape
    B = BLOCK * 2
    ny, nx = h // B, w // B
    hists, valid = np.zeros((ny, nx, 59)), np.zeros((ny, nx), bool)
    for i in range(ny):
        for j in range(nx):
            m = bg[i * B:(i + 1) * B, j * B:(j + 1) * B]
            if m.sum() > B * B * 0.5:
                hists[i, j] = ff.lbp_hist(codes[i * B:(i + 1) * B, j * B:(j + 1) * B], m)
                valid[i, j] = True
    if valid.sum() < 10:
        return np.zeros((ny, nx), np.float32), valid
    med = np.median(hists[valid], axis=0)
    d = np.array([[ff.chi2(hists[i, j], med) if valid[i, j] else 0 for j in range(nx)] for i in range(ny)], np.float32)
    return _z(d, valid) * valid, valid


def printed_rules(gray):
    """Mask of printed ruled lines and box borders, including dotted lines."""
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
    h_rules = cv2.morphologyEx(joined, cv2.MORPH_OPEN, np.ones((1, 60), np.uint8))
    joined_v = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8))
    v_rules = cv2.morphologyEx(joined_v, cv2.MORPH_OPEN, np.ones((25, 1), np.uint8))
    return cv2.dilate(h_rules | v_rules, np.ones((11, 11), np.uint8)) > 0


def edge_map(gray, ink, structure=None):
    """Straight background edges that are not printed rules or boxes."""
    mask = cv2.dilate(ink, np.ones((7, 7), np.uint8))
    bg = cv2.inpaint(gray, mask, 5, cv2.INPAINT_TELEA)
    hp = cv2.absdiff(bg, cv2.GaussianBlur(bg, (0, 0), 3))
    edges = cv2.Canny(hp, 12, 30)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=3)
    out = np.zeros(gray.shape, np.float32)
    printed = printed_rules(gray).astype(np.uint8)
    if structure is not None:
        for y, x1, x2 in structure.hlines:
            cv2.line(printed, (x1, y), (x2, y), 1, 9)
        for r in structure.rects:
            cv2.rectangle(printed, (r.x, r.y), (r.x2, r.y2), 1, 9)
    if lines is None:
        return out
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        if min(ang, 180 - ang) > 4 and abs(ang - 90) > 4:
            continue   # pasted patches have axis-aligned borders; guilloche curves do not
        seg = np.zeros(gray.shape, np.uint8)
        cv2.line(seg, (x1, y1), (x2, y2), 1, 3)
        if (seg & printed).sum() > 0.5 * seg.sum():
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        cv2.line(out, (x1, y1), (x2, y2), min(1.0, length / 250.0), 7)   # longer straight cuts are stronger evidence
    return out


def frequency_map(gray):
    def hf_ratio(b):
        f = np.abs(np.fft.fftshift(np.fft.fft2(b.astype(np.float32) - b.mean())))
        c = b.shape[0] // 2
        low = f[c - 4:c + 4, c - 4:c + 4].sum()
        return (f.sum() - low) / (f.sum() + 1e-6)
    return np.abs(_z(_blocks(gray, hf_ratio, BLOCK * 2)))


def forensic_thresholds():
    """Per-map region-strength thresholds calibrated on genuine validation cheques (if available)."""
    from .tamper_models import calibration
    return calibration().get("document_maps", {})


def duplicated_glyphs(gray, boxes, min_ncc=0.99):   # validation: 0/60 genuine above, 83% of copies
    """Pairs of characters that are pixel-identical up to scaling: handwriting never repeats exactly."""
    patches = []
    for b in boxes:
        p = b.pad(2, 2).crop(gray)
        if p.size and min(p.shape) >= 8:
            patches.append((b, cv2.resize(p, (24, 32)).astype(np.float32)))
    pairs = []
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            bi, pi = patches[i]
            bj, pj = patches[j]
            if bi.iou(bj) > 0.2 or abs(bi.h - bj.h) > 0.5 * max(bi.h, bj.h):
                continue
            a, b = pi - pi.mean(), pj - pj.mean()
            ncc = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
            if ncc >= min_ncc and a.std() > 10:
                pairs.append((bi, bj, ncc))
    return pairs


# ---------------------------------------------------------------------------
# Registration with a trusted original
# ---------------------------------------------------------------------------
def register(moving, fixed):
    """Align `moving` onto `fixed` (both BGR, similar size). Returns (warped, info)."""
    g1 = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    info = {"method": None, "inliers": 0, "matches": 0}
    H = None
    if d1 is not None and d2 is not None and len(k1) > 20 and len(k2) > 20:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        info["matches"] = len(good)
        if len(good) >= 12:
            src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            info["inliers"] = int(inl.sum()) if inl is not None else 0
            info["method"] = "orb_homography"
    if H is None or info["inliers"] < 12:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(g2.astype(np.float32), g1.astype(np.float32), warp, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
            H = np.vstack([cv2.invertAffineTransform(warp), [0, 0, 1]])
            info["method"] = "ecc_affine"
        except cv2.error:
            H = np.eye(3)
            info["method"] = "none"
    warped = cv2.warpPerspective(moving, H, (fixed.shape[1], fixed.shape[0]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, info


def difference_map(suspect, original):
    """Registered pixel comparison. Returns (0-1 map, binary change mask, info)."""
    if original.shape[:2] != suspect.shape[:2]:
        original = cv2.resize(original, (suspect.shape[1], suspect.shape[0]), interpolation=cv2.INTER_AREA)
    aligned, info = register(original, suspect)
    a = cv2.cvtColor(cv2.GaussianBlur(aligned, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(suspect, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):   # photometric normalisation (different scanner exposure)
        a[..., c] = (a[..., c] - a[..., c].mean()) / (a[..., c].std() + 1e-6) * b[..., c].std() + b[..., c].mean()
    d = np.sqrt(((a - b) ** 2).sum(2))
    # tolerate residual misalignment: compare against the local minimum over a 3x3 neighbourhood shift
    d = cv2.erode(d, np.ones((3, 3), np.uint8))
    med = float(np.median(d))
    mad = 1.4826 * float(np.median(np.abs(d - med)))
    thr = max(22.0, med + 10 * mad)   # scanner noise and residual misalignment stay well below this
    change = (d > thr).astype(np.uint8)   # no opening: the erosion above already absorbs 1-px misalignment
    m = 8                                  # warped borders of the original are extrapolated, not observed
    change[:m], change[-m:], change[:, :m], change[:, -m:] = 0, 0, 0, 0
    change = cv2.morphologyEx(change, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    info["threshold"] = thr
    info["median_residual"] = float(np.median(d))
    return np.clip(d / (2 * thr), 0, 1).astype(np.float32), change, info


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _regions(m, thr=0.5, min_area=60):
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m >= thr).astype(np.uint8))
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area >= min_area:
            out.append((BBox(int(x), int(y), int(w), int(h)), float(m[lab == k].max()), int(area)))
    return sorted(out, key=lambda r: -r[1] * r[2])


def field_of(box, fields):
    best, name = 0.0, None
    for n, f in fields.items():
        if f.bbox is not None:
            ov = box.overlap_fraction(f.bbox)
            if ov > best:
                best, name = ov, n
    return name if best > 0.3 else None


def analyze_document(color, file_bytes=None, original=None, fields=None, structure=None, char_boxes=()):
    """Run all forensic analyses on the normalised cheque image."""
    fields = fields or {}
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    ink = ff.ink_mask(color)
    shape = gray.shape
    res = ForensicsResult()
    res.jpeg = jpeg_info(file_bytes)
    is_jpeg = res.jpeg["is_jpeg"]

    z, _ = ela_map(color, gray)
    res.maps["ela"] = _upsample(_to_unit(z), shape)
    z, _ = noise_map(gray, ink)
    res.maps["noise"] = _upsample(_to_unit(z, 4.0, 8.0), shape)
    z, q = jpeg_ghost_map(color)
    res.maps["jpeg_ghost"] = _upsample(_to_unit(z), shape)
    res.stats["ghost_global_quality"] = q
    mis, blocky, offset = jpeg_grid_map(gray)
    res.stats.update({"blockiness": round(blocky, 4), "grid_offset": offset})
    # the 8x8 grid test only means something for JPEG sources: periodic security printing mimics blocking
    grid_reliable = is_jpeg and blocky > 0.15
    res.maps["jpeg_grid"] = _upsample(cv2.morphologyEx(mis, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) if grid_reliable
                                      else np.zeros_like(mis), shape)
    z, _ = texture_map(gray, ink)
    res.maps["texture"] = _upsample(_to_unit(z, 3.0, 7.0), shape)
    res.maps["edges"] = edge_map(gray, ink, structure)
    res.maps["frequency"] = _upsample(_to_unit(frequency_map(gray), 3.5, 7.0), shape)
    weights = dict(MAP_WEIGHTS)
    if not is_jpeg:     # ELA and JPEG cues are weak evidence for lossless sources
        for k in ("ela", "jpeg_ghost"):
            weights[k] *= 0.4
    if original is not None:
        dmap, change, info = difference_map(color, original)
        res.maps["difference"] = dmap
        res.registration = dict(info, changed_fraction=float(change.mean()))
        for box, peak, area in _regions(change.astype(np.float32), 0.5, 40)[:12]:
            f = field_of(box, fields)
            res.indicators.append(Indicator(
                "document_forensics", "difference_from_original", "high" if f in
                ("amount_numeric", "amount_words", "payee", "date", "signature") else "medium", min(1.0, area / 400),
                f"This region differs from the trusted original after alignment"
                f"{f' (in the {f.replace(chr(95), chr(32))} field)' if f else ''}. "
                f"Registration: {info['method']}, {info['inliers']} inlier matches.",
                f, box, {"area_px": area}))
    cal = forensic_thresholds()
    max_area = 0.25 * shape[0] * shape[1]      # image-wide differences are not local tampering evidence
    weighted = {k: m * weights[k] for k, m in res.maps.items()}
    res.fused = np.max(np.stack(list(weighted.values())), axis=0)
    res.stats["map_weights"] = weights
    res.stats["calibrated"] = bool(cal)
    # per-map strongest local region (statistics used for calibration and by the risk model)
    res.stats["map_max"], above = {}, {}
    for k, m in weighted.items():
        if k == "difference":
            continue
        local = [(b, p) for b, p, a in _regions(m, 0.3 * weights[k], 150) if a <= max_area]
        res.stats["map_max"][k] = round(max((p for _, p in local), default=0.0), 4)
        above[k] = m >= cal.get(k, 0.6)
    corroborated = 0
    for k, m in weighted.items():
        if k == "difference":
            continue
        thr = cal.get(k, 0.6)
        for box, peak, area in _regions(m, 0.75 * thr, 150)[:3]:
            if area > max_area:
                continue
            others = [o for o in above if o != k and box.crop(above[o]).mean() > 0.1]
            f = field_of(box, fields)
            standalone = k in ("edges",)     # ELA / JPEG / noise / texture / frequency cues need corroboration
            sev = "medium" if peak >= thr and (others or standalone) else "low"
            corroborated += int(sev == "medium" and bool(others))
            corr = (f" Corroborated by: {', '.join(others)}." if others else
                    " Not corroborated by any other analysis, so it is weak evidence.")
            res.indicators.append(Indicator(
                "document_forensics", f"{k}_anomaly", sev, float(peak),
                f"{MAP_TEXT[k][0].upper() + MAP_TEXT[k][1:]}"
                f"{f' (in the {f.replace(chr(95), chr(32))} field)' if f else ''}.{corr} Image-level cues like this "
                "also arise from scanning, compression, folds or stamps, so they are supporting evidence only.",
                f, box, {"peak": round(peak, 3), "area_px": area, "threshold": round(thr, 3),
                         "corroborating_maps": others}))
    res.stats["corroborated_regions"] = corroborated
    for bi, bj, ncc in duplicated_glyphs(gray, list(char_boxes))[:3]:
        res.indicators.append(Indicator(
            "document_forensics", "duplicated_glyph", "high", ncc,
            f"Two handwritten characters are pixel-identical (correlation {ncc:.3f}); handwriting never repeats "
            "exactly, so one was probably copied digitally.", field_of(bi, fields), bi,
            {"other_bbox": bj.as_list(), "ncc": ncc}))
    return res
