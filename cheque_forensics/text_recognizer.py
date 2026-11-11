"""Built-in CRNN-CTC text-line recogniser for cheque fields.

Used because Tesseract is optional (and weak on handwriting). Trained only on
synthetic renderings (training/train_text_recognizer.py); accuracy on real
handwriting is unmeasured, so its readings carry a confidence that the
validation engine takes into account.
"""
from functools import lru_cache

import cv2
import numpy as np

from .config import TEXT_HEIGHT, TEXT_RECOGNIZER_WEIGHTS, TEXT_WIDTH

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,./-=&:"
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARSET)}   # 0 is the CTC blank
NUMERIC_CHARS = set("0123456789,./-=")
DIGITS_ONLY = set("0123456789")


def encode(text):
    return [CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID]


def prepare_line(gray):
    """Gray crop -> float32 (TEXT_HEIGHT, TEXT_WIDTH) with ink high, background ~0."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    k = max(3, (gray.shape[0] // 2) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    ink = np.clip(bg - g, 0, None)
    hi = np.percentile(ink, 99.5)
    ink = np.clip(ink / max(hi, 8.0), 0, 1)
    h, w = ink.shape
    new_w = max(8, int(round(w * TEXT_HEIGHT / max(h, 1))))
    new_w = min(new_w, TEXT_WIDTH)
    line = cv2.resize(ink, (new_w, TEXT_HEIGHT), interpolation=cv2.INTER_AREA)
    out = np.zeros((TEXT_HEIGHT, TEXT_WIDTH), np.float32)
    out[:, :new_w] = line
    return out


def ctc_decode(logprobs, allowed=None):
    """Greedy CTC decoding. logprobs: (T, C). Returns (text, mean char confidence, per-char confidences)."""
    if allowed is not None:
        mask = np.full(logprobs.shape[1], -1e9, np.float32)
        mask[0] = 0
        for c in allowed:
            if c in CHAR_TO_ID:
                mask[CHAR_TO_ID[c]] = 0
        logprobs = logprobs + mask
        logprobs = logprobs - np.logaddexp.reduce(logprobs, axis=1, keepdims=True)
    probs = np.exp(logprobs)
    best = probs.argmax(1)
    text, confs, prev = [], [], 0
    for t, k in enumerate(best):
        if k != 0 and k != prev:
            text.append(CHARSET[k - 1])
            confs.append(float(probs[t, k]))
        prev = k
    return "".join(text).strip(), (float(np.mean(confs)) if confs else 0.0), confs


@lru_cache(maxsize=1)
def _model():
    from .device import load_checkpoint
    from .nets import CRNN
    model = CRNN(len(CHARSET) + 1)
    meta = load_checkpoint(model, TEXT_RECOGNIZER_WEIGHTS)
    return model, meta


def available():
    try:
        _model()
        return True
    except FileNotFoundError:   # missing or incompatible checkpoint (device.ModelUnavailable)
        return False


def recognize_lines(crops, allowed=None):
    """Read a list of gray/BGR line crops. Returns [(text, confidence)]."""
    import torch
    from .device import get_device
    model, _ = _model()
    batch = np.stack([prepare_line(c) for c in crops])[:, None]
    with torch.no_grad():
        logits = model(torch.from_numpy(batch).to(get_device()))
        logp = logits.log_softmax(2).cpu().numpy()
    return [ctc_decode(lp, allowed)[:2] for lp in logp]


def split_lines(gray, min_gap=3):
    """Split a field crop into text lines using the horizontal ink projection."""
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    # ignore long ruled lines
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(15, g.shape[1] // 4)), np.uint8))
    ink = ink & (1 - lines)
    rows = ink.sum(1) > max(2, g.shape[1] * 0.004)
    spans, start = [], None
    for y, on in enumerate(list(rows) + [False]):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= 6:
                spans.append([start, y])
            start = None
    merged = []
    for s in spans:   # merge spans separated by tiny gaps (descenders, i-dots)
        if merged and s[0] - merged[-1][1] < min_gap + max(3, (merged[-1][1] - merged[-1][0]) // 3):
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    tall = [s for s in merged if s[1] - s[0] >= max(8, g.shape[0] * 0.18)]
    if len(tall) <= 1:
        return [g]
    pad = 4
    return [g[max(0, a - pad):min(g.shape[0], b + pad)] for a, b in tall]
