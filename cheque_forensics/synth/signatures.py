"""Signature samples for synthetic cheques: CEDAR when downloaded, a rendered fallback otherwise.

CEDAR (Kalera, Srihari & Xu, 2004): 55 writers x 24 genuine + 24 skilled forgeries,
http://www.cedar.buffalo.edu/NIJ/data/signatures.rar. Writers are split so that no
writer appears in more than one of train/val/test.
"""
import os
import re
from functools import lru_cache

import cv2
import numpy as np

from ..config import RAW_DATA_DIR

CEDAR_DIR = os.path.join(RAW_DATA_DIR, "cedar", "signatures")
WRITER_SPLITS = {"train": list(range(1, 36)), "val": list(range(36, 46)), "test": list(range(46, 56))}
SAMPLES_PER_WRITER = 24


def cedar_available():
    return os.path.isdir(os.path.join(CEDAR_DIR, "full_org")) and os.path.isdir(os.path.join(CEDAR_DIR, "full_forg"))


def cedar_path(writer, index, forged=False):
    if forged:
        return os.path.join(CEDAR_DIR, "full_forg", f"forgeries_{writer}_{index}.png")
    return os.path.join(CEDAR_DIR, "full_org", f"original_{writer}_{index}.png")


@lru_cache(maxsize=4096)
def load_cedar(writer, index, forged=False):
    img = cv2.imread(cedar_path(writer, index, forged), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(cedar_path(writer, index, forged))
    return img


def cedar_writers():
    """Writer ids that have both genuine and forged samples on disk."""
    if not cedar_available():
        return []
    ids = set()
    for name in os.listdir(os.path.join(CEDAR_DIR, "full_org")):
        m = re.match(r"original_(\d+)_\d+\.png", name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def synthetic_signature(name, rng, writer_font):
    """Fallback signature (rendered name plus flourish) when CEDAR is not downloaded."""
    from PIL import Image, ImageDraw
    from .fonts import load_font
    font = load_font(writer_font, 64)
    text = name.split()[0] + " " + name.split()[-1][0] + "."
    w = int(font.getlength(text)) + 60
    img = Image.new("L", (w, 130), 255)
    d = ImageDraw.Draw(img)
    d.text((20, 20), text, font=font, fill=0)
    pts = [(10 + i * (w - 20) / 20, 100 + 12 * np.sin(i / 3 + rng.uniform(0, 3))) for i in range(21)]
    d.line(pts, fill=0, width=3)
    return np.asarray(img)
