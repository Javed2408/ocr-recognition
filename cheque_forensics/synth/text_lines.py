"""Synthetic text-line crops (cheque field style) for training the CRNN recogniser."""
import cv2
import numpy as np

from .backgrounds import security_paper
from .cheque_generator import amount_as_written, amount_in_words, random_amount, random_payee
from .fonts import MICR_FONT, printed_fonts
from .render import Writer, random_writer, write_text

KINDS = ["words", "words", "words", "payee", "payee", "amount", "amount", "date", "ifsc", "account", "micr", "random"]
ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def random_text(rng, kind):
    if kind == "words":
        words = amount_in_words(random_amount(rng), rng)
        if rng.random() < 0.4:   # a wrapped line holds only part of the phrase
            parts = words.split()
            cut = int(rng.integers(1, len(parts) + 1))
            words = " ".join(parts[:cut] if rng.random() < 0.5 else parts[-cut:])
        return words
    if kind == "payee":
        return random_payee(rng)
    if kind == "amount":
        return amount_as_written(random_amount(rng), rng)
    if kind == "date":
        d = f"{rng.integers(1, 29):02d}{rng.integers(1, 13):02d}{rng.integers(2024, 2028)}"
        return d if rng.random() < 0.7 else f"{d[:2]}/{d[2:4]}/{d[4:]}"
    if kind == "ifsc":
        letters, alnum = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        code = "".join(rng.choice(letters, 4)) + "0" + "".join(rng.choice(alnum[:10] if rng.random() < 0.7 else alnum, 6))
        return f"IFSC: {code}" if rng.random() < 0.6 else code
    if kind == "account":
        return "".join(str(d) for d in rng.integers(0, 10, int(rng.integers(9, 17))))
    if kind == "micr":
        g = lambda n: "".join(str(d) for d in rng.integers(0, 10, n))
        return " ".join([g(6), g(9), g(6), g(2)][: int(rng.integers(1, 5))])
    n = int(rng.integers(3, 14))
    return "".join(rng.choice(list(ALNUM + "  ,./-&"), n)).strip() or "A1"


def render_line(rng, split="train"):
    """Returns (gray uint8 crop, text)."""
    kind = KINDS[rng.integers(len(KINDS))]
    text = random_text(rng, kind)
    handwritten = kind in ("words", "payee", "amount", "date") or (kind == "random" and rng.random() < 0.5)
    if handwritten:
        writer = random_writer(rng, split, digits=kind in ("amount", "date", "random"))
    elif kind == "micr":
        writer = Writer("micr", MICR_FONT, int(rng.integers(24, 32)), (25, 25, 25), 1.0, 0, 0, 1, 0, 0)
    else:
        fonts = printed_fonts(split)
        writer = Writer("print", fonts[rng.integers(len(fonts))], int(rng.integers(16, 30)),
                        tuple(int(c) for c in rng.integers(10, 120, 3)), 1.0, 0, 0, 1, 0, 0)
    h = int(writer.size * 2.2)
    w = int(writer.size * (len(text) + 4) * 0.8) + 40
    canvas = security_paper(h, w, np.random.default_rng(rng.integers(1 << 30)), rng)
    base = int(h * rng.uniform(0.62, 0.72))
    if kind == "date" and rng.random() < 0.7:   # one digit per printed box
        text = text.replace("/", "")
        x, bw = 10, int(writer.size * 0.9)
        for ch in text:
            cv2.rectangle(canvas, (x, base - writer.size), (x + bw, base + 6), (90, 60, 40), 1)
            write_text(canvas, ch, writer, x + bw * 0.15, base, rng)
            x += bw + int(rng.integers(0, 4))
        x0, x1 = 6, min(w, x + 2)
    else:
        if handwritten and rng.random() < 0.7:   # ruled line under the writing
            y = base + int(rng.integers(2, 8))
            cv2.line(canvas, (0, y), (w, y), (90, 60, 40), 1)
        box, _ = write_text(canvas, text, writer, 12 + int(rng.integers(0, 10)), base, rng, printed=not handwritten)
        if box is None:
            return render_line(rng, split)
        x0, x1 = box.x, box.x2
    m = rng.integers(2, 14, 2)
    crop = canvas[int(rng.uniform(0.0, 0.25) * h):int(h - rng.uniform(0.0, 0.15) * h),
                  max(0, x0 - int(m[0])):min(w, x1 + int(m[1]))]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if rng.random() < 0.4:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 1.0))
    g += rng.normal(0, rng.uniform(0, 5), g.shape)
    g = np.clip(g, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 95))])
        g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return g, text
