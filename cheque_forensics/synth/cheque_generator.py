"""Synthetic Indian cheque leaves with ground-truth field boxes.

Each template id deterministically defines a fictional bank's layout (zone
positions, colours, security pattern, label language) within the CTS-2010
conventions; templates are split train/val/test so evaluation always uses
layouts never seen in training.
"""
import datetime as dt
from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

from ..amount_parser import format_indian_number, number_to_indian_words
from ..config import CANONICAL_HEIGHT, CANONICAL_WIDTH
from ..schema import BBox
from .backgrounds import security_paper
from .fonts import MICR_FONT, RUPEE_FONT, printed_fonts
from .render import Writer, ink_alpha_from_scan, random_writer, write_text, composite_ink
from . import signatures as sigs

W, H = CANONICAL_WIDTH, CANONICAL_HEIGHT
TEMPLATE_SPLITS = {"train": list(range(0, 28)), "val": list(range(28, 34)), "test": list(range(34, 40))}
REFERENCE_DATE = dt.date(2026, 11, 1)

BANK_WORDS_A = ["Sahyadri", "Narmada", "Konark", "Vindhya", "Saraswat", "Godavari", "Kaveri", "Ajanta",
                "Malabar", "Deccan", "Himalaya", "Ganga", "Sindhu", "Aravali", "Nilgiri", "Charminar"]
BANK_WORDS_B = ["Co-operative Bank Ltd.", "Bank", "Urban Bank", "Gramin Bank", "Commercial Bank", "Sahakari Bank Ltd."]
CITIES = ["Pune", "Nagpur", "Indore", "Kochi", "Jaipur", "Lucknow", "Surat", "Mysuru", "Nashik", "Bhopal",
          "Coimbatore", "Vadodara", "Patna", "Ranchi", "Guwahati", "Mangaluru"]
FIRST = ["Ramesh", "Priya", "Anil", "Sunita", "Vikram", "Lakshmi", "Arjun", "Meena", "Rahul", "Kavita",
         "Suresh", "Deepa", "Mohan", "Asha", "Karthik", "Fatima", "Imran", "Gurpreet", "Neha", "Rajesh",
         "Sanjay", "Pooja", "Venkat", "Anjali", "Harish", "Nisha", "Abdul", "Joseph", "Shweta", "Prakash"]
LAST = ["Kumar", "Sharma", "Patel", "Iyer", "Reddy", "Singh", "Nair", "Gupta", "Das", "Joshi", "Rao",
        "Khan", "Verma", "Menon", "Pillai", "Banerjee", "Kulkarni", "Shetty", "Agarwal", "Chopra"]
BUSINESS = ["Traders", "Enterprises", "Agencies", "& Sons", "Stores", "Textiles", "Electricals", "Pharma",
            "Associates", "Builders", "Motors", "Hardware"]
TXN_CODES = ["10", "11", "29", "31"]


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
@dataclass
class Template:
    id: int
    bank_name: str
    branch: str
    bank_code: str
    print_color: tuple
    accent_color: tuple
    print_font: str
    hindi_labels: bool
    dotted_lines: bool
    two_line_words: bool
    amount_box_border: bool
    zones: dict                    # name -> BBox of the writing zone
    anchors: dict                  # internal layout positions
    seed: int


def template_split(template_id):
    for split, ids in TEMPLATE_SPLITS.items():
        if template_id in ids:
            return split
    return "train"


def make_template(template_id):
    rng = np.random.default_rng(10_007 + template_id)
    split = template_split(template_id)
    fonts = printed_fonts(split)
    a = {}
    a["date_x"] = int(W * rng.uniform(0.70, 0.75))
    a["date_y"] = int(H * rng.uniform(0.06, 0.12))
    a["date_box"] = int(W * rng.uniform(0.024, 0.029))
    a["date_h"] = int(H * rng.uniform(0.07, 0.085))
    a["date_gap"] = int(rng.integers(0, 4))
    a["pay_y"] = int(H * rng.uniform(0.31, 0.35))
    a["pay_label_w"] = int(W * rng.uniform(0.05, 0.08))
    a["bearer_x"] = int(W * rng.uniform(0.80, 0.87))
    a["words_y1"] = a["pay_y"] + int(H * rng.uniform(0.11, 0.13))
    two_lines = bool(rng.random() < 0.6)
    a["words_y2"] = a["words_y1"] + int(H * rng.uniform(0.09, 0.10)) if two_lines else None
    a["box_x"] = int(W * rng.uniform(0.72, 0.77))
    a["box_w"] = int(W * 0.965) - a["box_x"]
    a["box_h"] = int(H * rng.uniform(0.10, 0.13))
    if two_lines:   # box sits between the two rupees lines, to their right
        a["box_y"] = int((a["words_y1"] + a["words_y2"]) / 2 - a["box_h"] / 2 - H * 0.03)
    else:
        a["box_y"] = a["words_y1"] - int(a["box_h"] * 0.75)
    a["words_end"] = a["box_x"] - int(W * 0.02)
    a["words_line1_end"] = a["words_end"]
    a["acct_x"] = int(W * rng.uniform(0.03, 0.07))
    a["acct_y"] = int(H * rng.uniform(0.66, 0.72))
    a["sig_x"] = int(W * rng.uniform(0.60, 0.68))
    a["sig_y"] = int(H * rng.uniform(0.60, 0.65))
    a["sig_bottom"] = int(H * rng.uniform(0.80, 0.83))
    a["micr_x"] = int(W * rng.uniform(0.17, 0.24))
    a["micr_base"] = int(H * rng.uniform(0.925, 0.945))
    a["micr_size"] = int(rng.integers(26, 31))
    a["header_size"] = int(rng.integers(26, 34))
    bank = f"{BANK_WORDS_A[rng.integers(len(BANK_WORDS_A))]} {BANK_WORDS_B[rng.integers(len(BANK_WORDS_B))]}"
    letters = "ABCDEFGHJKLMNPRSTUVWY"
    code = "Q" + "".join(letters[i] for i in rng.integers(0, len(letters), 3))   # Q-prefixed: not a real bank code
    colors = [(110, 40, 10), (40, 40, 40), (30, 30, 120), (90, 60, 20), (60, 90, 30)]
    t = Template(
        id=template_id, bank_name=bank, branch=f"{CITIES[rng.integers(len(CITIES))]} Main Branch",
        bank_code=code, print_color=colors[rng.integers(len(colors))], accent_color=colors[rng.integers(len(colors))],
        print_font=fonts[rng.integers(len(fonts))], hindi_labels=bool(rng.random() < 0.5),
        dotted_lines=bool(rng.random() < 0.3), two_line_words=two_lines,
        amount_box_border=bool(rng.random() < 0.8), zones={}, anchors=a, seed=10_007 + template_id,
    )
    return t


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
@dataclass
class ChequeContent:
    payee: str
    date: dt.date
    amount: int
    amount_text: str
    words_text: str
    ifsc: str
    account_number: str
    cheque_number: str
    micr_code: str
    account_id: str
    txn_code: str


def random_amount(rng):
    r = rng.random()
    if r < 0.4:
        return int(rng.integers(1, 100)) * 1000
    if r < 0.6:
        return int(rng.integers(20, 500)) * 5000
    if r < 0.85:
        return int(rng.integers(100, 100_000))
    return int(rng.integers(100_000, 1_000_000))


def amount_as_written(amount, rng, allow_open=True):
    """How a person writes the figure, e.g. '1,25,000/-', '125000/-', '5,000=00', '5000'."""
    grouped = format_indian_number(amount) if rng.random() < 0.65 else str(amount)
    r = rng.random()
    if r < 0.6:
        return grouped + "/-"
    if r < 0.75:
        return grouped + "=00"
    if r < 0.85:
        return grouped + ".00"
    return grouped if allow_open else grouped + "/-"


def amount_in_words(amount, rng):
    words = number_to_indian_words(amount)
    style = rng.random()
    words = words.title() if style < 0.7 else (words.upper() if style < 0.85 else words)
    if rng.random() < 0.85:
        words += " Only" if not words.isupper() else " ONLY"
    return words


def random_payee(rng):
    if rng.random() < 0.3:
        return f"{LAST[rng.integers(len(LAST))]} {BUSINESS[rng.integers(len(BUSINESS))]}"
    return f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}"


def random_content(rng, template, reference_date=REFERENCE_DATE):
    amount = random_amount(rng)
    date = reference_date - dt.timedelta(days=int(rng.integers(0, 80)))
    return ChequeContent(
        payee=random_payee(rng), date=date, amount=amount, amount_text=amount_as_written(amount, rng),
        words_text=amount_in_words(amount, rng),
        ifsc=template.bank_code + "0" + "".join(str(d) for d in rng.integers(0, 10, 6)),
        account_number="".join(str(d) for d in rng.integers(0, 10, int(rng.integers(11, 16)))),
        cheque_number="".join(str(d) for d in rng.integers(0, 10, 6)),
        micr_code="".join(str(d) for d in rng.integers(0, 10, 9)),
        account_id="".join(str(d) for d in rng.integers(0, 10, 6)),
        txn_code=TXN_CODES[rng.integers(len(TXN_CODES))],
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
@dataclass
class SyntheticCheque:
    image: np.ndarray              # BGR uint8, canonical size
    paper: np.ndarray              # printed leaf without handwriting (for erasure simulation)
    fields: dict                   # field name -> BBox (writing zone / printed region)
    content: ChequeContent
    template: Template
    writer: Writer
    payee_writer: Writer
    chars: dict                    # field -> [(char, BBox)] of handwritten characters
    ink_boxes: dict                # field -> BBox of the handwriting actually written
    signature: dict = dc_field(default_factory=dict)   # writer id, sample index, forged flag
    split: str = "train"


def _printed_writer(font, size, color):
    return Writer(family="print", font=font, size=size, ink=color, pressure=1.0, thickness=0,
                  slant=0.0, spacing=1.0, jitter=0.0, rotation=0.0)


def _hline(img, x1, x2, y, color, dotted, thickness=1):
    if dotted:
        for x in range(int(x1), int(x2), 6):
            cv2.line(img, (x, y), (min(x + 2, int(x2)), y), color, thickness)
    else:
        cv2.line(img, (int(x1), y), (int(x2), y), color, thickness)


def _draw_micr_symbol(img, x, base, size, kind, color):
    """E-13B-style transit / on-us symbols drawn as bars (OCR-A has no MICR glyphs)."""
    h = int(size * 0.7)
    if kind == "onus":
        cv2.rectangle(img, (x, base - h), (x + 3, base), color, -1)
        cv2.rectangle(img, (x + 6, base - h), (x + 11, base - h + 5), color, -1)
        cv2.rectangle(img, (x + 6, base - 5), (x + 11, base), color, -1)
    else:
        cv2.rectangle(img, (x, base - h), (x + 4, base - h // 2), color, -1)
        cv2.rectangle(img, (x + 7, base - h // 2), (x + 11, base), color, -1)
    return x + 16


def print_leaf(template, content, rng):
    """Paper plus all pre-printed elements. Returns (image, printed field boxes, label positions)."""
    a = template.anchors
    trng = np.random.default_rng(template.seed)
    img = security_paper(H, W, trng, rng)
    col, acc = template.print_color, template.accent_color
    font = template.print_font
    pw = lambda size, c=col: _printed_writer(font, size, c)
    fields = {}

    # bank logo + name, branch, IFSC
    cx, cy = int(W * 0.045), int(H * 0.095)
    cv2.circle(img, (cx, cy), 26, acc, 3)
    write_text(img, template.bank_name[0], pw(30, acc), cx - 10, cy + 11, rng, printed=True)
    write_text(img, template.bank_name, pw(a["header_size"], acc), int(W * 0.085), int(H * 0.10), rng, printed=True)
    write_text(img, template.branch, pw(17), int(W * 0.085), int(H * 0.15), rng, printed=True)
    box, _ = write_text(img, f"IFSC: {content.ifsc}", pw(18), int(W * 0.085), int(H * 0.195), rng, printed=True)
    fields["ifsc"] = box.pad(4, 3)
    write_text(img, "Valid for three months from the date of issue", pw(12), a["date_x"] - 10,
               a["date_y"] - 8, rng, printed=True)

    # date boxes
    x = a["date_x"]
    labels = "DDMMYYYY"
    date_boxes = []
    for i in range(8):
        cv2.rectangle(img, (x, a["date_y"]), (x + a["date_box"], a["date_y"] + a["date_h"]), col, 1)
        write_text(img, labels[i], pw(10), x + a["date_box"] // 2 - 3, a["date_y"] + a["date_h"] + 11, rng, printed=True)
        date_boxes.append(BBox(x, a["date_y"], a["date_box"], a["date_h"]))
        x += a["date_box"] + a["date_gap"] + (4 if i in (1, 3) else 0)
    fields["date"] = BBox.from_xyxy(date_boxes[0].x, date_boxes[0].y, date_boxes[-1].x2, date_boxes[-1].y2).pad(3, 3)

    # pay line
    write_text(img, "Pay", pw(22), int(W * 0.03), a["pay_y"] - 6, rng, printed=True)
    if template.hindi_labels:
        _put_unicode(img, "भुगतान करें", int(W * 0.03), a["pay_y"] - 46, 12, col)
    _hline(img, int(W * 0.03) + a["pay_label_w"], a["bearer_x"] - 8, a["pay_y"], col, template.dotted_lines)
    write_text(img, "OR BEARER", pw(14), a["bearer_x"], a["pay_y"] - 4, rng, printed=True)
    fields["payee"] = BBox.from_xyxy(int(W * 0.03) + a["pay_label_w"], a["pay_y"] - int(H * 0.085),
                                     a["bearer_x"] - 8, a["pay_y"] + 4)

    # rupees line(s)
    write_text(img, "Rupees", pw(18), int(W * 0.03), a["words_y1"] - 6, rng, printed=True)
    if template.hindi_labels:
        _put_unicode(img, "रुपये", int(W * 0.03), a["words_y1"] - 38, 13, col)
    words_x = int(W * 0.03) + int(W * 0.075)
    _hline(img, words_x, a["words_line1_end"], a["words_y1"], col, template.dotted_lines)
    bottom = a["words_y1"]
    if a["words_y2"]:
        _hline(img, int(W * 0.03), a["words_end"], a["words_y2"], col, template.dotted_lines)
        bottom = a["words_y2"]
    fields["amount_words"] = BBox.from_xyxy(int(W * 0.03) if a["words_y2"] else words_x,
                                            a["words_y1"] - int(H * 0.09), a["words_line1_end"], bottom + 4)

    # amount box with printed rupee sign
    bx, by, bw, bh = a["box_x"], a["box_y"], a["box_w"], a["box_h"]
    if template.amount_box_border:
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), col, 2)
    _put_unicode(img, "₹", bx + 6, by + int(bh * 0.15), int(bh * 0.55), col, font_name=RUPEE_FONT)
    fields["amount_numeric"] = BBox(bx + int(bh * 0.5), by + 2, bw - int(bh * 0.5) - 2, bh - 4)

    # account number (printed)
    write_text(img, "A/c No.", pw(15), a["acct_x"], a["acct_y"] - 4, rng, printed=True)
    acct_box, _ = write_text(img, content.account_number, pw(22), a["acct_x"] + 75, a["acct_y"], rng, printed=True)
    cv2.rectangle(img, (acct_box.x - 6, acct_box.y - 6), (acct_box.x2 + 6, acct_box.y2 + 6), col, 1)
    fields["account_number"] = acct_box.pad(6, 6)
    write_text(img, "SB / CA / OD", pw(12), a["acct_x"], a["acct_y"] + 26, rng, printed=True)

    # signature zone
    write_text(img, "Please sign above", pw(14), a["sig_x"] + int(W * 0.08), a["sig_bottom"] + 18, rng, printed=True)
    write_text(img, "Authorised Signatory", pw(12), a["sig_x"] + int(W * 0.09), a["sig_bottom"] + 34, rng, printed=True)
    fields["signature"] = BBox.from_xyxy(a["sig_x"], a["sig_y"], int(W * 0.975), a["sig_bottom"])

    # MICR band (OCR-A digits as an E-13B stand-in)
    micr = _printed_writer(MICR_FONT, a["micr_size"], (25, 25, 25))
    x, base = a["micr_x"], a["micr_base"]
    x = _draw_micr_symbol(img, x, base, a["micr_size"], "onus", (25, 25, 25))
    chq_box, _ = write_text(img, content.cheque_number, micr, x, base, rng, printed=True)
    x = _draw_micr_symbol(img, chq_box.x2 + 4, base, a["micr_size"], "onus", (25, 25, 25)) + 18
    b2, _ = write_text(img, content.micr_code, micr, x, base, rng, printed=True)
    x = _draw_micr_symbol(img, b2.x2 + 4, base, a["micr_size"], "transit", (25, 25, 25)) + 18
    b3, _ = write_text(img, content.account_id, micr, x, base, rng, printed=True)
    x = _draw_micr_symbol(img, b3.x2 + 4, base, a["micr_size"], "onus", (25, 25, 25)) + 18
    b4, _ = write_text(img, content.txn_code, micr, x, base, rng, printed=True)
    fields["cheque_number"] = chq_box.pad(4, 4)
    fields["micr"] = BBox.from_xyxy(a["micr_x"] - 4, min(chq_box.y, b2.y) - 5, b4.x2 + 5, base + 6)
    write_text(img, "CTS-2010", pw(11), int(W * 0.03), int(H * 0.80), rng, printed=True)
    return img, fields, date_boxes


def _put_unicode(img, text, x, y, size, color, font_name="Nirmala.ttf"):
    from PIL import Image, ImageDraw
    from .fonts import load_font
    font = load_font(font_name, size)
    w = int(font.getlength(text)) + 4
    mask = Image.new("L", (w, int(size * 1.6)), 0)
    ImageDraw.Draw(mask).text((2, 0), text, font=font, fill=255)
    alpha = np.asarray(mask, np.float32) / 255.0
    composite_ink(img, alpha, x, y, color, 1.0, None, blur=0.3)


def write_handwriting(img, template, content, writer, payee_writer, date_boxes, rng):
    """Fill the leaf by hand. Returns (chars per field, ink boxes per field)."""
    a = template.anchors
    chars, ink = {}, {}

    # date: one digit per box
    digits = content.date.strftime("%d%m%Y")
    dw = writer if writer.family not in ("kunstler", "palace", "french_script", "vladimir", "rage") else payee_writer
    date_chars = []
    for d, box in zip(digits, date_boxes):
        dwr = dw.variant(rng, ink_shift=False)
        dwr.size = int(box.h * rng.uniform(0.85, 1.05))
        cov, cb = write_text(img, d, dwr, box.x + box.w * rng.uniform(0.12, 0.3),
                             box.y + box.h * rng.uniform(0.78, 0.9), rng)
        date_chars += cb
    chars["date"] = date_chars
    ink["date"] = _union([b for _, b in date_chars])

    # payee
    pz = template_zone_payee(template)
    cov, cb = write_text(img, content.payee, payee_writer, pz[0] + rng.uniform(10, 80), a["pay_y"] - rng.uniform(5, 10),
                         rng, max_width=pz[1] - pz[0] - 90)
    chars["payee"], ink["payee"] = cb, cov

    # amount in words (wrap to the second line when the template has one)
    words = content.words_text
    line1_x = int(W * 0.03) + int(W * 0.075) + rng.uniform(10, 40)
    line1_w = a["words_line1_end"] - line1_x - 10
    alpha_w = _text_width(words, writer, rng)
    word_chars = []
    if a["words_y2"] and alpha_w > line1_w * 0.95:
        parts = words.split()
        cut = len(parts)
        while cut > 1 and _text_width(" ".join(parts[:cut]), writer, rng) > line1_w * 0.95:
            cut -= 1
        c1, cb1 = write_text(img, " ".join(parts[:cut]), writer, line1_x, a["words_y1"] - rng.uniform(5, 10), rng,
                             max_width=int(line1_w))
        c2, cb2 = write_text(img, " ".join(parts[cut:]), writer, int(W * 0.03) + rng.uniform(5, 30),
                             a["words_y2"] - rng.uniform(5, 10), rng, max_width=a["words_end"] - int(W * 0.05))
        word_chars = cb1 + cb2
        ink["amount_words"] = _union([c1, c2])
    else:
        c1, cb1 = write_text(img, words, writer, line1_x, a["words_y1"] - rng.uniform(5, 10), rng,
                             max_width=int(line1_w))
        word_chars, ink["amount_words"] = cb1, c1
    chars["amount_words"] = word_chars

    # figures in the amount box; some writers leave a gap after the rupee sign
    nb = BBox(a["box_x"], a["box_y"], a["box_w"], a["box_h"])
    aw = writer.variant(rng, ink_shift=False)
    aw.size = int(nb.h * rng.uniform(0.62, 0.78))
    if writer.family in ("kunstler", "palace", "french_script", "vladimir", "rage"):
        aw = payee_writer.variant(rng, ink_shift=False)
        aw.size = int(nb.h * rng.uniform(0.62, 0.78))
    start = nb.x + int(nb.h * 0.6) + int(rng.uniform(4, 60))
    cov, cb = write_text(img, content.amount_text, aw, start, nb.y + nb.h * rng.uniform(0.72, 0.82), rng,
                         max_width=nb.x2 - start - 6)
    chars["amount_numeric"], ink["amount_numeric"] = cb, cov
    return chars, ink


def template_zone_payee(template):
    a = template.anchors
    return int(W * 0.03) + a["pay_label_w"], a["bearer_x"]


def _text_width(text, writer, rng):
    from .fonts import load_font
    return load_font(writer.font, writer.size).getlength(text) * writer.spacing


def _union(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return BBox.from_xyxy(min(b.x for b in boxes), min(b.y for b in boxes),
                          max(b.x2 for b in boxes), max(b.y2 for b in boxes))


def place_signature(img, zone, sig_gray, ink, rng, pressure=0.95):
    """Composite a scanned signature into the signature zone. Returns the covered BBox."""
    alpha = ink_alpha_from_scan(sig_gray)
    ys, xs = np.nonzero(alpha > 0.15)
    if len(xs) == 0:
        return None
    alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    target_h = zone.h * rng.uniform(0.65, 0.9)
    scale = min(target_h / alpha.shape[0], zone.w * 0.85 / alpha.shape[1])
    alpha = cv2.resize(alpha, (max(4, int(alpha.shape[1] * scale)), max(4, int(alpha.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    alpha = np.clip(cv2.dilate(alpha, np.ones((2, 2), np.uint8)) * 1.6, 0, 1)
    x = zone.x + int(rng.uniform(0, max(1, zone.w - alpha.shape[1])))
    y = zone.y + int(rng.uniform(0, max(1, zone.h - alpha.shape[0])))
    return composite_ink(img, alpha, x, y, ink, pressure, rng, blur=0.5)


def generate_cheque(template_id, rng, split=None, signature=None, content=None, writer=None):
    """One genuine synthetic cheque.

    signature: (writer_id, sample_index) from CEDAR, or None to choose one from the split
    (falls back to a rendered signature when CEDAR is unavailable).
    """
    template = make_template(template_id)
    split = split or template_split(template_id)
    content = content or random_content(rng, template)
    writer = writer or random_writer(rng, split)
    payee_writer = writer if rng.random() > 0.15 else random_writer(rng, split, digits=True)
    if writer.family in ("kunstler", "palace", "french_script", "vladimir", "rage") and payee_writer is writer:
        payee_writer = random_writer(rng, split, digits=True, ink_name=writer.ink_name)
    img, fields, date_boxes = print_leaf(template, content, rng)
    paper = img.copy()
    chars, ink = write_handwriting(img, template, content, writer, payee_writer, date_boxes, rng)

    sig_info = {}
    if sigs.cedar_available():
        if signature is None:
            wid = int(rng.choice(sigs.WRITER_SPLITS[split]))
            signature = (wid, int(rng.integers(1, sigs.SAMPLES_PER_WRITER + 1)))
        sig_img = sigs.load_cedar(*signature)
        sig_info = {"source": "cedar", "writer": signature[0], "index": signature[1], "forged": False}
    else:
        sig_img = sigs.synthetic_signature(content.payee, rng, writer.font)
        sig_info = {"source": "synthetic", "writer": None, "index": None, "forged": False}
    ink["signature"] = place_signature(img, fields["signature"], sig_img, writer.ink, rng)
    return SyntheticCheque(image=img, paper=paper, fields=fields, content=content, template=template,
                           writer=writer, payee_writer=payee_writer, chars=chars, ink_boxes=ink,
                           signature=sig_info, split=split)
