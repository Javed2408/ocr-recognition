"""Cheque OCR and digit recognition.

Pipeline:
  1. Validate the input image (resolution, blank page, brightness, contrast,
     sharpness, skew) and stop early if it is unusable.
  2. Binarise it and run Tesseract OCR on the whole image.
  3. Find character-sized contours and classify each one with a CNN trained
     on MNIST.
  4. Validate the results (OCR confidence, digits found, numeric amount vs
     written amount on the cheque), then save the annotated image and verify
     the saved file.
"""
import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import cv2
import matplotlib.pyplot as plt
import numpy as np

# TensorFlow is imported inside the functions that use it, so other modules
# (the cheque_forensics package) can reuse the image checks and amount parsing
# without paying for the TensorFlow import.

try:
    import pytesseract
except ImportError:  # OCR is optional; digit recognition still works without it
    pytesseract = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(SCRIPT_DIR, "check.png")
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "digit_cnn.h5")
WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Input image quality thresholds
MIN_SIZE = (300, 150)            # (width, height) in pixels
BRIGHTNESS_RANGE = (40, 245)     # acceptable mean grey level
MIN_CONTRAST = 30.0              # standard deviation of grey levels
MIN_SHARPNESS = 100.0            # variance of the Laplacian; lower means blurrier
MIN_INK_RATIO = 0.005            # fraction of pixels that are text; below this the page is blank
MAX_SKEW_DEG = 3.0

# Digit detection and result thresholds
MIN_DIGIT_AREA = 100             # bounding-box area in pixels
MIN_DIGIT_HEIGHT = 10            # pixels
MAX_DIGIT_HEIGHT_FRAC = 0.25     # of image height; larger regions are borders or boxes
MIN_HEIGHT_VS_MEDIAN = 0.5       # blobs shorter than this x median character height are punctuation
DIGIT_ASPECT_RANGE = (0.8, 6.0)  # height / width
MIN_CONFIDENCE = 0.80            # CNN softmax score needed to accept a prediction
MIN_OCR_CONFIDENCE = 60.0        # mean Tesseract word confidence (0-100)


# ---------------------------------------------------------------------------
# Check reporting
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    status: str  # PASS, WARN, FAIL or SKIP
    detail: str


def print_report(title, checks):
    print(f"\n=== {title} ===")
    for check in checks:
        print(f"  [{check.status}] {check.name}: {check.detail}")


def has_failures(checks):
    return any(check.status == "FAIL" for check in checks)


# ---------------------------------------------------------------------------
# 1. CNN model definition and training
# ---------------------------------------------------------------------------
def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_save_model(model_path=DEFAULT_MODEL):
    import tensorflow as tf
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
    X_test = (X_test.astype("float32") / 255.0).reshape(-1, 28, 28, 1)

    model = build_cnn_model()
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model


def load_model(model_path=DEFAULT_MODEL):
    if not os.path.exists(model_path):
        print(f"No model at {model_path}; training a new one on MNIST...")
        return train_and_save_model(model_path)
    print(f"Loading model from {model_path}")
    import tensorflow as tf
    return tf.keras.models.load_model(model_path, compile=False)  # inference only


# ---------------------------------------------------------------------------
# 2. Preprocessing and input image checks
# ---------------------------------------------------------------------------
def binarize(gray):
    """Otsu threshold with text white (255) on black, as findContours and MNIST expect.

    Returns (binary, inverted), where inverted is True when the source had light
    text on a dark background and was flipped.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inverted = np.count_nonzero(binary) > binary.size / 2
    if inverted:
        binary = cv2.bitwise_not(binary)
    return binary, inverted


def estimate_skew(binary):
    """Median angle in degrees of long, near-horizontal lines (borders, underlines), or None."""
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=100,
                            minLineLength=binary.shape[1] // 4, maxLineGap=10)
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = (angle + 90) % 180 - 90  # fold into [-90, 90)
        if abs(angle) < 45:
            angles.append(angle)
    return float(np.median(angles)) if angles else None


def validate_input_image(image, gray, binary, inverted):
    """Check that the image is usable for OCR before running the expensive steps."""
    checks = []
    h, w = image.shape[:2]
    min_w, min_h = MIN_SIZE

    if w >= min_w and h >= min_h:
        checks.append(Check("Resolution", "PASS", f"{w}x{h}"))
    else:
        checks.append(Check("Resolution", "FAIL", f"{w}x{h} is below the minimum {min_w}x{min_h}"))

    ink = np.count_nonzero(binary) / binary.size
    polarity = " (light text on dark background, inverted)" if inverted else ""
    if ink < MIN_INK_RATIO:
        checks.append(Check("Content", "FAIL", f"only {ink:.2%} text pixels, so the image looks blank"))
    else:
        checks.append(Check("Content", "PASS", f"{ink:.2%} text pixels{polarity}"))

    brightness = float(gray.mean())
    low, high = BRIGHTNESS_RANGE
    if brightness < low:
        checks.append(Check("Brightness", "WARN", f"mean {brightness:.0f}/255 is too dark"))
    elif brightness > high:
        checks.append(Check("Brightness", "WARN", f"mean {brightness:.0f}/255 looks overexposed"))
    else:
        checks.append(Check("Brightness", "PASS", f"mean {brightness:.0f}/255"))

    contrast = float(gray.std())
    status = "PASS" if contrast >= MIN_CONTRAST else "WARN"
    checks.append(Check("Contrast", status, f"std dev {contrast:.1f} (minimum {MIN_CONTRAST:.0f})"))

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness >= MIN_SHARPNESS:
        checks.append(Check("Sharpness", "PASS", f"Laplacian variance {sharpness:.0f}"))
    else:
        checks.append(Check("Sharpness", "WARN",
                            f"Laplacian variance {sharpness:.0f} (minimum {MIN_SHARPNESS:.0f}), image may be blurry"))

    skew = estimate_skew(binary)
    if skew is None:
        checks.append(Check("Skew", "SKIP", "no long straight lines to measure against"))
    elif abs(skew) > MAX_SKEW_DEG:
        checks.append(Check("Skew", "WARN", f"{skew:+.1f} deg (limit {MAX_SKEW_DEG:.0f}), consider straightening the image"))
    else:
        checks.append(Check("Skew", "PASS", f"{skew:+.1f} deg"))

    return checks


# ---------------------------------------------------------------------------
# 3. OCR (Tesseract)
# ---------------------------------------------------------------------------
def setup_tesseract(cmd=None):
    """Point pytesseract at a Tesseract binary. Returns True if one is usable."""
    if pytesseract is None:
        return False
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    elif shutil.which("tesseract") is None and os.path.exists(WINDOWS_TESSERACT):
        pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT
    try:
        pytesseract.get_tesseract_version()
    except (pytesseract.TesseractNotFoundError, OSError):
        return False
    return True


def run_ocr(binary):
    """Returns (text, mean word confidence 0-100)."""
    page = cv2.bitwise_not(binary)  # Tesseract expects dark text on a light background
    data = pytesseract.image_to_data(page, config="--oem 3 --psm 6",
                                     output_type=pytesseract.Output.DICT)
    lines, confidences = {}, []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confidences.append(conf)
    text = "\n".join(" ".join(words) for words in lines.values())
    return text, (float(np.mean(confidences)) if confidences else 0.0)


# ---------------------------------------------------------------------------
# 4. CNN digit recognition
# ---------------------------------------------------------------------------
def find_digit_candidates(binary):
    """Bounding boxes of character-sized blobs, roughly in reading order.

    Uses connected components rather than external contours so that characters
    inside a printed box (like the cheque's amount box) are still found.
    """
    img_h = binary.shape[0]
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for x, y, w, h, _ in stats[1:]:  # label 0 is the background
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w * h < MIN_DIGIT_AREA or h < MIN_DIGIT_HEIGHT or h > img_h * MAX_DIGIT_HEIGHT_FRAC:
            continue
        if not DIGIT_ASPECT_RANGE[0] <= h / w <= DIGIT_ASPECT_RANGE[1]:
            continue  # underlines, wide boxes, stray marks
        boxes.append((x, y, w, h))
    if not boxes:
        return boxes
    median_h = float(np.median([b[3] for b in boxes]))
    boxes = [b for b in boxes if b[3] >= median_h * MIN_HEIGHT_VS_MEDIAN]  # drop dots, commas
    row_h = max(1, int(median_h))
    boxes.sort(key=lambda b: ((b[1] + b[3] // 2) // row_h, b[0]))
    return boxes


def to_mnist_format(binary, box):
    """Crop a character and centre it in a 28x28 frame like MNIST (20px glyph, 4px margin)."""
    x, y, w, h = box
    roi = binary[y:y + h, x:x + w]
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = roi / 255.0
    return canvas


def recognize_digits(binary, model):
    """Returns ([(box, digit, confidence), ...], number of candidate regions)."""
    boxes = find_digit_candidates(binary)
    if not boxes:
        return [], 0
    batch = np.stack([to_mnist_format(binary, b) for b in boxes])[..., np.newaxis]
    probs = model.predict(batch, verbose=0)  # one batched call instead of one per region
    digits = [(box, int(p.argmax()), float(p.max()))
              for box, p in zip(boxes, probs) if p.max() >= MIN_CONFIDENCE]
    return digits, len(boxes)


def annotate(image, digits):
    annotated = image.copy()
    for (x, y, w, h), digit, _ in digits:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, str(digit), (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
    return annotated


# ---------------------------------------------------------------------------
# 5. Output checks
# ---------------------------------------------------------------------------
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
SCALES = {"hundred": 100, "thousand": 1_000, "lakh": 100_000, "million": 1_000_000, "crore": 10_000_000}


def words_to_number(words):
    """['four', 'hundred', 'ten'] -> 410"""
    total = current = 0
    for word in words:
        if word in UNITS:
            current += UNITS[word]
        elif word in TENS:
            current += TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word in SCALES:
            total += max(current, 1) * SCALES[word]
            current = 0
    return total + current


def parse_written_amount(text):
    """Value of the longest run of number words in the text (e.g. 'Four Hundred Ten')."""
    best, run = [], []
    for token in re.findall(r"[a-z]+", text.lower()) + [""]:
        if token in UNITS or token in TENS or token in SCALES or (token == "and" and run):
            run.append(token)
            continue
        words = [t for t in run if t != "and"]
        if len(words) > len(best):
            best = words
        run = []
    return words_to_number(best) if best else None


def parse_numeric_amount(text):
    """First dollar or rupee amount in the text, e.g. '$410.00' -> 410.0, '₹1,25,000/-' -> 125000.0

    Accepts Western (1,250,000) and Indian (12,50,000) digit grouping.
    """
    match = re.search(r"(?:\$|₹|\bRs\.?|\bINR)\s*(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d{2}))?", text)
    if not match:
        return None
    return float(match.group(1).replace(",", "") + "." + (match.group(2) or "00"))


def check_amounts(text):
    """Cross-check the numeric amount box against the amount written in words."""
    if not text:
        return Check("Amount match", "SKIP", "needs OCR text")
    numeric, written = parse_numeric_amount(text), parse_written_amount(text)
    if numeric is None and written is None:
        return Check("Amount match", "SKIP", "no amount found, image may not be a cheque")
    if numeric is None or written is None:
        missing = "numeric ($)" if numeric is None else "written"
        return Check("Amount match", "WARN", f"could not find the {missing} amount")
    if int(numeric) == written:
        return Check("Amount match", "PASS", f"${numeric:,.2f} matches the written amount ({written:,})")
    return Check("Amount match", "FAIL", f"numeric ${numeric:,.2f} does not match the written amount ({written:,})")


def validate_results(ocr_text, ocr_conf, digits, candidates):
    checks = []
    if ocr_text is None:
        checks.append(Check("OCR", "SKIP", "Tesseract not available"))
    elif not ocr_text.strip():
        checks.append(Check("OCR", "FAIL", "no text extracted"))
    else:
        status = "PASS" if ocr_conf >= MIN_OCR_CONFIDENCE else "WARN"
        checks.append(Check("OCR confidence", status,
                            f"mean word confidence {ocr_conf:.1f}/100 (minimum {MIN_OCR_CONFIDENCE:.0f})"))

    if digits:
        checks.append(Check("Digits", "PASS", f"{len(digits)} of {candidates} character regions "
                                              f"classified with >= {MIN_CONFIDENCE:.0%} confidence"))
    else:
        checks.append(Check("Digits", "WARN", f"none of {candidates} character regions classified "
                                              f"with >= {MIN_CONFIDENCE:.0%} confidence"))

    checks.append(check_amounts(ocr_text))
    return checks


def save_and_verify(image, path):
    """Write the annotated image, then read it back to confirm the file is valid."""
    try:
        written = cv2.imwrite(path, image)
    except cv2.error as exc:
        return Check("Output image", "FAIL", f"could not write {path}: {exc}")
    if not written:
        return Check("Output image", "FAIL", f"could not write {path}")
    reloaded = cv2.imread(path)
    if reloaded is None:
        return Check("Output image", "FAIL", f"{path} was written but cannot be read back")
    if reloaded.shape != image.shape:
        return Check("Output image", "FAIL", f"{path} has shape {reloaded.shape}, expected {image.shape}")
    h, w = image.shape[:2]
    return Check("Output image", "PASS", f"saved and verified {path} ({w}x{h})")


# ---------------------------------------------------------------------------
# 6. Display and main
# ---------------------------------------------------------------------------
def show_pipeline(image, gray, binary, annotated):
    panels = [
        (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Original Image", None),
        (gray, "Grayscale Image", "gray"),
        (cv2.bitwise_not(binary), "Thresholded Image", "gray"),
        (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "Digit Recognition Output", None),
    ]
    plt.figure(figsize=(12, 8))
    for i, (img, title, cmap) in enumerate(panels, start=1):
        plt.subplot(2, 2, i)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OCR and CNN digit recognition for cheque images, with image quality checks.")
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE,
                        help="input image (default: check.png next to this script)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Keras model file; a new one is trained on MNIST if it is missing")
    parser.add_argument("--output",
                        help="where to save the annotated image (default: <image>_annotated.png)")
    parser.add_argument("--tesseract-cmd",
                        help="path to the tesseract executable if it is not on PATH")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="don't open the matplotlib window")
    return parser.parse_args(argv)


def main(argv=None):
    # Windows uses a legacy encoding when output is piped or redirected, which garbles
    # OCR text such as curly quotes and can crash on characters it cannot represent.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    output_path = args.output or os.path.splitext(args.image)[0] + "_annotated.png"

    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        return 1
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: {args.image} is not a readable image file")
        return 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary, inverted = binarize(gray)

    input_checks = validate_input_image(image, gray, binary, inverted)
    print_report("Input image checks", input_checks)
    if has_failures(input_checks):
        print("\nInput image failed quality checks; stopping.")
        return 1

    model = load_model(args.model)
    digits, candidates = recognize_digits(binary, model)
    annotated = annotate(image, digits)
    print("\nRecognised digits (CNN, reading order):", " ".join(str(d) for _, d, _ in digits) or "none")

    ocr_text, ocr_conf = None, 0.0
    if setup_tesseract(args.tesseract_cmd):
        ocr_text, ocr_conf = run_ocr(binary)
        print("\n<<< OCR Result >>>\n" + ocr_text)
    else:
        print("\nTesseract OCR is not available, so text extraction was skipped. "
              "Install pytesseract and the Tesseract binary, or pass --tesseract-cmd.")

    output_checks = validate_results(ocr_text, ocr_conf, digits, candidates)
    output_checks.append(save_and_verify(annotated, output_path))
    print_report("Output checks", output_checks)

    if args.show:
        show_pipeline(image, gray, binary, annotated)
    return 1 if has_failures(output_checks) else 0


if __name__ == "__main__":
    sys.exit(main())
