"""Paths, canonical geometry and thresholds shared by the cheque forensics modules.

Detection thresholds that come from validation data are not stored here; they
are written by the training scripts into models/*.json and loaded at runtime.
"""
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
METRICS_DIR = os.path.join(MODELS_DIR, "metrics")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
SYNTH_DIR = os.path.join(DATA_DIR, "synthetic")
REGISTRY_DIR = os.path.join(DATA_DIR, "signature_registry")
FONTS_DIR = os.environ.get("CHEQUE_FONTS_DIR", r"C:\Windows\Fonts")

# CTS-2010 cheque leaf: 8 in x 3.67 in (203.2 mm x 93.98 mm)
CTS_ASPECT = 203.2 / 93.98
CANONICAL_WIDTH = 1200
CANONICAL_HEIGHT = round(CANONICAL_WIDTH / CTS_ASPECT)   # 555
ASPECT_RANGE = (1.35, 2.9)       # accepted width/height of a detected cheque quadrilateral

FIELDS = (
    "payee", "date", "amount_numeric", "amount_words", "cheque_number",
    "account_number", "ifsc", "signature", "micr",
)
HANDWRITTEN_FIELDS = ("payee", "date", "amount_numeric", "amount_words")
PRINTED_FIELDS = ("cheque_number", "account_number", "ifsc", "micr")

# Model input sizes
FIELD_SEG_SIZE = (128, 288)      # (height, width) of the field segmentation input
SEG_BASE_CHANNELS = 12           # U-Net width for the field and boundary segmenters
CHEQUE_SEG_SIZE = (192, 192)     # boundary segmentation input
TEXT_HEIGHT, TEXT_WIDTH = 32, 384
SIGNATURE_SIZE = (96, 192)       # (height, width)
TAMPER_PATCH = (64, 256)         # (height, width) of field windows for tamper models

# Model files
FIELD_SEG_WEIGHTS = os.path.join(MODELS_DIR, "field_segmenter.pt")
CHEQUE_SEG_WEIGHTS = os.path.join(MODELS_DIR, "cheque_segmenter.pt")
TEXT_RECOGNIZER_WEIGHTS = os.path.join(MODELS_DIR, "text_recognizer.pt")
SIAMESE_WEIGHTS = os.path.join(MODELS_DIR, "siamese_signature.pt")
SIGNATURE_CALIBRATION = os.path.join(MODELS_DIR, "signature_calibration.json")
TAMPER_CLASSIFIER_WEIGHTS = os.path.join(MODELS_DIR, "tamper_classifier.pt")
TAMPER_AUTOENCODER_WEIGHTS = os.path.join(MODELS_DIR, "tamper_autoencoder.pt")
TAMPER_UNET_WEIGHTS = os.path.join(MODELS_DIR, "tamper_unet.pt")
TAMPER_CALIBRATION = os.path.join(MODELS_DIR, "tamper_calibration.json")
RISK_CALIBRATION = os.path.join(MODELS_DIR, "risk_calibration.json")

# Input validation for uploads
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Retention policy: uploaded cheques are processed in memory and never written
# to disk by the web app. Analysis results (including rendered images) are kept
# in memory for this long so the PDF report can be downloaded, then purged.
RESULT_RETENTION_SECONDS = int(os.environ.get("CHEQUE_RESULT_RETENTION_SECONDS", 30 * 60))
MAX_CACHED_RESULTS = 20

# Format gate: images unlike the CTS-2010 cheques the models were trained on are capped at REQUIRES REVIEW.
# Layout agreement of the field segmenter was >= 0.39 on every synthetic validation cheque, 0.22 on a US cheque.
CTS_FORMAT_MIN_LAYOUT = 0.30
CTS_FORMAT_ASPECT = (1.9, 2.45)

# OCR
MIN_TEXT_CONFIDENCE = 0.60       # CRNN mean character probability needed to trust a reading
CHEQUE_VALIDITY_MONTHS = 3       # RBI: cheques are valid for three months from the date of issue
