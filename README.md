# Cheque OCR, Digit Recognition and Forgery-Risk Analysis

Reads a cheque image with Tesseract OCR and classifies each character-sized region with a small CNN trained on MNIST. The script checks the input image before processing and checks the results afterwards.

## Setup

```
pip install -r requirements.txt
```

OCR also needs the Tesseract program. On Windows, install it from the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki). The script finds it automatically in `C:\Program Files\Tesseract-OCR`. Otherwise, add it to `PATH` or pass `--tesseract-cmd`. Without Tesseract, the script still runs the image checks and CNN digit recognition and skips the OCR steps.

`digit_cnn.h5` is the trained model. If it is missing, a new one is trained on MNIST (5 epochs) and saved.

## Usage

```
python aiproject.py                       # uses check.png next to the script
python aiproject.py path\to\cheque.png    # any other image
python aiproject.py --no-show --output result.png
```

| Option | Meaning |
|---|---|
| `--output PATH` | Where to save the annotated image (default `<image>_annotated.png`) |
| `--model PATH` | Model file to load, or to train and save if it is missing |
| `--tesseract-cmd PATH` | Tesseract executable, if it is not on `PATH` |
| `--no-show` | Don't open the matplotlib window |

The exit code is `0` on success and `1` if any check fails.

## Checks

**Input image** (runs before the model loads; any `FAIL` stops the run):

| Check | What it looks at | Result |
|---|---|---|
| Resolution | Width and height at least 300x150 | FAIL |
| Content | Share of text pixels after thresholding, to catch blank images | FAIL |
| Brightness | Mean grey level between 40 and 245 | WARN |
| Contrast | Standard deviation of grey levels at least 30 | WARN |
| Sharpness | Variance of the Laplacian at least 100 (lower means blurry) | WARN |
| Skew | Angle of long straight lines (border, underlines) within ±3° | WARN |

Light text on a dark background is detected and inverted automatically.

**Output:**

| Check | What it looks at |
|---|---|
| OCR confidence | Mean Tesseract word confidence at least 60 |
| Digits | At least one region classified with 80% or higher CNN confidence |
| Amount match | The `$` amount equals the amount written in words (e.g. `$410.00` and "Four Hundred Ten") |
| Output image | The annotated image was written and reads back with the right size |

All thresholds are constants at the top of `aiproject.py`.

## Tests

```
python -m pytest -q
```

## Known limitations

The CNN has only the ten MNIST digit classes, so it labels every character-shaped region as a digit, letters included. Printed fonts also differ from MNIST handwriting. For example, a `1` with a flag at the top is often read as `7`. Read the digit labels next to the OCR text, not on their own.

The `$` amount parser now also accepts `₹`, `Rs.` and `INR`, and words with lakh and crore.

# Cheque forgery-risk analysis (Indian cheques)

The `cheque_forensics` package analyses CTS-2010 Indian cheques for signs of forgery and alteration: it detects the cheque and its fields, verifies the signature against registered samples with a Siamese CNN, and looks for altered amounts, dates and payee names using image forensics and deep models. **Results are risk indicators with evidence, not proof of forgery.** Design, datasets, evaluation results and limitations are in [docs/CHEQUE_FORENSICS.md](docs/CHEQUE_FORENSICS.md).

## Web interface

```
python -m webapp.app          # then open http://127.0.0.1:5000
```

The **Forgery detection** tab takes a cheque image (drag and drop), optionally the account holder's registered signatures and a trusted image of the same cheque taken before it was handed over. Results: risk category and score with the factors behind it, detected fields and extracted values, signature comparison, per-field alteration evidence, forensic heatmaps, every validation check, processing information and a PDF report. The **Digit OCR** tab runs the original pipeline above.

Uploads are processed in memory and never written to disk; results are kept in memory for 30 minutes (`CHEQUE_RESULT_RETENTION_SECONDS`) so the PDF can be downloaded, and can be deleted immediately.

## Command line

```
python -m cheque_forensics cheque.jpg --ref sig1.png --ref sig2.png --original before.jpg --pdf report.pdf --json result.json
```

Exit code: `0` for LOW RISK, `1` otherwise, `2` for unreadable input.

## Models and training

Trained weights and calibration files are in `models/`; evaluation results in `models/metrics/*.json`. To retrain everything (synthetic data generation, CEDAR download required for signatures):

```
# CEDAR signatures (research use): extract to data/raw/cedar/signatures/{full_org,full_forg}
curl -L -o signatures.rar https://cedar.buffalo.edu/NIJ/data/signatures.rar
bash training/run_all.sh      # 2-4 hours on a laptop CPU
```

Individual steps: `python -m training.train_segmenters`, `train_text_recognizer`, `train_siamese`, `train_tamper_models`, `calibrate_risk` (risk model + system evaluation), `evaluate_fields`. Models run on an NVIDIA GPU automatically when the CUDA build of PyTorch is installed; set `CHEQUE_DEVICE=cpu` to force the CPU.

## Tests

```
python -m pytest -q            # original tests + tests/ for every phase of the forgery analysis
```

Tests that need trained weights or the CEDAR data are skipped when those are missing.
