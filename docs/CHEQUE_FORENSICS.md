# Cheque forgery-risk analysis: design, data, evaluation and limitations

This document describes the `cheque_forensics` package added to the cheque OCR project. It is written for
engineers who will run, retrain or extend the system, and for reviewers who need to judge how far its outputs
can be trusted.

**Every output is a risk indicator with evidence, never proof.** The system was trained and calibrated on
synthetic Indian cheques and on the CEDAR signature set; its accuracy on real cheques has not been measured.

## 1. Architecture

```
upload ──► cheque_preprocessor ──► field_extractor ──► signature_extractor ──► signature_verifier (Siamese CNN)
           (cheque_detector:          (U-Net field        (ink vs. security        (registered references,
            boundary U-Net +           segmentation +      background)              calibrated 3-way decision)
            contours, perspective,     structure + OCR)
            orientation, deskew,              │
            CLAHE, threshold)                 ├──► amount_tampering_detector ─┐  (figures + words)
                                              ├──► field_alteration_detector ─┤  (date, payee)
                                              │      classical forensics + CNN/Grad-CAM + autoencoder + U-Net
                                              └──► document_forensics ────────┤  (ELA, noise, JPEG ghost/grid, LBP,
                                                     + registration with a     │   edges, frequency, copied glyphs)
                                                       trusted original        │
                                                                               ▼
                         suspicious_region_detector ──► validation_engine ──► risk_assessor ──► report_generator
                         (merged regions, heatmap)       (10 checks)           (learned logistic   (images, PDF)
                                                                               score + rules)
```

`pipeline.ChequeAnalyzer` runs the stages and records per-stage timings and which trained models were
available. `webapp/app.py` (Flask) and `python -m cheque_forensics` are the two front ends.

| Module | Responsibility |
|---|---|
| `cheque_detector.py` | Cheque boundary: Canny contours, bright-paper segmentation and a U-Net mask, scored by area, CTS-2010 aspect ratio, rectangularity and edge support; perspective warp to 1200 px width |
| `cheque_preprocessor.py` | Input quality checks (reused from `aiproject.py`), 0/180° orientation (field-layout score), deskew (`aiproject.estimate_skew`), median + non-local-means denoising, illumination flattening, CLAHE, Gaussian adaptive threshold, normalisation; homography back to the upload |
| `field_extractor.py` | Nine fields (payee, date, amount in figures and words, cheque number, account number, IFSC, signature, MICR) from a U-Net field segmenter, snapped to ruled lines / date boxes / amount box, CTS-2010 priors as fallback; bounding boxes stored in normalised and original coordinates |
| `ocr_engine.py`, `text_recognizer.py` | Built-in CRNN-CTC recogniser (field-constrained alphabets), Tesseract when installed (preferred for printed fields), the original MNIST CNN for per-digit ambiguity |
| `amount_parser.py` | Indian amounts (`1,25,000/-`, lakh/crore, `=00`), OCR-tolerant amount-in-words parsing, DDMMYYYY dates, RBI 3-month validity, IFSC/MICR parsing |
| `signature_extractor.py` | Ink/background separation by darkness and chroma, signature-line removal, aspect-preserving preprocessing |
| `signature_verifier.py` | Siamese CNN (shared weights, L2-normalised 128-d embeddings, contrastive loss), multi-reference scoring, calibrated thresholds, side-by-side comparison, hashed-key signature registry |
| `forensic_features.py` | Stroke width (distance transform), Local Binary Patterns, edge density, double-ink overlap, ink colour (Lab), paper texture around characters, robust z-scores within a field |
| `field_alteration_detector.py` | Per-field analysis combining classical evidence and the deep models into a calibrated fused score |
| `amount_tampering_detector.py` | Amount-specific checks: digit ambiguity (MNIST CNN), Indian digit grouping, open-ended figures, amount-in-words grammar |
| `tamper_models.py` | Inference for the tampering CNN (+Grad-CAM), convolutional autoencoder and U-Net |
| `document_forensics.py` | Whole-image forensics and registration (ORB + RANSAC homography, ECC fallback) with pixel comparison against a trusted original |
| `suspicious_region_detector.py` | Merges all localised evidence into numbered regions and a heatmap |
| `validation_engine.py` | The ten checks listed in section 4 |
| `risk_assessor.py` | Learned risk score, categories and per-factor contributions |
| `report_generator.py` | UI images and the PDF report |

Training code lives only in `training/`; inference loads weights from `models/` and runs on CUDA when a GPU
build of PyTorch sees a GPU (`CHEQUE_DEVICE=cpu|cuda` overrides), otherwise on CPU. All models in this
repository were trained on CPU (the installed PyTorch wheel is CPU-only); `pip install torch` with the CUDA
wheel enables the RTX GPU without code changes.

## 2. Datasets

| Dataset | Checked | Availability and licence | Labels | Used for |
|---|---|---|---|---|
| **CEDAR signatures** (Kalera, Srihari & Xu 2004) | 2026-11-03: `cedar.buffalo.edu/NIJ/data/signatures.rar` downloads (253,587,033 bytes) | Free for research; no explicit licence file. Not redistributed here (`data/` is git-ignored) | 55 writers × 24 genuine + 24 skilled forgeries | Siamese training/calibration/test; signatures placed on synthetic cheques |
| **SSBI / Synthetic Signature Bankcheck Images** (Khan et al., ICDAR 2024 workshop) | Release `ssbi-dataset-v1.0.0.zip`, 784 MB, at `github.com/dfki-av/bank-check-security` | CC BY-NC 4.0 (per README; GitHub shows "Other") | 4,360 US-style cheques, 19 signers, COCO boxes for genuine/forged signatures, date, amounts, payee | Not used: US layouts, few signers, no per-writer reference sets; a good next dataset for signature *detection* |
| **GPDS-960 / GPDS Synthetic** | Literature and figshare listing | GPDS-960 withdrawn / licence by signed fax request; GPDS Synthetic by request | Genuine + forgeries | Not used (not freely downloadable) |
| **IDRBT Cheque Image Dataset** (112 cheques, 4 Indian banks) | `idrbt.ac.in` returned HTTP 403 to automated download | Citation required; no stated licence | Ink/writer metadata, no field boxes | Not used. A 2026 derivative with field boxes and 295 synthetic cheques exists on Hugging Face (`royayush/cheque-synthetic-images`, Apache-2.0) but is gated behind a login |
| **Multilingual Signature Verification Dataset** (Hugging Face, `rakshitdabral/...`) | API listing: 315 signers × 24 images | MIT claimed, but it repackages CEDAR and Hindi/Bengali (BHSig260-style) data whose original terms differ | Genuine only, no forgeries | Not used: no forgery labels and overlapping writers would leak CEDAR test writers |
| **Synthetic Indian cheques** (this project, `cheque_forensics/synth/`) | Generated | Own data; fictional banks (Q-prefixed IFSC codes) | Pixel masks of every alteration, field boxes, field values, signature provenance | Field/boundary segmenters, CRNN, tamper models, all calibration and system evaluation |

No public dataset of altered *Indian* cheques with alteration labels was found, so alteration detection is
trained and evaluated on synthetic data generated with controlled modifications:

* **Layouts**: 40 fictional bank templates following CTS-2010 zones (date boxes top right, payee line, one- or
  two-line amount in words, amount box with printed ₹, account number, signature zone, MICR band), with random
  geometry, colours, guilloche security patterns, bilingual labels and dotted/solid lines.
* **Handwriting**: 21 handwriting-style font families with per-writer size, slant, spacing, stroke thickness,
  pen pressure, per-glyph jitter and elastic distortion, blue/black inks with flow variation; 15% of genuine
  cheques have the payee written by a different hand (natural cross-field variation).
* **Alterations** (pixel masks recorded): digit squeezed before the amount, digit appended after it (with the
  `/-` erased and rewritten), digit overwritten, pen strokes added (1→7, 1→4, 0→6/9, 3→8), digit erased
  (chemical or mechanical simulation) and rewritten, amount in words extended (`Six`→`Sixty`) or erased and
  rewritten, date digits overwritten/stroked/erased, payee appended or erased and rewritten, digit copied
  digitally, rectangular region pasted from another leaf (then JPEG re-compressed), and skilled signature
  forgeries from CEDAR. The forger uses the same hand and pen family 45% of the time.
* **Scan simulation**: exposure, blur, noise and JPEG (quality 70-96) on 60% of images; independent re-scans
  with small rotation/shift serve as "trusted originals".

**Leakage prevention.** Train/validation/test splits are disjoint in cheque templates (28/6/6), handwriting font
families (14/3/4), printed fonts, and CEDAR writers (1-35 / 36-45 / 46-55). Thresholds are always chosen on
validation data and metrics reported on test data.

## 3. Models

| Model | Input | Architecture | Objective |
|---|---|---|---|
| Boundary segmenter | 192×192 photo | U-Net (12 base channels, depth 4) | BCE + Dice on cheque mask |
| Field segmenter | 128×288 normalised cheque | U-Net (12, depth 4), 10 classes | weighted CE + Dice |
| Text recogniser | 32×384 line | CNN (7 conv) + BiLSTM(128) | CTC, 73 classes |
| Siamese verifier | 96×192 signature | shared CNN branch → 128-d L2-normalised embedding | contrastive loss (margin 1) on genuine/skilled/random pairs |
| Tampering CNN | 4×64×256 field window (BGR + high-pass) | 7 conv + global max pool | BCE; Grad-CAM on last conv layer |
| Autoencoder | same windows, genuine only | strided conv encoder (32-ch bottleneck) / transposed-conv decoder | L1 reconstruction |
| Tampering U-Net | same windows | U-Net (12, depth 3) | BCE + Dice on altered pixels |

Retrain everything with `bash training/run_all.sh` (2-4 hours on a 20-thread laptop CPU).

## 4. Validation checks and risk assessment

| Check | Fails when |
|---|---|
| Amount in figures vs words | both read with OCR confidence ≥ 0.60 and the values differ (WARN if confidence is low); factor-of-10 differences and digit-containment are called out |
| Signature verification | Siamese score above the calibrated reject threshold (WARN when inconclusive, SKIP without references) |
| Overwriting / alteration of the amount | fused figure or word evidence above its calibrated threshold |
| Date field anomalies | invalid, stale (> 3 months, RBI rule) or post-dated date, or date alteration evidence |
| Payee field anomalies | payee alteration evidence; blank payee warns |
| Image manipulation indicators | pixel differences versus a trusted original, pixel-identical (copied) characters, or corroborated image-forensic anomalies (WARN) |
| Missing or unreadable fields | critical fields not found or not parseable |
| OCR confidence | critical fields below 0.60 |
| Image quality | resolution, blank page, brightness, contrast, sharpness, skew (from `aiproject.py`) |
| Cheque format (CTS-2010) | WARN when the proportions are outside 1.9-2.45:1 or the field-layout agreement is below 0.30 (every synthetic validation cheque scored ≥ 0.39; the repository's US sample cheque scores 0.22). A WARN caps the result at REQUIRES REVIEW, because the detectors are unreliable outside their training domain |
| Ink across fields | information only: different pens between fields are common and never a fraud indicator |

**Thresholds.** Each detector's flagging threshold is the 95th percentile of its score on unaltered validation
fields (5% false alarms); the fused field score (strongest detector relative to its threshold) is calibrated the
same way. Image-forensic maps are thresholded at the 95th percentile of their strongest region on unaltered
validation cheques, and ELA/JPEG/noise/texture/frequency cues are only raised to "medium" when a second map
agrees: they are never sufficient on their own.

**Risk score.** `risk_assessor.risk_features` turns the checks into 13 numeric factors. A logistic regression
(standardised factors, L2, balanced classes) fitted on labelled validation cheques gives P(altered or forged);
the score is 100×P. Each factor's displayed contribution is weight × standardised value (log-odds relative to
an average validation cheque), and `models/risk_calibration.json` stores the weights and each factor's
univariate ROC-AUC as justification. Categories:

* **LOW RISK**: score below `t_low` (≤ 5% of altered validation cheques fall below it), no failed check and no
  signature mismatch;
* **HIGH RISK**: score at or above `t_high` (≥ 90% precision on validation) **and** at least two independent
  significant failed checks;
* **REQUIRES REVIEW**: everything else, including poor image quality, unreadable fields, inconclusive
  signatures and single strong indicators.

## 5. Privacy and retention

* The web app decodes uploads in memory and never writes them to disk; it binds to 127.0.0.1.
* Results, rendered images and PDFs are held in memory for `CHEQUE_RESULT_RETENTION_SECONDS` (default 30 min,
  at most 20 results), then purged; **Delete result now** removes one immediately.
* API responses are `Cache-Control: no-store`; a strict Content-Security-Policy blocks third-party scripts;
  all text from OCR is inserted into the page as text, never as HTML.
* The signature registry (`SignatureRegistry`) stores enrolled reference signatures under
  `data/signature_registry/<salted SHA-256 of the account number>`; enrolment is explicit and `delete()` removes
  an account's samples. Set `CHEQUE_REGISTRY_SALT` per deployment. Decide and document how long enrolled
  signatures are kept before using it with real customers.
* `data/` (datasets, synthetic caches, registry) is git-ignored.

## 6. Evaluation results

All numbers below are measured on the **synthetic test split** (templates, handwriting font families and
CEDAR signers never used for training or calibration), except signature verification, which uses CEDAR
test writers 46-55. They describe performance on synthetic data only. The JSON files in `models/metrics/`
hold the full results; `bash training/run_all.sh` regenerates them.

### Phase 1: preprocessing and field detection (`phase1_fields.json`, 120 test cheques)

| Metric | Classical / priors only | With learned models |
|---|---|---|
| Cheque found in a perspective desk photo (corner error ≤ 2% of width) | 79.2% | **85.8%** (median corner error 5 px) |
| Field localisation, mean IoU over 9 fields | 0.395 | **0.802** |
| Fields localised with IoU ≥ 0.5 | 0-100% depending on field | **100% for all 9 fields** |
| 0° / 180° orientation | not attempted | **100%** |

Per-field IoU with the learned models: amount in words 0.95, amount in figures 0.92, payee 0.85, signature 0.82,
date 0.81, cheque number 0.78, IFSC 0.73, account number 0.73, MICR 0.63. Boundary U-Net validation mask IoU
0.982; field U-Net validation mean pixel IoU 0.920.

### OCR (`text_recognizer.json`, `system_evaluation.json`)

* CRNN on synthetic text lines: test character error rate **8.5%**, exact-line accuracy 61.6% (validation fonts
  are harder: CER 16.0%).
* End to end, after field detection and crop cleaning (240 test cheques): amount in figures read exactly
  **75.4%**, amount in words parsed to the right value 52.5%, date 62.9%, IFSC (Tesseract) 67.9%, cheque number
  from the MICR line 20.4% (the synthetic MICR uses OCR-A as a stand-in for E-13B; a dedicated MICR reader was
  not built).

### Phase 2: signature verification (`signature_verification.json`, CEDAR writers 46-55, 4 references each)

| | ROC-AUC | EER | FAR / FRR at the validation EER threshold | Precision / recall (forgery = positive) |
|---|---|---|---|---|
| Skilled forgeries | 0.864 | 19.5% | 30.0% / 10.0% | 0.894 / 0.700 |
| Random forgeries (other writers) | 0.983 | 8.0% | 6.0% / 10.0% | 0.825 / 0.940 |

With the three-way decision (accept if FAR ≤ 5% on validation, reject if FRR ≤ 5%), 54% of skilled-forgery
comparisons are inconclusive; of the decided ones, 18.7% of forgeries are accepted and 5.1% of genuine
signatures rejected. On test cheques, no forged signature was accepted (48% flagged as mismatch, 52%
inconclusive) and 5.1% of genuine signatures were flagged; only 40.5% of genuine signatures were accepted
outright. This is a compact model trained on CPU; published CEDAR results are considerably better.

### Phases 3-5: alteration detection per field (`tamper_detection.json`, 1,200 test fields, 122 altered)

| Detector | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Classical forensics (stroke, ink, overlap, edges, paper) | 0.577 | 0.258 | 0.197 | 0.223 |
| Squeezed-character spacing | 0.480 | 0.070 | 0.041 | 0.052 |
| Tampering CNN | **0.804** | 0.444 | 0.262 | 0.330 |
| Autoencoder anomaly | 0.524 | 0.153 | 0.074 | 0.099 |
| Tampering U-Net | 0.745 | 0.276 | 0.443 | 0.340 |
| **Fused field score** (flags a field) | 0.748 | 0.256 | 0.352 | 0.297 |

* Thresholds target a 5% false-alarm rate on validation fields; on test fields the fused score raised false
  alarms on **11.6%** of unaltered fields (date field 26%), because validation and test use different
  handwriting styles.
* Localisation: U-Net pixel IoU with the altered pixels 0.23 on altered fields; a medium/high indicator box
  overlapped the altered area in 54% of altered fields.
* Field-level recall by alteration: overwritten digit 52%, erased-and-rewritten words 62%, copied glyph 50%,
  erased and rewritten digit 40%, pasted region 38%, prefix digit 31%, added stroke 29%, appended digit 22%,
  payee appended 14%, payee rewritten 10%.
* Trusted-original comparison (80 test cheques with an original): **97.1%** of pixel alterations found, 6.5%
  false alarms. Pixel-identical copied glyphs: 0 of 60 unaltered validation cheques flagged, 83% of copies found.

### Phase 6: whole-system risk assessment (`system_evaluation.json`, 240 test cheques, 117 altered or forged)

| Metric | Value |
|---|---|
| Risk score ROC-AUC | **0.714** |
| HIGH RISK precision / recall | **91.7%** / 18.8% |
| Altered or forged cheques rated LOW RISK | **3.4%** (4 of 117) |
| Genuine cheques rated LOW RISK | 7.3% (9 of 123); 91.1% REQUIRES REVIEW, 1.6% HIGH RISK |
| Amount check: FAIL / flagged (FAIL or WARN) when figures and words really differ | 30.0% / 88.6% |
| Amount check: false FAIL when they agree | 8.2% |
| Not rated LOW, by alteration | 80-100% for every alteration type (prefix digit 90%, added stroke 94%, payee appended 80%) |
| Processing time per cheque (CPU, 6 analyses in parallel) | median 8.9 s, 95th percentile 10.9 s |

Learned weights (log-odds per standard deviation): signature dissimilarity 0.58, pixel changes vs original
0.50, unreadable fields 0.46, duplicated glyphs 0.45, amount mismatch 0.31, amount-in-words alteration 0.19,
image forensics 0.17, payee alteration 0.15; the amount-figures and date-alteration weights are about zero and
`date_invalid` is slightly negative, because those features carried almost no signal on validation (univariate
AUCs 0.50-0.63, stored in `risk_calibration.json`).

**Reading the result.** The system is conservative: it rarely calls an altered cheque low risk and its high-risk
verdicts are usually right, but it sends most genuine cheques to manual review. In its current state it is a
triage and evidence tool, not an automatic approver.

## 7. Limitations and known false positives

* **Synthetic training data.** No labelled dataset of altered Indian cheques exists publicly, so alteration
  models were trained and calibrated on synthetic cheques written with handwriting fonts. Real handwriting, real
  inks, real erasure chemistry and real scanners will differ; expect lower accuracy and recalibrate on real
  labelled cheques before any operational use.
* **Out-of-domain images** (non-CTS layouts, printed amounts, photos of other documents) are recognised by the
  format check and capped at REQUIRES REVIEW. Before that gate was added, the genuine US sample cheque in this
  repository was rated HIGH RISK: the tampering CNN gave its printed amount a 100% alteration probability and the
  date U-Net fired on the US date line.
* **Field-level false positives** occur on 11.6% of unaltered test fields (26% of dates): unusual but natural
  handwriting, writing that touches box borders, and the date U-Net's low calibrated threshold (0.20) are the
  main causes. Each is shown with its explanation so a reviewer can dismiss it.
* **Weak spots:** a digit squeezed into the gap before the amount in the same hand is hard to see at field level
  (field recall 31%; the figures-vs-words check and the trusted original catch it when available; the
  corresponding unit test is marked as a known failure). Payee alterations are rarely detected at field level
  (10-14%).
* **Signature verification** was trained on CEDAR (55 Western-script writers). It has not been evaluated on
  Indian signers or Devanagari/regional-script signatures, and natural variation means a mismatch only calls
  for manual verification.
* **OCR:** the amount in words is read correctly about half the time; low-quality readings downgrade the amount
  check to WARN instead of FAIL. MICR reading is weak.
* **Image forensics** (ELA, JPEG, noise, texture, frequency) are weak cues on scanned paper with security
  printing; they never raise an indicator above "low" without a second, independent map agreeing.
* **Compute:** all models are small enough for CPU training (2-4 hours for the whole chain); larger models,
  more epochs and GPU training are the obvious next improvements.
