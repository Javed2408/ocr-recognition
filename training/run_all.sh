#!/bin/bash
# Full training + calibration + evaluation chain (roughly 2-4 hours on a 20-thread laptop CPU).
# Order matters: tamper calibration must exist before risk calibration, which adds document-map thresholds.
set -e
cd "$(dirname "$0")/.."
LOG=models/metrics
if [ "${1:-}" = "--after-crnn" ]; then
  while ! grep -q "^val " $LOG/train_text_recognizer.log 2>/dev/null; do sleep 30; done
else
  python -m training.train_segmenters --n-train 1600 --n-val 240 --epochs 8 > $LOG/train_segmenters.log 2>&1
  python -m training.train_text_recognizer > $LOG/train_text_recognizer.log 2>&1
fi
python -m training.train_siamese > $LOG/train_siamese.log 2>&1
python -m training.train_tamper_models --n-train 1500 --n-val 300 --n-test 300 > $LOG/train_tamper_models.log 2>&1
python -m training.calibrate_risk --n-val 240 --n-test 240 > $LOG/calibrate_risk.log 2>&1
python -m training.evaluate_fields --n 120 > $LOG/evaluate_fields.log 2>&1
echo "ALL DONE"
