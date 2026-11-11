"""Command line: python -m cheque_forensics CHEQUE [--ref SIG ...] [--original IMG] [--json OUT] [--pdf OUT]"""
import argparse
import json
import sys

import cv2


def main(argv=None):
    ap = argparse.ArgumentParser(description="Forgery-risk analysis of an Indian bank cheque image.")
    ap.add_argument("cheque", help="cheque image (PNG/JPEG/TIFF)")
    ap.add_argument("--ref", action="append", default=[], help="registered genuine signature image (repeatable)")
    ap.add_argument("--original", help="trusted image of the same cheque before it was handed over")
    ap.add_argument("--json", help="write the full result as JSON")
    ap.add_argument("--pdf", help="write the PDF report")
    ap.add_argument("--tesseract-cmd", help="path to the tesseract executable if it is not on PATH")
    args = ap.parse_args(argv)
    from .pipeline import ChequeAnalyzer
    try:
        with open(args.cheque, "rb") as f:
            data = f.read()
        refs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in args.ref]
        if any(r is None for r in refs):
            raise ValueError("a reference signature could not be read")
        original = cv2.imread(args.original) if args.original else None
        result = ChequeAnalyzer(tesseract_cmd=args.tesseract_cmd).analyze(data, references=refs, original=original)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    risk = result.risk
    print(f"\n{risk.category}" + (f"  (score {risk.score:.0f}/100)" if risk.score is not None else ""))
    for r in risk.reasons:
        print(f"  - {r}")
    print("\nChecks:")
    for c in result.checks:
        print(f"  [{c.status}] {c.name}: {c.explanation}")
    print(f"\n{risk.disclaimer}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    if args.pdf:
        from .report_generator import generate_pdf
        with open(args.pdf, "wb") as f:
            f.write(generate_pdf(result))
    return 0 if risk.category == "LOW RISK" else 1


if __name__ == "__main__":
    sys.exit(main())
