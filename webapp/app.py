"""Web interface: the original digit OCR (aiproject.py) and the cheque forgery-risk analysis.

    python -m webapp.app            # http://127.0.0.1:5000

Privacy: uploaded images are decoded in memory and never written to disk.
Results (including rendered images and the PDF) are kept in memory for
RESULT_RETENTION_SECONDS so the report can be downloaded, then purged; the
user can delete a result immediately. The server binds to localhost.
"""
import os
import secrets
import sys
import threading
import time
from collections import OrderedDict

from flask import Flask, abort, jsonify, render_template, request, send_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import io  # noqa: E402

import cv2  # noqa: E402

import aiproject  # noqa: E402
from cheque_forensics.config import (ALLOWED_IMAGE_EXTENSIONS, MAX_CACHED_RESULTS, MAX_UPLOAD_BYTES,  # noqa: E402
                                     RESULT_RETENTION_SECONDS)
from cheque_forensics.pipeline import decode_image, model_status  # noqa: E402

MAX_REFERENCES = 8


class ResultCache:
    """In-memory results with expiry; nothing is persisted."""

    def __init__(self, ttl=RESULT_RETENTION_SECONDS, max_items=MAX_CACHED_RESULTS):
        self.ttl, self.max_items = ttl, max_items
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def _purge(self):
        now = time.time()
        for k in [k for k, (t, _) in self._items.items() if now - t >= self.ttl]:
            del self._items[k]
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def put(self, value):
        key = secrets.token_urlsafe(16)
        with self._lock:
            self._purge()
            self._items[key] = (time.time(), value)
        return key

    def get(self, key):
        with self._lock:
            self._purge()
            item = self._items.get(key)
            return item[1] if item else None

    def delete(self, key):
        with self._lock:
            return self._items.pop(key, None) is not None


def create_app(analyzer=None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES * (MAX_REFERENCES + 2)
    cache = ResultCache()
    state = {"analyzer": analyzer, "digit_model": None}
    model_lock = threading.Lock()     # models (and Grad-CAM gradients) are not thread-safe

    def get_analyzer():
        if state["analyzer"] is None:
            from cheque_forensics.pipeline import ChequeAnalyzer
            state["analyzer"] = ChequeAnalyzer()
        return state["analyzer"]

    def read_upload(storage, what):
        if storage is None or not storage.filename:
            return None, None
        ext = os.path.splitext(storage.filename)[1].lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(f"{what}: unsupported file type '{ext or 'none'}' "
                             f"(allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))})")
        data = storage.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"{what}: file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        try:
            return decode_image(data), data
        except ValueError as exc:
            raise ValueError(f"{what}: {exc}") from exc

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data: blob:; "
                                                   "style-src 'self'; script-src 'self'")
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Upload too large."), 413

    @app.get("/")
    def index():
        return render_template("index.html", retention_minutes=RESULT_RETENTION_SECONDS // 60,
                               max_mb=MAX_UPLOAD_BYTES // (1024 * 1024))

    @app.get("/api/status")
    def status():
        return jsonify(models=model_status(), tesseract=aiproject.setup_tesseract(),
                       retention_seconds=RESULT_RETENTION_SECONDS, storage="in-memory only")

    @app.post("/api/cheque/analyze")
    def analyze():
        try:
            cheque, cheque_bytes = read_upload(request.files.get("cheque"), "Cheque image")
            if cheque is None:
                raise ValueError("Please upload a cheque image.")
            refs = []
            for i, f in enumerate(request.files.getlist("references")[:MAX_REFERENCES]):
                img, _ = read_upload(f, f"Reference signature {i + 1}")
                if img is not None:
                    refs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            original, _ = read_upload(request.files.get("original"), "Trusted original")
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        from cheque_forensics.report_generator import png_b64, render_images
        with model_lock:
            try:
                result = get_analyzer().analyze(cheque, cheque_bytes, refs, original)
                images = render_images(result)
            except Exception as exc:   # report the failure, never a stack trace with image data
                app.logger.exception("analysis failed")
                return jsonify(error=f"Analysis failed: {type(exc).__name__}: {str(exc)[:200]}"), 500
        key = cache.put({"result": result, "images": images, "pdf": None})
        payload = result.to_dict()
        payload["id"] = key
        payload["images"] = {k: png_b64(v) for k, v in images.items()}
        payload["retention_seconds"] = RESULT_RETENTION_SECONDS
        return jsonify(payload)

    @app.get("/api/cheque/<key>/report.pdf")
    def report(key):
        item = cache.get(key)
        if item is None:
            abort(404, "Result expired or deleted.")
        if item["pdf"] is None:
            from cheque_forensics.report_generator import generate_pdf
            with model_lock:
                item["pdf"] = generate_pdf(item["result"], item["images"])
        return send_file(io.BytesIO(item["pdf"]), mimetype="application/pdf", as_attachment=True,
                         download_name="cheque_analysis_report.pdf")

    @app.delete("/api/cheque/<key>")
    def delete(key):
        return jsonify(deleted=cache.delete(key))

    @app.post("/api/ocr")
    def digit_ocr():
        """The original pipeline from aiproject.py: input checks, Tesseract OCR, MNIST CNN digits."""
        try:
            image, _ = read_upload(request.files.get("image"), "Image")
            if image is None:
                raise ValueError("Please upload an image.")
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        from cheque_forensics.report_generator import png_b64
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary, inverted = aiproject.binarize(gray)
        input_checks = aiproject.validate_input_image(image, gray, binary, inverted)
        out = {"input_checks": [c.__dict__ for c in input_checks]}
        if aiproject.has_failures(input_checks):
            out["error"] = "Input image failed quality checks."
            return jsonify(out), 422
        with model_lock:
            if state["digit_model"] is None:
                state["digit_model"] = aiproject.load_model(aiproject.DEFAULT_MODEL)
            digits, candidates = aiproject.recognize_digits(binary, state["digit_model"])
        ocr_text, ocr_conf = None, 0.0
        if aiproject.setup_tesseract():
            ocr_text, ocr_conf = aiproject.run_ocr(binary)
        checks = aiproject.validate_results(ocr_text, ocr_conf, digits, candidates)
        out.update(digits=[d for _, d, _ in digits], ocr_text=ocr_text, ocr_confidence=ocr_conf,
                   output_checks=[c.__dict__ for c in checks],
                   annotated=png_b64(aiproject.annotate(image, digits)))
        return jsonify(out)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
