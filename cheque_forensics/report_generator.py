"""Images for the UI and the downloadable PDF analysis report.

The PDF is drawn with matplotlib's object API (Figure + PdfPages, no pyplot
state), so it is safe to generate from web-server threads.
"""
import base64
import datetime as dt
import io
import textwrap

import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Patch

from . import signature_verifier as sv
from .suspicious_region_detector import FIELD_NAMES, heatmap_image, overlay

# Status colours (always paired with an icon and a label) and the validated diverging pair
CATEGORY_STYLE = {"LOW RISK": ("#0ca30c", "✔"), "REQUIRES REVIEW": ("#fab219", "!"), "HIGH RISK": ("#d03b3b", "✖")}
STATUS_STYLE = {"PASS": ("#0ca30c", "✔"), "WARN": ("#fab219", "!"), "FAIL": ("#d03b3b", "✖"),
                "SKIP": ("#898781", "–"), "INFO": ("#2a78d6", "i")}
TOWARD_ALTERED, TOWARD_GENUINE = "#e34948", "#2a78d6"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e3e2dc"


def png_b64(img, max_width=1400):
    if img.shape[1] > max_width:
        img = cv2.resize(img, (max_width, int(img.shape[0] * max_width / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def render_images(result):
    """All visualisations of an AnalysisResult as BGR images."""
    pre = result.pre
    imgs = {}
    orig = pre.original.copy()
    scale = max(1, orig.shape[1] // 900)
    for name, f in result.fields.items():
        if f.original_polygon:
            pts = np.array(f.original_polygon, np.int32)
            cv2.polylines(orig, [pts], True, (40, 120, 230), 2 * scale)
            cv2.putText(orig, FIELD_NAMES.get(name, name), tuple(pts[0] + [3, 14 * scale]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * scale, (40, 120, 230), scale)
    quad = pre.detection.quad.astype(np.int32)
    cv2.polylines(orig, [quad], True, (60, 160, 60), 3 * scale)
    imgs["original_fields"] = orig
    imgs["normalized_fields"] = overlay(pre.color, None, (), result.fields)
    imgs["suspicious_regions"] = overlay(pre.color, result.heatmap, result.regions, result.fields)
    imgs["heatmap"] = heatmap_image(result.heatmap, pre.color)
    for k in ("difference", "ela", "noise", "texture", "jpeg_ghost", "edges"):
        if result.forensics and k in result.forensics.maps:
            imgs[f"forensics_{k}"] = heatmap_image(result.forensics.maps[k], pre.color)
    steps = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in (pre.gray, pre.enhanced, 255 - pre.binary)]
    imgs["preprocessing_steps"] = np.vstack([cv2.resize(s, (600, int(600 * s.shape[0] / s.shape[1]))) for s in steps])
    for name, fa in result.analyses.items():
        if fa.crop_box is None:
            continue
        crop = fa.crop_box.crop(pre.color)
        hm = cv2.resize(fa.heatmap, (crop.shape[1], crop.shape[0])) if fa.heatmap is not None else None
        view = heatmap_image(hm, crop) if hm is not None and hm.max() > 0.05 else crop.copy()
        for ind in fa.indicators:
            if ind.bbox is not None and ind.severity in ("medium", "high"):
                b = ind.bbox.shift(-fa.crop_box.x, -fa.crop_box.y)
                cv2.rectangle(view, (b.x, b.y), (b.x2, b.y2), (30, 30, 220) if ind.severity == "high" else (0, 140, 255), 2)
        imgs[f"field_{name}"] = np.vstack([crop, view])
    if result.signature_extraction is not None and result.signature_extraction.present:
        imgs["signature_extracted"] = cv2.cvtColor(result.signature_extraction.image, cv2.COLOR_GRAY2BGR)
        if result.references:
            imgs["signature_comparison"] = sv.comparison_image(result.signature_extraction.image, result.references,
                                                               result.signature)
    return imgs


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
A4 = (8.27, 11.69)


def _page(title, subtitle=None):
    fig = Figure(figsize=A4)
    fig.text(0.06, 0.965, title, fontsize=15, weight="bold", color=INK)
    if subtitle:
        fig.text(0.06, 0.945, subtitle, fontsize=8.5, color=INK2)
    return fig


def _wrap(text, width):
    return "\n".join(textwrap.wrap(str(text), width)) or ""


def _image(fig, img, rect, title=None):
    ax = fig.add_axes(rect)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, color=INK2, loc="left")
    return ax


def _contributions(fig, rect, factors):
    """Diverging horizontal bars: log-odds pushed toward 'altered' (red) or 'genuine' (blue)."""
    data = [f for f in factors if f.get("contribution") is not None][:10][::-1]
    ax = fig.add_axes(rect)
    if not data:
        ax.text(0, 0.5, "Risk model not calibrated: no factor weights to show.", fontsize=9, color=INK2)
        ax.set_axis_off()
        return
    vals = np.array([f["contribution"] for f in data])
    colors = [TOWARD_ALTERED if v > 0 else TOWARD_GENUINE for v in vals]
    ax.barh(range(len(data)), vals, color=colors, height=0.62, edgecolor="white", linewidth=2)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([_wrap(f["label"], 42) for f in data], fontsize=7.5, color=INK)
    ax.tick_params(axis="x", labelsize=7.5, colors=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    lim = max(abs(vals).max(), 0.5) * 1.35
    ax.set_xlim(-lim, lim)
    for i in np.argsort(-abs(vals))[:5]:   # label the largest contributions only
        v = vals[i]
        ax.text(v + (0.03 if v >= 0 else -0.03) * lim, i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=7, color=INK2)
    ax.set_xlabel("Contribution to risk score (log-odds vs. an average validation cheque)", fontsize=7.5, color=INK2)
    ax.legend(handles=[Patch(color=TOWARD_ALTERED, label="pushes toward altered"),
                       Patch(color=TOWARD_GENUINE, label="pushes toward genuine")],
              fontsize=7, frameon=False, loc="lower right")
    ax.set_title("What drove the score", fontsize=10, color=INK, loc="left")


def generate_pdf(result, images=None, out=None):
    """Write the multi-page PDF report. Returns bytes when `out` is None."""
    images = images or render_images(result)
    d = result.to_dict()
    risk = d["risk"]
    buf = out or io.BytesIO()
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with PdfPages(buf) as pdf:
        # page 1: verdict
        fig = _page("Cheque Forgery-Risk Analysis", f"Generated {stamp} · cheque_forensics v{d['info']['version']}")
        colour, icon = CATEGORY_STYLE[risk["category"]]
        fig.patches.append(FancyBboxPatch(
            (0.06, 0.835), 0.88, 0.085, boxstyle="round,pad=0.005,rounding_size=0.01", transform=fig.transFigure,
            facecolor=colour, alpha=0.14, edgecolor=colour, linewidth=1.5))
        fig.text(0.08, 0.885, f"{icon}  {risk['category']}", fontsize=20, weight="bold", color=colour)
        score = "uncalibrated" if risk["score"] is None else f"{risk['score']:.0f} / 100"
        th = risk.get("thresholds") or {}
        fig.text(0.08, 0.851, f"Risk score: {score}" + (f"   (low < {th['low']:.0f}, high ≥ {th['high']:.0f})" if th else ""),
                 fontsize=10, color=INK)
        y = 0.80
        fig.text(0.06, y, "Why", fontsize=11, weight="bold", color=INK)
        for r in risk["reasons"][:6]:
            y -= 0.022
            fig.text(0.075, y, _wrap("• " + r, 105), fontsize=8, color=INK, va="top")
            y -= 0.011 * r.count("\n") + 0.011 * (len(r) // 105)
        _contributions(fig, [0.36, 0.34, 0.56, 0.32], risk["factors"])
        fig.text(0.06, 0.08, _wrap(risk["disclaimer"], 120), fontsize=8, color=INK2, style="italic")
        pdf.savefig(fig)

        # page 2: checks
        fig = _page("Validation checks", "Each check is an indicator with its evidence, not a verdict.")
        y = 0.915
        for c in d["checks"]:
            col, ic = STATUS_STYLE.get(c["status"], (MUTED, "?"))
            text = _wrap(c["explanation"], 110)
            lines = text.count("\n") + 1
            if y - 0.018 * (lines + 1) < 0.04:
                pdf.savefig(fig)
                fig, y = _page("Validation checks (continued)"), 0.915
            fig.text(0.06, y, f"{ic} {c['status']}", fontsize=8.5, weight="bold", color=col)
            fig.text(0.16, y, f"{c['name']}  ·  severity {c['severity']}", fontsize=9, weight="bold", color=INK)
            fig.text(0.16, y - 0.016, text, fontsize=7.6, color=INK2, va="top")
            y -= 0.016 + 0.0125 * lines + 0.014
        pdf.savefig(fig)

        # page 3: images
        fig = _page("Cheque fields and suspicious regions",
                    "Field boxes in blue; suspicious regions numbered in red (high) / orange (medium); heatmap = fused evidence.")
        _image(fig, images["normalized_fields"], [0.06, 0.64, 0.88, 0.28], "Detected fields (normalised cheque)")
        _image(fig, images["suspicious_regions"], [0.06, 0.34, 0.88, 0.28], "Suspicious regions and evidence heatmap")
        y = 0.31
        for i, r in enumerate(d["regions"][:6], start=1):
            t = _wrap(f"{i}. [{r['severity']}] {FIELD_NAMES.get(r['field'], r['field'])}: {r['explanations'][0]}", 118)
            fig.text(0.06, y, t, fontsize=7.4, color=INK, va="top")
            y -= 0.012 * (t.count("\n") + 1) + 0.006
            if y < 0.05:
                break
        if not d["regions"]:
            fig.text(0.06, y, "No region reached medium severity.", fontsize=8, color=INK2, va="top")
        pdf.savefig(fig)

        # page 4: fields, signature, forensics
        fig = _page("Extracted values, signature and image forensics")
        rows = [(FIELD_NAMES.get(k, k), f["text"] or "—", "—" if f["text_confidence"] is None else f"{f['text_confidence']:.2f}",
                 f["ocr_engine"] or "—", f["source"]) for k, f in d["fields"].items() if k != "signature"]
        ax = fig.add_axes([0.06, 0.66, 0.88, 0.26])
        ax.set_axis_off()
        tbl = ax.table(cellText=[[r[0], r[1][:48], r[2], r[3], r[4]] for r in rows],
                       colLabels=["Field", "Read as", "OCR conf.", "Engine", "Located by"], loc="upper left",
                       cellLoc="left", colLoc="left", colWidths=[0.17, 0.43, 0.1, 0.12, 0.18])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        for (r_, c_), cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID)
            if r_ == 0:
                cell.set_text_props(weight="bold", color=INK)
        sig = d["signature"] or {}
        fig.text(0.06, 0.63, "Signature verification", fontsize=11, weight="bold", color=INK)
        fig.text(0.06, 0.61, _wrap(sig.get("explanation", "not performed"), 115), fontsize=7.8, color=INK2, va="top")
        if "signature_comparison" in images:
            _image(fig, images["signature_comparison"], [0.06, 0.43, 0.88, 0.14])
        fig.text(0.06, 0.40, "Image forensics", fontsize=11, weight="bold", color=INK)
        for i, k in enumerate([k for k in ("forensics_difference", "forensics_ela", "forensics_noise", "forensics_texture")
                               if k in images][:3]):
            _image(fig, images[k], [0.06 + i * 0.30, 0.25, 0.28, 0.13], k.replace("forensics_", "").upper())
        fig.text(0.06, 0.22, _wrap("Forensic maps highlight areas that differ from the rest of this image. Compression, "
                                   "scanning, folds and stamps also cause such differences; they support, but never "
                                   "replace, the field-level evidence.", 120), fontsize=7.5, color=INK2, va="top")
        pdf.savefig(fig)

        # page 5: processing information
        fig = _page("Processing information")
        info = d["info"]
        lines = [f"Device: {info['device']}    OCR engines: {', '.join(info['ocr_engines']) or 'none'}",
                 f"Input: {info['input_size'][0]}x{info['input_size'][1]} px, {info['file_format']}; "
                 f"cheque detection: {d['preprocessing']['detection']['method']} "
                 f"(confidence {d['preprocessing']['detection']['confidence']:.2f}); deskew {d['preprocessing']['deskew_degrees']}°",
                 f"Reference signatures: {info['n_reference_signatures']}; trusted original used: {info['trusted_original_used']}",
                 "Timings (s): " + ", ".join(f"{k} {v}" for k, v in d["timings_seconds"].items()),
                 "Models: " + ", ".join(f"{k}={'yes' if v else 'NO'}" if isinstance(v, bool) else f"{k}={v}"
                                        for k, v in info["models"].items()),
                 "Data handling: the uploaded images were processed in memory and are not stored by the application; "
                 "the analysis result is kept in memory only until it expires.",
                 "Limitations: models were trained and calibrated on synthetic Indian cheques and CEDAR signatures; "
                 "accuracy on real cheques has not been measured. See docs/CHEQUE_FORENSICS.md."]
        y = 0.91
        for line in lines:
            t = _wrap(line, 112)
            fig.text(0.06, y, t, fontsize=8, color=INK, va="top")
            y -= 0.0135 * (t.count("\n") + 1) + 0.012
        if "preprocessing_steps" in images:
            _image(fig, images["preprocessing_steps"], [0.06, 0.08, 0.5, 0.5], "Preprocessing: grayscale, CLAHE, adaptive threshold")
        pdf.savefig(fig)
    if out is None:
        return buf.getvalue()
    return out
