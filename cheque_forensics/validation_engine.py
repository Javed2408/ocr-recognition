"""Validation checks combining the outputs of all detection modules.

Each check has a status (PASS, WARN, FAIL, SKIP, INFO), a severity and a
plain-language explanation of what was found and why it matters. A FAIL means
"needs investigation", not "forged".
"""
import datetime as dt
from dataclasses import asdict, dataclass, field as dc_field

from . import amount_parser as ap
from .config import CHEQUE_VALIDITY_MONTHS, MIN_TEXT_CONFIDENCE
from .schema import SEVERITY_ORDER, jsonable

CRITICAL_FIELDS = ("amount_numeric", "amount_words", "date", "payee")
LABEL = {"amount_numeric": "amount in figures", "amount_words": "amount in words", "date": "date",
         "payee": "payee name", "signature": "signature", "cheque_number": "cheque number",
         "account_number": "account number", "ifsc": "IFSC code", "micr": "MICR line"}


@dataclass
class ValidationCheck:
    id: str
    name: str
    status: str                  # PASS, WARN, FAIL, SKIP or INFO
    severity: str                # info, low, medium, high
    explanation: str
    evidence: dict = dc_field(default_factory=dict)
    field: str = None

    def to_dict(self):
        d = asdict(self)
        d["evidence"] = jsonable(self.evidence)
        return d


def _rs(v):
    return ap.format_rupees(int(round(v))) + (f".{int(round(v * 100)) % 100:02d}" if abs(v - round(v)) > 1e-6 else "")


# 1 -------------------------------------------------------------------------
def check_amount_consistency(fields):
    num, words = fields.get("amount_numeric"), fields.get("amount_words")
    nv, wv = (num.value if num else None), (words.value if words else None)
    ev = {"figures_text": num.text if num else None, "words_text": words.text if words else None,
          "figures_value": nv, "words_value": wv,
          "figures_confidence": num.text_confidence if num else None,
          "words_confidence": words.text_confidence if words else None}
    if nv is None or wv is None:
        missing = " and ".join(n for n, v in (("figures", nv), ("words", wv)) if v is None)
        return ValidationCheck("amount_consistency", "Amount in figures vs words", "WARN", "medium",
                               f"Could not compare: the amount in {missing} could not be read or parsed. "
                               "Verify the amount manually.", ev, "amount_numeric")
    recognised, match = ap.words_reading_quality(words.text)
    clean = ap.figures_are_clean(num.text) and recognised == 1.0 and match >= 0.9 and \
        ap.amount_words_are_well_formed(words.text)
    confident = min(num.text_confidence or 0, words.text_confidence or 0) >= MIN_TEXT_CONFIDENCE and clean
    ev["reading_quality"] = {"figures_well_formed": ap.figures_are_clean(num.text),
                             "words_recognised": round(recognised, 3), "words_match": round(match, 3)}
    if abs(nv - wv) < 0.5:
        return ValidationCheck("amount_consistency", "Amount in figures vs words", "PASS", "info",
                               f"Amount in figures {_rs(nv)} matches the amount in words '{words.text}'.", ev)
    ratio = max(nv, wv) / max(min(nv, wv), 1)
    hint = ""
    if abs(ratio - round(ratio)) < 1e-6 and round(ratio) in (10, 100):
        hint = f" The two differ by a factor of {int(round(ratio))}, which is typical of an added or removed zero."
    elif str(int(min(nv, wv))) in str(int(max(nv, wv))):
        hint = " One value contains the other's digits, as when digits are added before or after an amount."
    if confident:
        return ValidationCheck("amount_consistency", "Amount in figures vs words", "FAIL", "high",
                               f"Amount inconsistency detected: figures read as {_rs(nv)} but words read as "
                               f"'{words.text}' ({_rs(wv)}).{hint}", ev, "amount_numeric")
    return ValidationCheck("amount_consistency", "Amount in figures vs words", "WARN", "medium",
                           f"Possible amount inconsistency: figures read as {_rs(nv)}, words as '{words.text}' "
                           f"({_rs(wv)}), but the reading is uncertain (low confidence or partly unrecognised words), so "
                           f"this may be an OCR error.{hint} "
                           "Verify manually.", ev, "amount_numeric")


# 2 -------------------------------------------------------------------------
def check_signature(sig):
    if sig is None or sig.outcome == "unavailable":
        return ValidationCheck("signature", "Signature verification", "SKIP", "low",
                               (sig.explanation if sig else "No registered signature was provided.") +
                               " The signature must be verified manually.", {}, "signature")
    ev = sig.to_dict()
    if sig.outcome == "likely_match":
        return ValidationCheck("signature", "Signature verification", "PASS", "info", sig.explanation, ev, "signature")
    if sig.outcome == "inconclusive":
        return ValidationCheck("signature", "Signature verification", "WARN", "medium",
                               "Signature requires manual verification: " + sig.explanation, ev, "signature")
    return ValidationCheck("signature", "Signature verification", "FAIL", "high",
                           "Signature requires manual verification: similarity is below the calibrated "
                           "verification threshold. " + sig.explanation +
                           " Natural variation in a person's signature can also cause this.", ev, "signature")


def _alteration_check(cid, name, analyses, keys):
    found = [(k, i) for k in keys if k in analyses for i in analyses[k].indicators
             if i.severity in ("medium", "high")]
    flagged = [k for k in keys if k in analyses and analyses[k].suspicious]
    ev = {k: analyses[k].scores for k in keys if k in analyses}
    if not any(k in analyses for k in keys):
        return ValidationCheck(cid, name, "SKIP", "low", "The field could not be analysed.", ev)
    if not flagged and not found:
        return ValidationCheck(cid, name, "PASS", "info", "No signs of overwriting, erasure, inserted characters "
                               "or inconsistent ink were found.", ev, keys[0])
    top = sorted(found, key=lambda ki: -SEVERITY_ORDER[ki[1].severity])[:3]
    text = " ".join(f"[{LABEL[k]}] {i.explanation}" for k, i in top)
    if flagged:
        sev = "high" if any(i.severity == "high" for _, i in found) else "medium"
        return ValidationCheck(cid, name, "FAIL", sev,
                               f"Possible alteration in the {', '.join(LABEL[k] for k in flagged)} "
                               f"(evidence above the calibrated threshold). {text}",
                               dict(ev, indicators=[i.to_dict() for _, i in top]), flagged[0])
    return ValidationCheck(cid, name, "WARN", "low", "Individual indicators were raised, but the combined "
                           f"evidence stays below the calibrated threshold. {text}",
                           dict(ev, indicators=[i.to_dict() for _, i in top]), top[0][0])


# 3 -------------------------------------------------------------------------
def check_amount_overwriting(analyses):
    return _alteration_check("amount_overwriting", "Overwriting / alteration of the amount", analyses,
                             ["amount_numeric", "amount_words"])


# 4 -------------------------------------------------------------------------
def check_date(fields, analyses, today=None):
    today = today or dt.date.today()
    f = fields.get("date")
    alt = _alteration_check("date_anomalies", "Date field anomalies", analyses, ["date"])
    problems = []
    if f is None or not f.text:
        problems.append(("WARN", "medium", "The date could not be read."))
    else:
        date, err = ap.parse_cheque_date(f.text)
        if date is None:
            problems.append(("WARN" if (f.text_confidence or 0) < MIN_TEXT_CONFIDENCE else "FAIL", "medium",
                             f"The date '{f.text}' is not a valid date: {err}."))
        else:
            state, why = ap.date_validity(date, today, CHEQUE_VALIDITY_MONTHS)
            if state != "valid":
                problems.append(("FAIL", "medium", f"The cheque is {state}: {why}."))
    if not problems:
        if alt.status == "PASS" and f is not None and f.value is not None:
            alt.explanation = f"Date {f.value:%d-%m-%Y} is valid and shows no signs of alteration."
        return alt
    status = max([alt.status] + [p[0] for p in problems], key=["PASS", "SKIP", "WARN", "FAIL"].index)
    sev = max([alt.severity] + [p[1] for p in problems], key=SEVERITY_ORDER.get)
    text = " ".join(p[2] for p in problems)
    if alt.status in ("FAIL", "WARN"):
        text += " " + alt.explanation
    return ValidationCheck("date_anomalies", "Date field anomalies", status, sev, text,
                           dict(alt.evidence, date_text=f.text if f else None), "date")


# 5 -------------------------------------------------------------------------
def check_payee(fields, analyses):
    c = _alteration_check("payee_anomalies", "Payee field anomalies", analyses, ["payee"])
    f = fields.get("payee")
    if f is not None and "field appears empty" in f.notes:
        return ValidationCheck("payee_anomalies", "Payee field anomalies", "WARN", "medium",
                               "The payee line appears empty; a blank payee lets anyone fill in a name.", {}, "payee")
    return c


# 6 -------------------------------------------------------------------------
def check_image_manipulation(forensics):
    if forensics is None:
        return ValidationCheck("image_manipulation", "Image manipulation indicators", "SKIP", "low",
                               "Document forensics was not run.")
    strong = [i for i in forensics.indicators if i.severity in ("medium", "high")]
    reg = [i for i in strong if i.name == "difference_from_original"]
    dup = [i for i in strong if i.name == "duplicated_glyph"]
    ev = {"jpeg": forensics.jpeg, "registration": forensics.registration, "stats": forensics.stats,
          "indicators": [i.to_dict() for i in strong[:6]]}
    if reg:
        fields = sorted({i.field for i in reg if i.field})
        return ValidationCheck("image_manipulation", "Image manipulation indicators", "FAIL", "high",
                               f"Compared with the trusted original, {len(reg)} region(s) differ"
                               f"{' in: ' + ', '.join(LABEL.get(f, f) for f in fields) if fields else ''}. "
                               "These are actual pixel changes between the two images.", ev)
    if dup:
        return ValidationCheck("image_manipulation", "Image manipulation indicators", "FAIL", "high",
                               dup[0].explanation, ev, dup[0].field)
    if strong:
        return ValidationCheck("image_manipulation", "Image manipulation indicators", "WARN", "medium",
                               f"{len(strong)} corroborated image-forensic anomal{'y' if len(strong) == 1 else 'ies'}. "
                               + strong[0].explanation, ev, strong[0].field)
    note = "" if forensics.registration else " No trusted original was provided for pixel-level comparison."
    return ValidationCheck("image_manipulation", "Image manipulation indicators", "PASS", "info",
                           "No corroborated signs of digital editing (pasted regions, duplicated characters, "
                           "inconsistent compression or noise)." + note, ev)


# 7 -------------------------------------------------------------------------
def check_missing_fields(fields, signature_present):
    missing = []
    for name in CRITICAL_FIELDS:
        f = fields.get(name)
        if f is None or f.bbox is None or "field appears empty" in f.notes or not f.text:
            missing.append(name)
        elif f.value is None and name != "payee":
            missing.append(name)
    if not signature_present:
        missing.append("signature")
    ev = {"missing_or_unreadable": missing}
    if not missing:
        return ValidationCheck("missing_fields", "Missing or unreadable fields", "PASS", "info",
                               "All critical fields (amount in figures and words, date, payee, signature) were found "
                               "and read.", ev)
    sev = "high" if len(missing) >= 3 else "medium"
    return ValidationCheck("missing_fields", "Missing or unreadable fields", "WARN" if len(missing) < 3 else "FAIL",
                           sev, "Missing or unreadable: " + ", ".join(LABEL[m] for m in missing) +
                           ". These fields could not be checked automatically.", ev, missing[0])


# 8 -------------------------------------------------------------------------
def check_ocr_confidence(fields, engines):
    confs = {n: fields[n].text_confidence for n in CRITICAL_FIELDS + ("ifsc", "account_number", "micr")
             if n in fields and fields[n].text_confidence is not None and fields[n].text}
    if not confs:
        return ValidationCheck("ocr_confidence", "OCR confidence and uncertainty", "WARN", "medium",
                               "No text could be read" + ("" if engines else " (no OCR engine available)") + ".",
                               {"engines": engines})
    low = {n: c for n, c in confs.items() if c < MIN_TEXT_CONFIDENCE}
    mean = sum(confs.values()) / len(confs)
    ev = {"engines": engines, "confidence": confs, "threshold": MIN_TEXT_CONFIDENCE}
    caveat = (" The built-in recogniser was trained on synthetic handwriting only, so readings of real handwriting "
              "need manual confirmation." if "crnn" in engines else "")
    if not low:
        return ValidationCheck("ocr_confidence", "OCR confidence and uncertainty", "PASS", "info",
                               f"Mean OCR confidence {mean:.2f}; all fields above {MIN_TEXT_CONFIDENCE:.2f}." + caveat, ev)
    crit = [n for n in low if n in CRITICAL_FIELDS]
    return ValidationCheck("ocr_confidence", "OCR confidence and uncertainty", "WARN", "medium" if crit else "low",
                           f"Low OCR confidence for: {', '.join(LABEL[n] + f' ({c:.2f})' for n, c in low.items())}. "
                           "Values read from these fields are uncertain." + caveat, ev, crit[0] if crit else None)


def check_image_quality(quality, detection=None):
    bad = [c for c in quality if c.status in ("FAIL", "WARN")]
    ev = {c.name: f"{c.status}: {c.detail}" for c in quality}
    if detection is not None:
        ev["cheque_detection"] = detection.to_dict()
    if any(c.status == "FAIL" for c in bad):
        return ValidationCheck("image_quality", "Image quality", "FAIL", "high",
                               "Insufficient image quality: " + "; ".join(f"{c.name} {c.detail}" for c in bad) +
                               ". Results are unreliable; rescan the cheque.", ev)
    if bad:
        return ValidationCheck("image_quality", "Image quality", "WARN", "medium",
                               "Reduced image quality: " + "; ".join(f"{c.name} {c.detail}" for c in bad) +
                               ". Subtle alterations may be missed and false alarms are more likely.", ev)
    return ValidationCheck("image_quality", "Image quality", "PASS", "info", "Image quality is adequate.", ev)


def check_cheque_format(fmt):
    """Is this an Indian CTS-2010 cheque, the only format the detectors were trained on?"""
    from .config import CTS_FORMAT_ASPECT, CTS_FORMAT_MIN_LAYOUT
    if not fmt:
        return ValidationCheck("cheque_format", "Cheque format (CTS-2010)", "SKIP", "low",
                               "Format not verified (field segmentation model unavailable).")
    lo, hi = CTS_FORMAT_ASPECT
    ok_aspect = lo <= fmt["aspect"] <= hi
    ok_layout = fmt["layout_score"] is None or fmt["layout_score"] >= CTS_FORMAT_MIN_LAYOUT
    if ok_aspect and ok_layout:
        return ValidationCheck("cheque_format", "Cheque format (CTS-2010)", "PASS", "info",
                               f"Proportions ({fmt['aspect']:.2f}:1) and field layout match an Indian CTS-2010 cheque.", fmt)
    why = []
    if not ok_aspect:
        why.append(f"proportions {fmt['aspect']:.2f}:1 (CTS-2010 leaves are about 2.16:1)")
    if not ok_layout:
        why.append(f"field-layout agreement {fmt['layout_score']:.2f} (at least {CTS_FORMAT_MIN_LAYOUT:.2f} expected)")
    return ValidationCheck("cheque_format", "Cheque format (CTS-2010)", "WARN", "medium",
                           "The image does not look like an Indian CTS-2010 cheque: " + " and ".join(why) + ". The "
                           "detectors were trained only on CTS-2010 cheques, so their alteration and forgery indicators "
                           "are unreliable for this image; it cannot be rated low or high risk automatically.", fmt)


def cross_field_ink_note(analyses):
    """Different pens between fields are common (e.g. payee filled by a clerk): information only."""
    return ValidationCheck("cross_field_ink", "Ink and handwriting across fields", "INFO", "info",
                           "Fields are compared only within themselves; different ink colours or handwriting "
                           "between fields are common and are not treated as fraud indicators.")


def validate(fields, analyses, signature, forensics, quality, engines, signature_present, detection=None, today=None,
             cheque_format=None):
    checks = [
        check_amount_consistency(fields),
        check_signature(signature),
        check_amount_overwriting(analyses),
        check_date(fields, analyses, today),
        check_payee(fields, analyses),
        check_image_manipulation(forensics),
        check_missing_fields(fields, signature_present),
        check_ocr_confidence(fields, engines),
        check_image_quality(quality, detection),
        check_cheque_format(cheque_format),
        cross_field_ink_note(analyses),
    ]
    return checks
