"use strict";
// Cheque forgery-risk UI. All text from the server (including OCR output) is inserted with textContent.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const svgNS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

const FIELD_NAMES = {
  amount_numeric: "Amount (figures)", amount_words: "Amount (words)", payee: "Payee", date: "Date",
  signature: "Signature", cheque_number: "Cheque no.", account_number: "Account no.", ifsc: "IFSC", micr: "MICR line",
};
const CATEGORY = {
  "LOW RISK": { cls: "low", icon: "✔", tone: "good" },
  "REQUIRES REVIEW": { cls: "review", icon: "!", tone: "warning" },
  "HIGH RISK": { cls: "high", icon: "✖", tone: "critical" },
};
const STATUS = {
  PASS: ["good", "✔"], WARN: ["warning", "!"], FAIL: ["critical", "✖"], SKIP: ["muted", "–"], INFO: ["info", "i"],
  likely_match: ["good", "✔"], inconclusive: ["warning", "!"], suspicious_mismatch: ["critical", "✖"], unavailable: ["muted", "–"],
};
const pill = (key, label) => {
  const [tone, icon] = STATUS[key] || ["muted", "?"];
  return h("span", { class: "pill" }, h("span", { class: `icon ${tone}`, "aria-hidden": "true", text: icon }), label || key);
};
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));

// ---------------------------------------------------------------- tabs
$$(".tab").forEach((b) => b.addEventListener("click", () => {
  $$(".tab").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-selected", x === b); });
  $$(".tabpage").forEach((p) => { p.hidden = p.id !== `tab-${b.dataset.tab}`; });
}));
$$(".subtab").forEach((b) => b.addEventListener("click", () => showPanel(b.dataset.panel)));
function showPanel(name) {
  $$(".subtab").forEach((x) => x.classList.toggle("active", x.dataset.panel === name));
  $$(".panel").forEach((p) => { p.hidden = p.dataset.panel !== name; });
}

// ---------------------------------------------------------------- uploads
const files = { cheque: null, refs: [], original: null, ocr: null };

function wireDrop(dropId, inputId, onFiles) {
  const drop = $(dropId), input = $(inputId);
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => onFiles([...input.files]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragover"); }));
  drop.addEventListener("drop", (e) => onFiles([...e.dataTransfer.files].filter((f) => f.type.startsWith("image/") || /\.(tif|tiff)$/i.test(f.name))));
}
function preview(img, file) {
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  if (!file) { img.hidden = true; return; }
  img.dataset.url = URL.createObjectURL(file);
  img.src = img.dataset.url;
  img.hidden = false;
}
wireDrop("#drop-cheque", "#in-cheque", (fs) => { files.cheque = fs[0] || null; preview($("#prev-cheque"), files.cheque); refreshButtons(); });
wireDrop("#drop-original", "#in-original", (fs) => { files.original = fs[0] || null; preview($("#prev-original"), files.original); });
wireDrop("#drop-refs", "#in-refs", (fs) => {
  files.refs = fs.slice(0, 8);
  const box = $("#prev-refs");
  box.replaceChildren(...files.refs.map((f) => { const i = h("img", { alt: "reference signature" }); preview(i, f); return i; }));
});
wireDrop("#drop-ocr", "#in-ocr", (fs) => { files.ocr = fs[0] || null; preview($("#prev-ocr"), files.ocr); $("#btn-ocr").disabled = !files.ocr; });
function refreshButtons() { $("#btn-analyze").disabled = !files.cheque; }

$("#btn-clear").addEventListener("click", () => {
  Object.assign(files, { cheque: null, refs: [], original: null });
  ["#in-cheque", "#in-refs", "#in-original"].forEach((s) => { $(s).value = ""; });
  preview($("#prev-cheque"), null); preview($("#prev-original"), null); $("#prev-refs").replaceChildren();
  $("#results").hidden = true; setStatus(""); refreshButtons();
});

function setStatus(text, error = false, busy = false, id = "#status") {
  const el = $(id);
  el.classList.toggle("error", error);
  el.replaceChildren(...(busy ? [h("span", { class: "spinner", "aria-hidden": "true" })] : []), text);
}

// ---------------------------------------------------------------- analyse
let current = null;
$("#btn-analyze").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("cheque", files.cheque);
  files.refs.forEach((f) => fd.append("references", f));
  if (files.original) fd.append("original", files.original);
  $("#btn-analyze").disabled = true;
  setStatus("Analysing… the first run loads the models and can take a minute.", false, true);
  try {
    const res = await fetch("/api/cheque/analyze", { method: "POST", body: fd });
    const data = await res.json().catch(() => ({ error: `Server error ${res.status}` }));
    if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
    current = data;
    render(data);
    setStatus(`Done in ${fmt(data.timings_seconds.total, 1)} s.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    refreshButtons();
  }
});

function render(d) {
  renderVerdict(d);
  renderOverview(d);
  renderFields(d);
  renderSignature(d);
  renderTampering(d);
  renderForensics(d);
  renderChecks(d);
  $("#results").hidden = false;
  showPanel("overview");
  $("#verdict").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderVerdict(d) {
  const r = d.risk, c = CATEGORY[r.category];
  const v = $("#verdict");
  v.className = `card verdict ${c.cls}`;
  const score = r.score === null ? h("div", { class: "score", text: "Risk score not available (risk model not calibrated)" })
    : h("div", { class: "score" }, "Risk score ", h("strong", { text: `${Math.round(r.score)} / 100` }),
      r.thresholds && r.thresholds.low !== undefined ? `  ·  low < ${Math.round(r.thresholds.low)}, high ≥ ${Math.round(r.thresholds.high)}` : "");
  const pdf = h("a", { class: "btn primary", href: `/api/cheque/${encodeURIComponent(d.id)}/report.pdf`, text: "Download PDF report" });
  const del = h("button", { class: "btn ghost", text: "Delete result now", onclick: async () => {
    await fetch(`/api/cheque/${encodeURIComponent(d.id)}`, { method: "DELETE" });
    pdf.removeAttribute("href"); pdf.classList.add("disabled"); pdf.textContent = "Result deleted";
    del.disabled = true;
  } });
  v.replaceChildren(
    h("div", {}, h("div", { class: "badge-big" }, h("span", { class: `icon ${c.tone}`, "aria-hidden": "true", text: c.icon }), r.category), score),
    h("ul", { class: "reasons" }, r.reasons.map((x) => h("li", { text: x }))),
    h("div", { class: "verdict-actions" }, pdf, del),
    h("p", { class: "disclaimer", text: r.disclaimer }),
  );
}

function figure(src, caption) {
  if (!src) return null;
  return h("figure", { class: "figure" }, h("img", { src, alt: caption }), h("figcaption", { text: caption }));
}

function regionsList(regions) {
  if (!regions.length) return h("p", { class: "note", text: "No region reached medium severity." });
  return h("ol", { class: "regions" }, regions.slice(0, 8).map((r, i) => h("li", {},
    h("span", { class: `num ${r.severity}`, text: `${i + 1}` }),
    h("div", {}, h("strong", { text: `${FIELD_NAMES[r.field] || r.field} · ${r.severity}` }),
      r.explanations.slice(0, 2).map((e) => h("p", { class: "expl", text: e })),
      h("p", { class: "note", text: `Evidence: ${r.sources.join(", ")}` })))));
}

function renderOverview(d) {
  const p = $('.panel[data-panel="overview"]');
  p.replaceChildren(
    h("h2", { text: "Suspicious regions" }),
    figure(d.images.suspicious_regions, "Numbered boxes: suspicious regions (red = high, orange = medium). Colour wash: fused evidence heatmap."),
    regionsList(d.regions),
    h("h3", { text: "What drove the risk score" }),
    contributionChart(d.risk.factors),
  );
}

// Diverging horizontal bars: positive log-odds push toward "altered" (red), negative toward "genuine" (blue).
function contributionChart(factors) {
  const data = factors.filter((f) => f.contribution !== null).slice(0, 10);
  if (!data.length) {
    return h("div", {}, h("p", { class: "note", text: "Risk model not calibrated, so no learned weights to show. Evidence values:" }),
      factorTable(factors));
  }
  const W = 760, rowH = 26, left = 300, right = 60, top = 8, H = top + data.length * rowH + 28;
  const max = Math.max(0.5, ...data.map((f) => Math.abs(f.contribution))) * 1.15;
  const x = (v) => left + ((v + max) / (2 * max)) * (W - left - right);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Contribution of each factor to the risk score" });
  [-max / 2, max / 2].forEach((g) => svg.append(s("line", { x1: x(g), x2: x(g), y1: top, y2: H - 22, class: "grid" })));
  svg.append(s("line", { x1: x(0), x2: x(0), y1: top, y2: H - 22, class: "zero" }));
  const tip = $("#tooltip");
  const ranked = [...data].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 4);
  data.forEach((f, i) => {
    const y = top + i * rowH, v = f.contribution, x0 = x(Math.min(0, v)), w = Math.max(2, Math.abs(x(v) - x(0)));
    const label = s("text", { x: left - 10, y: y + rowH / 2 + 4, "text-anchor": "end" });
    label.textContent = f.label.length > 44 ? `${f.label.slice(0, 43)}…` : f.label;
    svg.append(label);
    svg.append(s("rect", { x: x0, y: y + 5, width: w, height: rowH - 10, rx: 4, class: v >= 0 ? "bar-alt" : "bar-gen" }));
    if (ranked.includes(f)) {
      const t = s("text", { x: v >= 0 ? x0 + w + 6 : x0 - 6, y: y + rowH / 2 + 4, "text-anchor": v >= 0 ? "start" : "end" });
      t.textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
      svg.append(t);
    }
    const hit = s("rect", { x: 0, y, width: W, height: rowH, class: "hit" });
    hit.addEventListener("mousemove", (e) => {
      tip.replaceChildren(h("strong", { text: f.label }), h("br"),
        `Evidence value ${fmt(f.value)} × learned weight ${fmt(f.weight)}`, h("br"),
        `Contribution ${v >= 0 ? "+" : ""}${v.toFixed(3)} log-odds (${v >= 0 ? "toward altered" : "toward genuine"})`);
      tip.hidden = false;
      tip.style.left = `${Math.min(e.clientX + 14, window.innerWidth - 330)}px`;
      tip.style.top = `${e.clientY + 14}px`;
    });
    hit.addEventListener("mouseleave", () => { tip.hidden = true; });
    svg.append(hit);
  });
  const axis = s("text", { x: x(0), y: H - 6, "text-anchor": "middle" });
  axis.textContent = "log-odds contribution vs. an average validation cheque";
  svg.append(axis);
  return h("div", { class: "chart-wrap" },
    h("div", { class: "legend" }, h("span", {}, h("i", { class: "alt" }), "pushes toward altered"),
      h("span", {}, h("i", { class: "gen" }), "pushes toward genuine")),
    svg, h("details", {}, h("summary", { text: "Table view" }), factorTable(factors)));
}

function factorTable(factors) {
  return h("div", { class: "table-scroll" }, h("table", {},
    h("thead", {}, h("tr", {}, ["Factor", "Evidence value", "Learned weight", "Contribution"].map((t) => h("th", { text: t })))),
    h("tbody", {}, factors.map((f) => h("tr", {}, h("td", { text: f.label }), h("td", { class: "mono", text: fmt(f.value) }),
      h("td", { class: "mono", text: fmt(f.weight) }), h("td", { class: "mono", text: fmt(f.contribution) }))))));
}

function renderFields(d) {
  const p = $('.panel[data-panel="fields"]');
  const rows = Object.values(d.fields).map((f) => {
    let value = f.value;
    if (value && typeof value === "object") value = Object.entries(value).filter(([, v]) => v).map(([k, v]) => `${k}: ${v}`).join(", ");
    return h("tr", {},
      h("td", { text: FIELD_NAMES[f.name] || f.name }),
      h("td", { class: "mono", text: f.text || "—" }),
      h("td", { class: "mono", text: value === null || value === undefined ? "—" : String(value) }),
      h("td", { class: "mono", text: fmt(f.text_confidence, 2) }),
      h("td", { text: f.ocr_engine || "—" }),
      h("td", { text: `${f.source} (${fmt(f.confidence, 2)})` }),
      h("td", { class: "mono", text: f.bbox ? `${f.bbox.x},${f.bbox.y} ${f.bbox.w}×${f.bbox.h}` : "—" }),
      h("td", { class: "note", text: f.notes.join("; ") }));
  });
  p.replaceChildren(
    h("h2", { text: "Detected fields and extracted values" }),
    h("div", { class: "two-col" },
      figure(d.images.original_fields, "Uploaded image: detected cheque boundary (green) and field regions mapped back (orange)."),
      figure(d.images.normalized_fields, "Cropped, perspective-corrected and deskewed cheque with field boxes.")),
    h("div", { class: "table-scroll" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Field", "Read as", "Parsed", "OCR conf.", "Engine", "Located by", "Box (x,y w×h)", "Notes"].map((t) => h("th", { text: t })))),
      h("tbody", {}, rows))),
    h("p", { class: "note", text: "Boxes are in normalised-cheque pixels; the PDF and JSON also include each field's corners in the uploaded image." }),
  );
}

function renderSignature(d) {
  const p = $('.panel[data-panel="signature"]');
  const sig = d.signature || { outcome: "unavailable", explanation: "Not performed." };
  const parts = [h("h2", { text: "Signature verification" }),
    h("p", {}, pill(sig.outcome, sig.outcome.replace("_", " "))),
    h("p", { text: sig.explanation })];
  if (d.images.signature_comparison) parts.push(figure(d.images.signature_comparison, "Registered reference signatures (left) and the signature extracted from the cheque (right)."));
  else if (d.images.signature_extracted) parts.push(figure(d.images.signature_extracted, "Signature extracted from the cheque."));
  if (sig.similarity !== null && sig.similarity !== undefined && sig.thresholds) {
    const t = sig.thresholds, agg = t.aggregation;
    parts.push(h("dl", { class: "kv" },
      h("dt", { text: "Similarity (1 − distance/2)" }), h("dd", { class: "mono", text: fmt(sig.similarity) }),
      h("dt", { text: `Decision score (${agg})` }), h("dd", { class: "mono", text: fmt(sig.distance) }),
      h("dt", { text: "Accept if ≤ / reject if ≥" }), h("dd", { class: "mono", text: `${fmt(t.accept)} / ${fmt(t.reject)}` }),
      h("dt", { text: "Calibrated forgery probability" }), h("dd", { class: "mono", text: fmt(sig.forgery_probability) }),
      h("dt", { text: "Distance to each reference" }), h("dd", { class: "mono", text: sig.per_reference_distance.map((x) => fmt(x)).join(", ") })));
    const lo = Math.min(t.accept, sig.distance) * 0.8, hi = Math.max(t.reject, sig.distance) * 1.2;
    const pct = (v) => `${Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100))}%`;
    const fill = h("div", { class: "fill" }); fill.style.width = pct(sig.distance);
    const tick = (v, label) => { const el = h("div", { class: "tick" }, h("span", { text: label })); el.style.left = pct(v); return el; };
    parts.push(h("div", { class: "meter", role: "img", "aria-label": "Decision score relative to thresholds" },
      fill, tick(t.accept, "accept"), tick(t.reject, "reject")));
  }
  if (sig.notes && sig.notes.length) parts.push(h("ul", { class: "note" }, sig.notes.map((n) => h("li", { text: n }))));
  parts.push(h("p", { class: "note", text: "A mismatch is a reason for manual verification, not proof of forgery: signatures vary naturally between occasions." }));
  p.replaceChildren(...parts);
}

function renderTampering(d) {
  const p = $('.panel[data-panel="tampering"]');
  const order = ["amount_numeric", "amount_words", "date", "payee"];
  const cards = order.filter((k) => d.field_analyses[k]).map((k) => {
    const a = d.field_analyses[k];
    const chips = Object.entries(a.scores).map(([name, v]) => {
      const thr = a.thresholds[name];
      const over = thr !== null && thr !== undefined && v >= thr;
      return h("span", { class: `score-chip${over ? " over" : ""}`, title: over ? "above calibrated threshold" : "" },
        `${name}: ${fmt(v)}${thr !== null && thr !== undefined ? ` / ${fmt(thr)}` : ""}`);
    });
    const inds = a.indicators.filter((i) => i.severity !== "info" || i.name.includes("open_ended"));
    return h("article", { class: "field-card" },
      h("header", {}, h("h3", { text: FIELD_NAMES[k] }),
        a.suspicious ? pill("FAIL", "possible alteration") : pill("PASS", "no alteration found")),
      d.images[`field_${k}`] ? h("img", { src: d.images[`field_${k}`], alt: `${FIELD_NAMES[k]} with evidence` }) : null,
      h("div", { class: "scores" }, chips),
      inds.length ? h("ul", {}, inds.map((i) => h("li", {}, h("strong", { text: `[${i.severity}] ` }), i.explanation)))
        : h("p", { class: "note", text: "No indicators." }),
      a.calibrated ? null : h("p", { class: "note", text: "Thresholds are defaults (not calibrated)." }));
  });
  p.replaceChildren(h("h2", { text: "Amount, date and payee alteration analysis" }),
    h("p", { class: "note", text: "Top image: the field as scanned. Bottom: evidence heatmap with flagged characters boxed. Scores are shown as value / calibrated threshold (5% false alarms on validation cheques); outlined chips exceed it." }),
    ...cards);
}

function renderForensics(d) {
  const p = $('.panel[data-panel="forensics"]');
  const maps = Object.keys(d.images).filter((k) => k.startsWith("forensics_") || k === "heatmap");
  const names = { heatmap: "Fused evidence", forensics_difference: "Difference vs original", forensics_ela: "Error level (ELA)",
    forensics_noise: "Noise", forensics_texture: "Texture (LBP)", forensics_jpeg_ghost: "JPEG ghost", forensics_edges: "Edges" };
  const img = h("img", { src: d.images.heatmap, alt: "forensic map" });
  const bar = h("div", { class: "mapbar" }, maps.map((k) => h("button", { class: `btn${k === "heatmap" ? " active" : ""}`, text: names[k] || k,
    onclick: (e) => { img.src = d.images[k]; $$(".mapbar .btn", p).forEach((b) => b.classList.toggle("active", b === e.target)); } })));
  const f = d.forensics || {};
  const inds = (f.indicators || []).filter((i) => i.severity !== "info");
  p.replaceChildren(h("h2", { text: "Image forensics" }), bar, h("figure", { class: "figure" }, img,
    h("figcaption", { text: "Each map is relative to this image: it shows areas unlike the rest of the cheque. Compression, scanning, folds and stamps also cause such differences." })),
    h("dl", { class: "kv" },
      h("dt", { text: "File format" }), h("dd", { text: f.jpeg && f.jpeg.is_jpeg ? `JPEG (estimated quality ${f.jpeg.estimated_quality ?? "?"})` : "Lossless or unknown (ELA/JPEG cues down-weighted)" }),
      h("dt", { text: "Trusted original" }), h("dd", { text: f.registration ? `registered by ${f.registration.method}, ${f.registration.inliers} inliers, ${(100 * f.registration.changed_fraction).toFixed(2)}% pixels changed` : "not provided" })),
    h("h3", { text: "Forensic indicators" }),
    inds.length ? h("ul", {}, inds.map((i) => h("li", {}, h("strong", { text: `[${i.severity}] ` }), i.explanation))) : h("p", { class: "note", text: "None." }));
}

function renderChecks(d) {
  const p = $('.panel[data-panel="checks"]');
  const info = d.info;
  p.replaceChildren(h("h2", { text: "Validation checks" }),
    h("div", { class: "table-scroll" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Status", "Check", "Severity", "Explanation"].map((t) => h("th", { text: t })))),
      h("tbody", {}, d.checks.map((c) => h("tr", {}, h("td", {}, pill(c.status)), h("td", { text: c.name }),
        h("td", { text: c.severity }), h("td", { text: c.explanation })))))),
    h("h2", { text: "Processing information" }),
    h("dl", { class: "kv" },
      h("dt", { text: "Total time" }), h("dd", { text: `${fmt(d.timings_seconds.total, 2)} s` }),
      h("dt", { text: "Stages" }), h("dd", { class: "mono", text: Object.entries(d.timings_seconds).filter(([k]) => k !== "total").map(([k, v]) => `${k} ${fmt(v, 2)}s`).join(" · ") }),
      h("dt", { text: "Device / OCR engines" }), h("dd", { text: `${info.device} · ${info.ocr_engines.join(", ") || "none"}` }),
      h("dt", { text: "Cheque detection" }), h("dd", { text: `${d.preprocessing.detection.method} (confidence ${fmt(d.preprocessing.detection.confidence, 2)}), deskew ${d.preprocessing.deskew_degrees}°, rotated 180°: ${d.preprocessing.rotated_180}` }),
      h("dt", { text: "Image quality" }), h("dd", { text: d.preprocessing.quality.map((q) => `${q.name}: ${q.status}`).join(" · ") }),
      h("dt", { text: "Models available" }), h("dd", { text: Object.entries(info.models).map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join("/") || "none" : v ? "yes" : "NO"}`).join(" · ") }),
      h("dt", { text: "Retention" }), h("dd", { text: `in memory only, deleted after ${Math.round(d.retention_seconds / 60)} minutes` })),
    figure(d.images.preprocessing_steps, "Preprocessing: grayscale, CLAHE-enhanced, adaptive threshold."));
}

// ---------------------------------------------------------------- original digit OCR tab
$("#btn-ocr").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("image", files.ocr);
  $("#btn-ocr").disabled = true;
  setStatus("Running OCR…", false, true, "#status-ocr");
  try {
    const res = await fetch("/api/ocr", { method: "POST", body: fd });
    const data = await res.json().catch(() => ({ error: `Server error ${res.status}` }));
    const box = $("#ocr-results");
    const checks = (list) => h("table", {}, h("tbody", {}, list.map((c) => h("tr", {}, h("td", {}, pill(c.status)), h("td", { text: c.name }), h("td", { text: c.detail })))));
    box.replaceChildren(h("h2", { text: "Input checks" }), checks(data.input_checks || []));
    if (data.annotated) {
      box.append(h("h2", { text: "Recognised digits" }), h("p", { class: "mono", text: data.digits.join(" ") || "none" }),
        figure(data.annotated, "CNN digit predictions (green boxes)."),
        h("h2", { text: "OCR text" }), h("pre", { class: "mono", text: data.ocr_text ?? "Tesseract not available" }),
        h("h2", { text: "Output checks" }), checks(data.output_checks));
    }
    box.hidden = false;
    setStatus(res.ok ? "Done." : data.error || "Failed", !res.ok, false, "#status-ocr");
  } catch (err) {
    setStatus(err.message, true, false, "#status-ocr");
  } finally {
    $("#btn-ocr").disabled = !files.ocr;
  }
});
