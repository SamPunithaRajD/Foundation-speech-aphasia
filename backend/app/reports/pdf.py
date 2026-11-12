"""Research report (PDF) generated from stored analysis data with ReportLab.

The layout is intentionally a research summary, not a diagnostic certificate:
every page carries the prototype disclaimer, and unavailable components are
reported as such rather than omitted or filled in.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.config import FORMAL_PROJECT_TITLE, PROJECT_NAME
from app.schemas.analysis import AnalysisRecord

DISCLAIMER = ("RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE. This report is generated automatically for research review. "
              "It is not a clinical diagnosis and must not be used for diagnosis or treatment decisions.")

_styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_styles["Heading1"], fontSize=15, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=_styles["Heading2"], fontSize=11.5, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#0f5257"))
BODY = ParagraphStyle("body", parent=_styles["BodyText"], fontSize=8.8, leading=11.5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"))
WARN = ParagraphStyle("warn", parent=BODY, textColor=colors.HexColor("#8a4b00"))


def _p(text, style=BODY) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def _table(rows, widths) -> Table:
    t = Table([[c if isinstance(c, Paragraph) else _p(c) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c4c4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _fmt(v, nd=3):
    if v is None:
        return "-"
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def build_report(record: AnalysisRecord) -> bytes:
    buf = io.BytesIO()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#8a4b00"))
        canvas.drawString(18 * mm, 10 * mm, "Research prototype - not a clinical diagnosis - not a certified medical device")
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{PROJECT_NAME} | {record.analysis_id} | page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title=f"{PROJECT_NAME} research report {record.analysis_id}",
                            author=PROJECT_NAME)
    W = A4[0] - 36 * mm
    el: list = [
        _p(PROJECT_NAME.upper() + " | RESEARCH PROTOTYPE", SMALL),
        _p("Speech analysis research report", H1),
        _p(FORMAL_PROJECT_TITLE, SMALL),
        Spacer(1, 5),
        _p(DISCLAIMER, WARN),
        _p("Summary", H2),
        _table([["Field", "Value"],
                ["Analysis ID", record.analysis_id],
                ["Data mode", "LIVE BACKEND RESULT (not simulated)"],
                ["Status", record.status.value],
                ["Created", record.created_at], ["Completed", record.completed_at or "-"],
                ["Report generated", generated],
                ["Representation strategy", f"{record.config.strategy} ({', '.join(record.config.models)})"]],
               [W * 0.32, W * 0.68]),
    ]
    if record.attention:
        el += [_p("Items needing attention", H2)] + [_p("- " + a, BODY) for a in record.attention]

    a, p = record.audio, record.preprocessing
    el.append(_p("Audio and preprocessing", H2))
    rows = [["Property", "Value"]]
    if a:
        rows += [["Format / decoder", f"{a.format} / {a.decoder}"], ["File size", f"{a.size_bytes / 1024:.0f} KB"],
                 ["Original sample rate / channels", f"{a.original_sample_rate} Hz / {a.original_channels}"]]
    rows += [["Final sample rate", f"{p.final_sample_rate} Hz" if p.final_sample_rate else "-"],
             ["Duration", f"{_fmt(p.duration_seconds, 2)} s"],
             ["Speech duration (VAD)", f"{_fmt(p.speech_duration_seconds, 2)} s"],
             ["Speech segments (VAD)", _fmt(p.segment_count)],
             ["Steps applied", ", ".join(p.steps) or "-"],
             ["Not applied", "; ".join(f"{k}: {v}" for k, v in p.steps_not_applied.items()) or "-"]]
    el.append(_table(rows, [W * 0.32, W * 0.68]))

    el.append(_p("Speech foundation model embeddings", H2))
    rows = [["Model", "State", "Dimension", "Frames", "Note"]]
    for r in record.embeddings.results:
        rows.append([f"{r.model_key} ({r.model_id})", r.state.value, _fmt(r.dimension), _fmt(r.frames), r.message or (r.pooling or "")])
    el.append(_table(rows, [W * 0.28, W * 0.18, W * 0.12, W * 0.1, W * 0.32]) if len(rows) > 1 else _p("Not run."))

    t = record.transcript
    el.append(_p("Automatic speech recognition transcript", H2))
    el.append(_p(f"State: {t.state.value} | Model: {t.model or '-'} | Language: {t.language or '-'}"
                 + (f" | Mean word token probability: {t.confidence['mean_word_probability']:.2f} (uncalibrated)"
                    if t.confidence else ""), SMALL))
    el.append(_p(f"Source: {t.source}. Transcription accuracy is unknown (no human reference transcript).", SMALL))
    if t.quality:
        q = t.quality
        el.append(_p(f"ASR-reported indicators: mean avg log-prob {q.get('mean_avg_logprob')}, max no-speech prob "
                     f"{q.get('max_no_speech_prob')}, max compression ratio {q.get('max_compression_ratio')}. "
                     f"Flags: {'; '.join(q.get('flags') or []) or 'none'}.", SMALL))
    if t.segments:
        el.append(_table([["Start", "End", "Text"]] + [[f"{s.start:.1f}s", f"{s.end:.1f}s", s.text] for s in t.segments],
                         [W * 0.1, W * 0.1, W * 0.8]))
    else:
        el.append(_p(t.message or "No transcript available."))

    el.append(_p("Clinical language features", H2))
    el.append(_p(record.features.disclaimer, SMALL))
    rows = [["Feature", "Value", "Unit", "Source", "Status / note"]]
    for f in record.features.features:
        rows.append([f.label, _fmt(f.value), f.unit, f.source, f.status + (f": {f.reason}" if f.reason else "")
                     + (f" ({f.note})" if f.note else "")])
    el.append(_table(rows, [W * 0.22, W * 0.1, W * 0.14, W * 0.12, W * 0.42]))

    fu = record.fusion
    el.append(_p("Multimodal fusion", H2))
    el.append(_p(f"State: {fu.state.value} | Strategy: {fu.strategy or '-'} | Unified dimension: {_fmt(fu.unified_dimension)} "
                 f"| Maturity: {fu.maturity.value}. {fu.message or ''}"))

    sv = record.severity
    tdef = sv.target or {}
    el.append(_p(sv.target_definition or "Severity score", H2))
    if sv.prediction_available and sv.severity_score is not None:
        el.append(_table([["Field", "Value"],
                          [sv.target_definition or "Severity score", f"{sv.severity_score:.1f} / 100 (higher = more impaired)"],
                          ["Predicted AQ", f"{sv.predicted_aq:.1f}"],
                          ["Band (AQ range label applied to predicted AQ)", sv.severity_band or "-"],
                          ["Uncertainty", (f"{sv.uncertainty['interval'][0]}-{sv.uncertainty['interval'][1]} "
                                           f"({sv.uncertainty.get('method')})") if sv.uncertainty else "NOT_AVAILABLE"],
                          ["Model version", sv.model_version or "-"],
                          ["Model stage", (sv.model_stage or "research model")
                           + (" - small-sample pilot, cross-validated only" if sv.model_stage == "PILOT" else "")],
                          ["Model evaluation", ("; ".join(f"{k}: {v}" for k, v in sv.evaluation_summary.items())
                                                if sv.evaluation_summary else "see model card")],
                          ["Evaluation status", (sv.classifier or {}).get("status", "Evaluated on held-out research participants")],
                          ["Clinical validation", "NOT_VALIDATED"],
                          ["This recording", sv.clinical_validity or "-"]], [W * 0.42, W * 0.58]))
        el.append(_p(f"Target: {tdef.get('transformation', '-')}. Source: {tdef.get('source', '-')}. "
                     f"Bands: {(tdef.get('bands') or {}).get('source', '-')} {sv.clinical_validity_reason or ''}", SMALL))
    else:
        el.append(_p(f"{sv.state.value}: {sv.message or 'No prediction available.'}", WARN))
        el.append(_p("No severity score is reported. No values have been estimated or simulated.", SMALL))
        if sv.raw_output is not None:
            el.append(_p(f"Raw model output (not a valid estimate, shown for transparency): {sv.raw_output} "
                         f"| model {sv.model_version or '-'}", SMALL))
    if sv.input_checks:
        ic = sv.input_checks
        el.append(_p(f"Input checks - distribution score {ic.get('ood_score')} vs limit {ic.get('ood_limit')} "
                     f"({ic.get('ood_definition')}); inputs imputed at inference: {ic.get('imputed_inputs') or 'none'}.", SMALL))
    for lim in sv.limitations:
        el.append(_p('- ' + lim, SMALL))

    ex = record.explainability
    el.append(_p("Explainability", H2))
    el.append(_p(ex.interpretation_note, SMALL))
    if ex.methods and any(m.attributions for m in ex.methods):
        for m in ex.methods:
            el.append(_p(f"{m.method.upper()} ({m.state.value}{'' if m.reliable is not False else ', UNRELIABLE'}) - explains: "
                         f"{m.target or '-'} | model {ex.model_version or '-'} | raw output {ex.explained_raw_output}"))
            if m.message:
                el.append(_p(m.message, SMALL))
            if m.attributions:
                el.append(_table([["Input", "Attribution"]] + [[x.feature, f"{x.contribution:+.4f}"] for x in m.attributions],
                                 [W * 0.6, W * 0.4]))
    else:
        el.append(_p(f"{ex.state.value}: {ex.message or 'Not available.'}", WARN))

    el += [_p("Limitations", H2)] + [_p("- " + x) for x in [
        "Research prototype output; no component has been clinically validated.",
        "Transcripts are produced by a general-purpose ASR model that is not adapted to aphasic speech; errors, omitted "
        "fillers and normalised repetitions are expected and propagate into language features.",
        "Clinical language features are automatic research measurements, not standardised clinical assessments.",
        "A severity score requires a model trained and evaluated on verified clinical scores; its absence is reported "
        "above rather than approximated. A score for an uploaded recording is never a clinical assessment.",
        "Model attributions (SHAP/LIME) describe model behaviour, not medical causation.",
    ]]
    doc.build(el, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()
