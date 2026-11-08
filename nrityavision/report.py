"""PDF (ReportLab) and CSV report generation from an analysis dictionary."""
from __future__ import annotations

import csv
import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import config as C
from .charts import fig_to_png, per_mudra_chart, timeline_chart
from .media import fmt_time

MAROON = colors.HexColor("#7a1f3d")
STATUS_BG = {C.CORRECT: "#dff3e6", C.NEEDS: "#fff0d1", C.INCORRECT: "#fbdcdc", C.MISSED: "#f3cccc",
             C.EXTRA: "#ececec", C.UNRECOGNISED: "#ececec"}


def _p(text, style):
    return Paragraph(escape(str(text)), style)


def _pct(v):
    return "-" if v is None else f"{v:.0f}%"


def build_pdf(a: dict, path) -> str:
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=MAROON, fontSize=20, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=MAROON, fontSize=12, spaceBefore=10, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, textColor=colors.grey)
    hdr = ParagraphStyle("hdr", parent=small, textColor=colors.white, fontName="Helvetica-Bold")

    s = a.get("summary", {})
    kind = a.get("kind", "video")
    story = [Paragraph("NrityaVision AI - Performance Report", h1),
             Paragraph("Kuchipudi hand-mudra analysis", sub), Spacer(1, 6)]

    acc = s.get("overall_accuracy")
    meta_rows = [
        ["Source", a.get("source_name", "-")],
        ["Analysis type", "Dance video" if kind == "video" else "Image"],
    ]
    if kind == "video":
        meta_rows.append(["Total duration", fmt_time(a.get("duration_s"))])
    meta_rows += [
        ["Dancer / user", a.get("user_name") or "-"],
        ["Expected sequence", a.get("expected_source", "None")],
        ["Mudras detected", str(s.get("segments_detected", 0))],
        ["Correct mudras", str(s.get("correct", 0))],
        ["Mudras needing improvement", str(s.get("needs_improvement", 0))],
        ["Incorrect / missed", str(s.get("incorrect", 0) + s.get("missed", 0))],
        ["Unclear stretches (not scored)", str(s.get("unrecognised", 0))],
        ["Overall mudra accuracy", "Not available (nothing could be evaluated)" if acc is None else f"{acc:.1f}%"],
        ["Mean similarity to reference", _pct(s.get("mean_similarity"))],
    ]
    mv = a.get("movement")
    if mv:
        meta_rows.append(["Movement similarity (DTW vs reference video)", _pct(mv.get("similarity"))])
    meta_rows.append(["Generated", a.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M")])
    t = Table([[_p(k, small), _p(v, body)] for k, v in meta_rows], colWidths=[62 * mm, 110 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4eef0")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [t]

    per = a.get("per_mudra") or {}
    if per:
        story.append(Paragraph("Individual mudra accuracy", h2))
        rows = [[_p(h, hdr) for h in ("Mudra", "Accuracy", "Status", "Evaluated", "Correct / Needs / Incorrect")]]
        for name, d in per.items():
            st = C.CORRECT if d["accuracy"] >= 75 else C.NEEDS if d["accuracy"] >= 55 else C.INCORRECT
            rows.append([_p(name, body), _p(f"{d['accuracy']:.0f}%", body), _p(st, body), _p(d["n"], body),
                         _p(f"{d['correct']} / {d['needs']} / {d['incorrect']}", body)])
        pt = Table(rows, colWidths=[40 * mm, 25 * mm, 40 * mm, 25 * mm, 42 * mm], repeatRows=1)
        style = [("BACKGROUND", (0, 0), (-1, 0), MAROON), ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey)]
        for i, (name, d) in enumerate(per.items(), start=1):
            st = C.CORRECT if d["accuracy"] >= 75 else C.NEEDS if d["accuracy"] >= 55 else C.INCORRECT
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(STATUS_BG[st])))
        pt.setStyle(TableStyle(style))
        story.append(pt)
        story += [Spacer(1, 6), Image(io.BytesIO(fig_to_png(lambda f: per_mudra_chart(f, per), (6.4, 2.6))),
                                      width=165 * mm, height=67 * mm)]
    items = a.get("items", [])
    if kind == "video" and items:
        story += [Spacer(1, 4), Image(io.BytesIO(fig_to_png(
            lambda f: timeline_chart(f, items, a.get("duration_s", 0)), (7.2, 2.2))), width=170 * mm, height=52 * mm)]

    if items:
        story.append(Paragraph("Detailed evaluation", h2))
        head = ["#", "Time" if kind == "video" else "Hand", "Expected", "Detected", "Similarity", "Status", "Suggestion"]
        rows = [[_p(h, hdr) for h in head]]
        for it in items:
            first = fmt_time(it.get("start_s")) if kind == "video" else it.get("hand", "")
            rows.append([_p(it["idx"], small), _p(first, small), _p(it.get("expected") or "-", small),
                         _p(it.get("detected") or "-", small), _p(_pct(it.get("similarity")), small),
                         _p(it["status"], small), _p(it.get("suggestion", ""), small)])
        dt = Table(rows, colWidths=[8 * mm, 14 * mm, 22 * mm, 22 * mm, 16 * mm, 24 * mm, 76 * mm], repeatRows=1)
        style = [("BACKGROUND", (0, 0), (-1, 0), MAROON), ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                 ("VALIGN", (0, 0), (-1, -1), "TOP")]
        for i, it in enumerate(items, start=1):
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(STATUS_BG.get(it["status"], "#ffffff"))))
        dt.setStyle(TableStyle(style))
        story.append(dt)

    mist = [m for m in a.get("mistakes", [])]
    if mist:
        story.append(Paragraph("Mistake timestamps", h2))
        rows = [[_p(h, hdr) for h in ("Time", "Type", "Issue", "Suggestion")]]
        for m in mist:
            ts = fmt_time(m["timestamp_s"], True) if kind == "video" else "-"
            rows.append([_p(ts, small), _p({"error": "Error", "warning": "Improve", "info": "Unclear"}.get(
                m.get("severity"), "Note"), small), _p(m["description"], small), _p(m.get("suggestion", ""), small)])
        mt = Table(rows, colWidths=[20 * mm, 18 * mm, 55 * mm, 89 * mm], repeatRows=1)
        mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), MAROON), ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                                ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(mt)

    q = a.get("quality") or {}
    notes = list(a.get("notes", []))
    if kind == "video" and q.get("frames_sampled"):
        notes.insert(0, f"Frames analysed: {q['frames_sampled']}; hands recognised in {q['hands_recognised_pct']:.0f}%, "
                        f"no hand visible in {q['no_hand_pct']:.0f}%, unclear in {q['unclear_pct']:.0f}%.")
    story.append(Paragraph("Notes and limitations", h2))
    method = ("Method: MediaPipe hand landmarks are normalised for hand size, mirrored for left hands and converted "
              "to rotation-invariant features. A trained classifier proposes the mudra; mudras with low confidence "
              "are reported as 'Unable to recognize'. Accuracy is the mean geometric similarity to labelled "
              "reference samples (wrong or missed mudras score 0). Classifier confidence is not treated as "
              "dance correctness.")
    for n in notes + [method]:
        story.append(Paragraph("- " + escape(n), small))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(15 * mm, 8 * mm, "NrityaVision AI - generated report")
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm, title="NrityaVision AI Report")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return str(path)


def build_csv(a: dict, path) -> str:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Index", "Start (s)", "End (s)", "Hand", "Expected", "Detected", "Similarity %",
                    "Confidence %", "Status", "Suggestion"])
        for it in a.get("items", []):
            sim, conf = it.get("similarity"), it.get("confidence")
            w.writerow([it["idx"], f"{it.get('start_s') or 0:.2f}", f"{it.get('end_s') or 0:.2f}", it.get("hand", ""),
                        it.get("expected", ""), it.get("detected", ""),
                        "" if sim is None else f"{sim:.1f}", "" if conf is None else f"{conf * 100:.1f}",
                        it["status"], it.get("suggestion", "")])
    return str(path)
