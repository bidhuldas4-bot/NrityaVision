"""Matplotlib charts drawn onto Figure objects (used by the Tkinter UI and the PDF report)."""
from __future__ import annotations

import io

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from . import config as C

STATUS_COLORS = {C.CORRECT: "#2e9e5b", C.NEEDS: "#f0a020", C.INCORRECT: "#d64545",
                 C.MISSED: "#8e2a2a", C.EXTRA: "#8a8f98", C.UNRECOGNISED: "#b0b4ba"}
ACCENT = "#7a1f3d"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def per_mudra_chart(fig: Figure, per_mudra: dict) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    _style(ax)
    if not per_mudra:
        ax.text(0.5, 0.5, "No evaluated mudras", ha="center", va="center", color="#777")
        ax.set_axis_off()
        return
    names = list(per_mudra)
    vals = [per_mudra[n]["accuracy"] for n in names]
    cols = [STATUS_COLORS[C.CORRECT] if v >= 75 else STATUS_COLORS[C.NEEDS] if v >= 55 else STATUS_COLORS[C.INCORRECT]
            for v in vals]
    bars = ax.bar(names, vals, color=cols)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=8)
    ax.set_title("Accuracy per mudra", fontsize=10)
    for b, v, n in zip(bars, vals, names):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%\n(n={per_mudra[n]['n']})",
                ha="center", fontsize=7)
    ax.axhline(75, color="#999", lw=0.8, ls="--")
    if len(names) > 6:
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()


def timeline_chart(fig: Figure, items: list, duration: float) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    _style(ax)
    timed = [it for it in items if it.get("end_s") is not None and it["end_s"] > it["start_s"]]
    if not timed:
        ax.text(0.5, 0.5, "No timeline available", ha="center", va="center", color="#777")
        ax.set_axis_off()
        return
    for it in timed:
        ax.broken_barh([(it["start_s"], it["end_s"] - it["start_s"])], (0, 1),
                       facecolors=STATUS_COLORS.get(it["status"], "#999"), edgecolor="white")
        w = it["end_s"] - it["start_s"]
        if w > max(duration, 1) * 0.035:
            ax.text(it["start_s"] + w / 2, 0.5, (it["detected"] or "")[:9], ha="center", va="center",
                    fontsize=6, color="white", rotation=90 if w < max(duration, 1) * 0.06 else 0)
    ax.set_xlim(0, max(duration, max(it["end_s"] for it in timed)))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title("Mudra timeline", fontsize=10)
    present = [s for s in (C.CORRECT, C.NEEDS, C.INCORRECT, C.EXTRA) if any(i["status"] == s for i in timed)]
    ax.legend(handles=[Patch(color=STATUS_COLORS[s], label=s) for s in present], fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)
    fig.tight_layout()


def history_chart(fig: Figure, rows: list[dict]) -> None:
    """rows: analyses (newest first) with overall_accuracy."""
    fig.clear()
    ax = fig.add_subplot(111)
    _style(ax)
    pts = [(r["id"], r["overall_accuracy"]) for r in reversed(rows) if r.get("overall_accuracy") is not None]
    if len(pts) < 1:
        ax.text(0.5, 0.5, "Run analyses to see your progress", ha="center", va="center", color="#777")
        ax.set_axis_off()
        return
    xs = list(range(1, len(pts) + 1))
    ax.plot(xs, [p[1] for p in pts], marker="o", color=ACCENT)
    ax.set_ylim(0, 105)
    ax.set_xticks(xs if len(xs) <= 12 else np.linspace(1, len(xs), 8, dtype=int))
    ax.set_xlabel("Analysis number", fontsize=8)
    ax.set_ylabel("Overall accuracy (%)", fontsize=8)
    ax.set_title("Progress over time", fontsize=10)
    fig.tight_layout()


def confusion_chart(fig: Figure, labels: list[str], cm: list[list[int]]) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.set_title("Cross-validated confusion matrix", fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "#222")
    fig.tight_layout()


def fig_to_png(draw_fn, size=(7.2, 2.8), dpi=150) -> bytes:
    fig = Figure(figsize=size, dpi=dpi)
    FigureCanvasAgg(fig)
    draw_fn(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return buf.getvalue()
