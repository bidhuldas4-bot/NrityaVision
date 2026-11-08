"""Reusable widgets: Page base class, image view, charts, tables, progress dialog, export helpers."""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from .. import config as C
from ..media import fmt_time
from ..report import build_csv, build_pdf
from . import theme as T

log = logging.getLogger("nrityavision")


class Page(ttk.Frame):
    """Base class for all dashboard pages."""

    def __init__(self, parent, ctx, app):
        super().__init__(parent, style="Page.TFrame")
        self.ctx, self.app = ctx, app
        self.columnconfigure(0, weight=1)
        self.build()

    def header(self, title: str, subtitle: str = "") -> ttk.Frame:
        f = ttk.Frame(self, style="Page.TFrame")
        f.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 8))
        ttk.Label(f, text=title, style="Title.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(f, text=subtitle, style="Sub.TLabel").pack(anchor="w")
        return f

    def build(self): ...
    def on_show(self): ...
    def on_hide(self): ...


class ImageView(tk.Canvas):
    """Canvas that shows a BGR frame scaled to fit while keeping the aspect ratio."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=T.DARK, highlightthickness=0, **kw)
        self._photo = None
        self._frame = None
        self._msg = ""
        self.bind("<Configure>", lambda e: self._render())

    def show_bgr(self, frame):
        self._frame = frame
        self._render()

    def show_message(self, text: str):
        self._frame, self._msg = None, text
        self._render()

    def clear(self):
        self.show_message("")

    def _render(self):
        self.delete("all")
        W, H = self.winfo_width(), self.winfo_height()
        if W < 10 or H < 10:
            return
        if self._frame is None:
            if self._msg:
                self.create_text(W // 2, H // 2, text=self._msg, fill="#9aa0a8", font=(T.FONT, 12),
                                 width=W - 40, justify="center")
            return
        h, w = self._frame.shape[:2]
        s = min(W / w, H / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        rgb = cv2.cvtColor(cv2.resize(self._frame, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.create_image(W // 2, H // 2, image=self._photo)


class ChartFrame(ttk.Frame):
    def __init__(self, master, figsize=(5, 3)):
        super().__init__(master, style="Card.TFrame")
        self.fig = Figure(figsize=figsize, dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw(self, fn, *args):
        try:
            fn(self.fig, *args)
        except Exception:  # noqa: BLE001
            log.exception("chart failed")
        self.canvas.draw_idle()


def make_tree(parent, columns: list[tuple[str, str, int]], height: int = 8, stretch_col: str | None = None):
    """columns: (id, heading, width). Returns (frame, tree)."""
    fr = ttk.Frame(parent, style="Card.TFrame")
    tree = ttk.Treeview(fr, columns=[c[0] for c in columns], show="headings", height=height, selectmode="browse")
    for cid, head, w in columns:
        tree.heading(cid, text=head)
        tree.column(cid, width=w, anchor="w", stretch=(cid == stretch_col))
    ys = ttk.Scrollbar(fr, orient="vertical", command=tree.yview)
    xs = ttk.Scrollbar(fr, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
    tree.grid(row=0, column=0, sticky="nsew")
    ys.grid(row=0, column=1, sticky="ns")
    xs.grid(row=1, column=0, sticky="ew")
    fr.rowconfigure(0, weight=1)
    fr.columnconfigure(0, weight=1)
    for st, bg in T.STATUS_BG.items():
        tree.tag_configure(st, background=bg)
    return fr, tree


def fill_items_tree(tree, items: list[dict], video: bool = True) -> None:
    tree.delete(*tree.get_children())
    for it in items:
        sim = it.get("similarity")
        tree.insert("", "end", iid=str(it["idx"]), tags=(it["status"],), values=(
            it["idx"], fmt_time(it.get("start_s"), True) if video else it.get("hand", ""),
            it.get("expected") or "-", it.get("detected") or "-", "-" if sim is None else f"{sim:.0f}%",
            it["status"], it.get("suggestion", "")))


ITEM_COLUMNS_VIDEO = [("idx", "#", 40), ("time", "Timestamp", 90), ("exp", "Expected", 110), ("det", "Detected", 110),
                      ("sim", "Similarity", 80), ("status", "Status", 130), ("tip", "Improvement suggestion", 520)]


def stat_card(parent, title: str, value: str = "-"):
    f = ttk.Frame(parent, style="Card.TFrame", padding=14)
    ttk.Label(f, text=title, style="CardSub.TLabel").pack(anchor="w")
    v = ttk.Label(f, text=value, style="Stat.TLabel")
    v.pack(anchor="w")
    return f, v


def run_with_progress(parent, title: str, work, on_done, determinate: bool = False, cancellable: bool = False):
    """Run ``work(update, cancel_event)`` in a thread behind a small modal dialog.

    ``update(frac, text)`` may be called from the worker. ``on_done(result, error)`` runs in the UI thread.
    """
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.transient(parent.winfo_toplevel())
    dlg.resizable(False, False)
    dlg.protocol("WM_DELETE_WINDOW", lambda: None)
    fr = ttk.Frame(dlg, padding=20)
    fr.pack(fill="both", expand=True)
    lbl = ttk.Label(fr, text="Working...", width=60)
    lbl.pack(anchor="w")
    bar = ttk.Progressbar(fr, mode="determinate" if determinate else "indeterminate", length=420, maximum=100)
    bar.pack(pady=10, fill="x")
    cancel = threading.Event()
    if cancellable:
        ttk.Button(fr, text="Cancel", command=cancel.set).pack(anchor="e")
    if not determinate:
        bar.start(12)
    dlg.update_idletasks()
    dlg.grab_set()
    q: queue.Queue = queue.Queue()

    def runner():
        try:
            q.put(("done", work(lambda f, t="": q.put(("p", f, t)), cancel), None))
        except Exception as exc:  # noqa: BLE001
            log.exception("background task failed")
            q.put(("done", None, exc))

    threading.Thread(target=runner, daemon=True).start()

    def poll():
        try:
            while True:
                m = q.get_nowait()
                if m[0] == "p":
                    if determinate and m[1] is not None:
                        bar["value"] = m[1] * 100
                    lbl.configure(text=m[2])
                else:
                    dlg.grab_release()
                    dlg.destroy()
                    on_done(m[1], m[2])
                    return
        except queue.Empty:
            pass
        dlg.after(80, poll)

    poll()


def open_file(path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}" >/dev/null 2>&1 &')
    except Exception:  # noqa: BLE001
        pass


def export_pdf_dialog(parent, ctx, analysis: dict) -> None:
    if not analysis:
        messagebox.showinfo("Export PDF", "There is no analysis to export yet.", parent=parent)
        return
    stem = Path(analysis.get("source_name", "report")).stem
    default = f"NrityaVision_{stem}_{datetime.now():%Y%m%d_%H%M}.pdf"
    path = filedialog.asksaveasfilename(parent=parent, title="Save PDF report", defaultextension=".pdf",
                                        initialdir=str(C.REPORTS_DIR), initialfile=default,
                                        filetypes=[("PDF", "*.pdf")])
    if not path:
        return
    try:
        build_pdf(analysis, path)
        if analysis.get("id"):
            ctx.db.add_report(analysis["id"], path)
    except Exception as exc:  # noqa: BLE001
        log.exception("pdf export failed")
        messagebox.showerror("Export PDF", f"Could not create the PDF:\n{exc}", parent=parent)
        return
    if messagebox.askyesno("Report saved", f"Saved to:\n{path}\n\nOpen it now?", parent=parent):
        open_file(path)


def export_csv_dialog(parent, analysis: dict) -> None:
    if not analysis:
        messagebox.showinfo("Export CSV", "There is no analysis to export yet.", parent=parent)
        return
    path = filedialog.asksaveasfilename(parent=parent, title="Save CSV", defaultextension=".csv",
                                        initialdir=str(C.REPORTS_DIR),
                                        initialfile=f"NrityaVision_{Path(analysis.get('source_name', 'r')).stem}.csv",
                                        filetypes=[("CSV", "*.csv")])
    if path:
        try:
            build_csv(analysis, path)
            messagebox.showinfo("Export CSV", f"Saved to:\n{path}", parent=parent)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export CSV", str(exc), parent=parent)
