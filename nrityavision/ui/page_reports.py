from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from ..charts import history_chart, per_mudra_chart
from ..media import fmt_time
from . import theme as T
from .widgets import (ITEM_COLUMNS_VIDEO, ChartFrame, Page, export_csv_dialog, export_pdf_dialog, fill_items_tree,
                      make_tree)


class ReportsPage(Page):
    def build(self):
        self.current = None
        self.rowconfigure(1, weight=1)
        self.header("Performance Reports", "History of analyses. Open one to review it, then export as PDF or CSV.")
        body = ttk.Frame(self, style="Page.TFrame")
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 16))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        fr, self.tree = make_tree(body, [("id", "ID", 40), ("when", "Date", 140), ("kind", "Type", 60),
                                         ("src", "Source", 200), ("exp", "Expected", 120), ("n", "Mudras", 60),
                                         ("ok", "Correct", 60), ("acc", "Accuracy", 70)], height=8, stretch_col="src")
        fr.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.open_selected())
        side = ttk.Frame(body, style="Card.TFrame", padding=8)
        side.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(12, 0))
        side.rowconfigure(1, weight=1)
        side.rowconfigure(3, weight=1)
        side.columnconfigure(0, weight=1)
        self.info = ttk.Label(side, text="Select a report.", style="Card.TLabel", justify="left", wraplength=380)
        self.info.grid(row=0, column=0, sticky="ew")
        self.ch1 = ChartFrame(side, (4, 2.6))
        self.ch1.grid(row=1, column=0, sticky="nsew")
        ttk.Label(side, text="Your progress", style="CardTitle.TLabel").grid(row=2, column=0, sticky="w")
        self.ch2 = ChartFrame(side, (4, 2.4))
        self.ch2.grid(row=3, column=0, sticky="nsew")
        lower = ttk.Frame(body, style="Page.TFrame")
        lower.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        lower.rowconfigure(0, weight=1)
        lower.columnconfigure(0, weight=1)
        fr2, self.items = make_tree(lower, ITEM_COLUMNS_VIDEO, height=6, stretch_col="tip")
        fr2.grid(row=0, column=0, sticky="nsew")
        b = ttk.Frame(lower, style="Page.TFrame")
        b.grid(row=1, column=0, sticky="ew", pady=6)
        ttk.Button(b, text="Export PDF", style="Accent.TButton",
                   command=lambda: export_pdf_dialog(self, self.ctx, self.current)).pack(side="left")
        ttk.Button(b, text="Export CSV", command=lambda: export_csv_dialog(self, self.current)).pack(side="left", padx=6)
        ttk.Button(b, text="Delete report", style="Danger.TButton", command=self.delete).pack(side="left")
        ttk.Button(b, text="Refresh", command=self.refresh).pack(side="left", padx=6)

    def on_show(self):
        self.refresh()

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        rows = self.ctx.db.list_analyses(self.ctx.user_id)
        for a in rows:
            acc = a["overall_accuracy"]
            self.tree.insert("", "end", iid=str(a["id"]), values=(
                a["id"], a["created_at"], a["kind"], a["source_name"], (a["expected_source"] or "None")[:28],
                a["total_items"], a["correct"], "-" if acc is None else f"{acc:.0f}%"))
        self.ch2.draw(history_chart, rows)

    def open_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        a = self.ctx.db.get_analysis(int(sel[0]))
        if not a:
            return
        self.current = a
        s = a.get("summary") or {}
        acc = a.get("overall_accuracy")
        ms = a.get("mean_similarity")
        acc_t = "n/a" if acc is None else "%.1f%%" % acc
        ms_t = "n/a" if ms is None else "%.0f%%" % ms
        self.info.configure(text=(
            "%s  (%s, %s)\nCorrect %s | Needs improvement %s | Incorrect/missed %s\n"
            "Overall accuracy: %s   Mean similarity: %s" % (
                a["source_name"], a["kind"], fmt_time(a.get("duration_s")), a["correct"],
                a["needs_improvement"], a["incorrect"], acc_t, ms_t)))
        self.ch1.draw(per_mudra_chart, a.get("per_mudra", {}))
        fill_items_tree(self.items, a["items"], a["kind"] == "video")

    def delete(self):
        sel = self.tree.selection()
        if sel and messagebox.askyesno("Delete", "Delete this report and its saved PDFs?", parent=self):
            self.ctx.db.delete_analysis(int(sel[0]))
            self.current = None
            self.items.delete(*self.items.get_children())
            self.refresh()
