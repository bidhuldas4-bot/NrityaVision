from __future__ import annotations

from tkinter import ttk

from ..media import fmt_time
from . import theme as T
from .widgets import Page, make_tree, stat_card


class DashboardPage(Page):
    def build(self):
        self.rowconfigure(3, weight=1)
        self.header("Dashboard", "NrityaVision AI - Kuchipudi hand-mudra recognition and evaluation")
        row = ttk.Frame(self, style="Page.TFrame")
        row.grid(row=1, column=0, sticky="ew", padx=24, pady=6)
        self.cards = {}
        for i, (k, t) in enumerate((("mudras", "Mudras in library"), ("samples", "Training samples"),
                                    ("analyses", "Analyses done"), ("avg", "Average accuracy"))):
            f, v = stat_card(row, t)
            f.grid(row=0, column=i, padx=(0, 12), sticky="ew")
            row.columnconfigure(i, weight=1)
            self.cards[k] = v

        mid = ttk.Frame(self, style="Page.TFrame")
        mid.grid(row=2, column=0, sticky="ew", padx=24, pady=6)
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mc = ttk.Frame(mid, style="Card.TFrame", padding=14)
        mc.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        ttk.Label(mc, text="Recognition model", style="CardTitle.TLabel").pack(anchor="w")
        self.model_lbl = ttk.Label(mc, text="", style="Card.TLabel", justify="left", wraplength=480)
        self.model_lbl.pack(anchor="w", pady=(4, 0))
        qc = ttk.Frame(mid, style="Card.TFrame", padding=14)
        qc.grid(row=0, column=1, sticky="nsew")
        ttk.Label(qc, text="Quick actions", style="CardTitle.TLabel").pack(anchor="w")
        bar = ttk.Frame(qc, style="Card.TFrame")
        bar.pack(anchor="w", pady=8)
        for key, text in (("live", "Live camera"), ("image", "Analyse image"), ("video", "Analyse video"),
                          ("library", "Collect samples / train")):
            ttk.Button(bar, text=text, command=lambda k=key: self.app.show(k)).pack(side="left", padx=(0, 8))

        bottom = ttk.Frame(self, style="Page.TFrame")
        bottom.grid(row=3, column=0, sticky="nsew", padx=24, pady=(6, 18))
        bottom.rowconfigure(1, weight=1)
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, text="Recent analyses", style="CardTitle.TLabel", background=T.BG).grid(row=0, column=0, sticky="w")
        fr, self.tree = make_tree(bottom, [("id", "ID", 50), ("when", "Date", 150), ("kind", "Type", 70),
                                           ("src", "Source", 260), ("n", "Mudras", 70), ("ok", "Correct", 70),
                                           ("acc", "Accuracy", 90)], height=6, stretch_col="src")
        fr.grid(row=1, column=0, sticky="nsew")

    def on_show(self):
        st = self.ctx.db.stats(self.ctx.user_id)
        self.cards["mudras"].configure(text=str(st["mudras"]))
        self.cards["samples"].configure(text=str(st["samples"]))
        self.cards["analyses"].configure(text=str(st["analyses"]))
        self.cards["avg"].configure(text="-" if st["avg_accuracy"] is None else f"{st['avg_accuracy']:.0f}%")
        clf = self.ctx.classifier
        if clf.ready:
            m = clf.meta
            warn = " (optimistic: record more separate sessions)" if m.get("optimistic") else ""
            self.model_lbl.configure(text=(
                f"Trained {m['trained_at']} - {m['model']} on {m['n_samples']} samples, {len(clf.classes)} mudras.\n"
                f"Cross-validated accuracy: {m['cv_accuracy']:.0%}{warn}.\nMudras: {', '.join(clf.classes)}"))
        else:
            self.model_lbl.configure(text=(
                "No trained model yet.\n1) Reference Mudra Library > Dataset & Training: capture or import labelled "
                "samples for each mudra (50-100+ per mudra, varied angles and people).\n2) Press 'Train model'.\n"
                "Until then the app only tracks hand landmarks and will not guess mudra names."))
        self.tree.delete(*self.tree.get_children())
        for a in self.ctx.db.list_analyses(self.ctx.user_id)[:30]:
            acc = a["overall_accuracy"]
            self.tree.insert("", "end", values=(a["id"], a["created_at"], a["kind"], a["source_name"],
                                                a["total_items"], a["correct"], "-" if acc is None else f"{acc:.0f}%"))
