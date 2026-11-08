from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .. import config as C
from . import theme as T
from .widgets import ImageView, Page, export_csv_dialog, export_pdf_dialog, run_with_progress

log = logging.getLogger("nrityavision")


class ImagePage(Page):
    def build(self):
        self.path = None
        self.analysis = None
        self.rowconfigure(2, weight=1)
        self.header("Upload Image Analysis", "Detect the hand, identify the mudra and compare it with the reference samples.")
        bar = ttk.Frame(self, style="Page.TFrame")
        bar.grid(row=1, column=0, sticky="ew", padx=24, pady=4)
        ttk.Button(bar, text="Choose image...", command=self.choose).pack(side="left")
        self.lbl = ttk.Label(bar, text="No image selected", style="Sub.TLabel")
        self.lbl.pack(side="left", padx=10)
        ttk.Label(bar, text="Expected mudra (optional):").pack(side="left", padx=(20, 4))
        self.expected = tk.StringVar(value="(none)")
        self.cb = ttk.Combobox(bar, textvariable=self.expected, state="readonly", width=18)
        self.cb.pack(side="left")
        ttk.Button(bar, text="Analyse", style="Accent.TButton", command=self.analyse).pack(side="left", padx=12)
        body = ttk.Frame(self, style="Page.TFrame")
        body.grid(row=2, column=0, sticky="nsew", padx=24, pady=8)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        self.view = ImageView(body)
        self.view.grid(row=0, column=0, sticky="nsew")
        self.view.show_message("Choose an image of a hand mudra")
        card = ttk.Frame(body, style="Card.TFrame", padding=10)
        card.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        card.rowconfigure(0, weight=1)
        card.columnconfigure(0, weight=1)
        self.txt = tk.Text(card, wrap="word", relief="flat", font=(T.FONT, 10), bg=T.CARD, state="disabled")
        self.txt.grid(row=0, column=0, sticky="nsew")
        for st, fg in T.STATUS_FG.items():
            self.txt.tag_configure(st, foreground=fg, font=(T.FONT, 12, "bold"))
        self.txt.tag_configure("h", font=(T.FONT, 11, "bold"))
        self.txt.tag_configure("m", foreground=T.MUTED)
        btns = ttk.Frame(self, style="Page.TFrame")
        btns.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 16))
        ttk.Button(btns, text="Save PDF report", command=lambda: export_pdf_dialog(self, self.ctx, self.analysis)).pack(side="left")
        ttk.Button(btns, text="Export CSV", command=lambda: export_csv_dialog(self, self.analysis)).pack(side="left", padx=8)

    def on_show(self):
        self.cb.configure(values=["(none)"] + self.ctx.db.mudra_names())

    def choose(self):
        p = filedialog.askopenfilename(parent=self, title="Choose a mudra image",
                                       filetypes=[("Images", " ".join("*" + e for e in C.IMAGE_EXTS)), ("All files", "*.*")])
        if p:
            self.path = p
            self.lbl.configure(text=Path(p).name)

    def _write(self, parts):
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        for text, tag in parts:
            self.txt.insert("end", text, tag)
        self.txt.configure(state="disabled")

    def analyse(self):
        if not self.path:
            messagebox.showinfo("Analyse", "Choose an image first.", parent=self)
            return
        exp = self.expected.get()
        exp = None if exp == "(none)" else exp
        an = self.ctx.image_analyzer()
        self.ctx.reload_profiles()

        def work(update, cancel):
            update(None, "Detecting hands...")
            return an.analyze(self.path, exp)

        def done(out, err):
            if err:
                messagebox.showerror("Image analysis", str(err), parent=self)
                return
            res, img = out
            self.view.show_bgr(img)
            aid = self.ctx.db.save_analysis(res, self.ctx.user_id)
            self.analysis = self.ctx.db.get_analysis(aid)
            self._show(self.analysis)

        run_with_progress(self, "Analysing image", work, done)

    def _show(self, a):
        parts = [(f"{a['source_name']}\n", "h")]
        if a.get("expected_source") not in (None, "None"):
            parts.append((f"Expected mudra: {a['expected_source']}\n", "m"))
        parts.append(("\n", ""))
        if not a["items"]:
            parts.append(("No hand detected.\n", C.UNRECOGNISED))
        for it in a["items"]:
            parts.append((f"{it['hand']} hand\n", "h"))
            parts.append((f"{it['detected']}", C.UNRECOGNISED if it["detected"] == C.UNRECOGNISED else "h"))
            if it.get("confidence") is not None:
                parts.append((f"   confidence {it['confidence']:.0%}", "m"))
            parts.append(("\n", ""))
            if it.get("similarity") is not None:
                parts.append((f"Similarity to reference: {it['similarity']:.0f}%\n", ""))
            parts.append((it["status"] + "\n", it["status"]))
            parts.append((it.get("suggestion", "") + "\n\n", ""))
        for n in a.get("notes", []):
            parts.append(("- " + n + "\n", "m"))
        self._write(parts)
