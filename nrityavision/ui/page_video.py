from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from .. import config as C
from ..analyzer import AnalysisCancelled
from ..charts import per_mudra_chart, timeline_chart
from ..media import fmt_time, open_video
from . import theme as T
from .widgets import (ITEM_COLUMNS_VIDEO, ChartFrame, ImageView, Page, export_csv_dialog, export_pdf_dialog,
                      fill_items_tree, make_tree)

log = logging.getLogger("nrityavision")


class VideoPlayer:
    def __init__(self, master, view, slider, time_var, on_time=None):
        self.master, self.view, self.slider, self.time_var, self.on_time = master, view, slider, time_var, on_time
        self.cap = self.cleanup = None
        self.fps, self.duration, self.playing, self._job, self._upd = 25.0, 0.0, False, None, False

    def load(self, path):
        self.release()
        self.cap, self.cleanup = open_video(path)
        f = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = f if 1 <= f <= 240 else 25.0
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.duration = n / self.fps if n > 0 else 0.0
        self.slider.configure(from_=0, to=max(self.duration, 1))
        self.seek(0)

    def release(self):
        self.pause()
        if self.cap:
            self.cap.release()
            self.cleanup()
        self.cap = None

    def _show_next(self):
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.view.show_bgr(frame)
        t = self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
        self._upd = True
        self.slider.set(t)
        self._upd = False
        self.time_var.set(f"{fmt_time(t, True)} / {fmt_time(self.duration)}")
        if self.on_time:
            self.on_time(t)
        return True

    def seek(self, sec):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(sec * self.fps)))
        self._show_next()

    def on_slider(self, v):
        if not self._upd and self.cap:
            self.seek(float(v))

    def toggle(self):
        self.pause() if self.playing else self.play()

    def play(self):
        if self.cap and not self.playing:
            self.playing = True
            self._tick()

    def pause(self):
        self.playing = False
        if self._job:
            self.master.after_cancel(self._job)
            self._job = None

    def _tick(self):
        if not self.playing:
            return
        t0 = time.time()
        if not self._show_next():
            self.playing = False
            return
        delay = max(1, int(1000 / self.fps - (time.time() - t0) * 1000))
        self._job = self.master.after(delay, self._tick)


class VideoPage(Page):
    def build(self):
        self.path = None
        self.analysis = None
        self.q = None
        self.cancel_evt = None
        self.ref_map = {}
        self.rowconfigure(2, weight=3)
        self.rowconfigure(3, weight=2)
        self.header("Upload Dance Video Analysis", "MP4, AVI or MOV (and MKV/M4V where your OpenCV build supports them).")
        c = ttk.Frame(self, style="Page.TFrame")
        c.grid(row=1, column=0, sticky="ew", padx=24, pady=2)
        r1 = ttk.Frame(c, style="Page.TFrame")
        r1.pack(fill="x")
        ttk.Button(r1, text="Select video...", command=self.choose).pack(side="left")
        self.lbl = ttk.Label(r1, text="No video selected", style="Sub.TLabel")
        self.lbl.pack(side="left", padx=10)
        ttk.Label(r1, text="Hand:").pack(side="left", padx=(16, 4))
        self.hand = tk.StringVar(value="Either")
        ttk.Combobox(r1, textvariable=self.hand, values=["Either", "Left", "Right"], state="readonly", width=8).pack(side="left")
        self.btn = ttk.Button(r1, text="Analyse video", style="Accent.TButton", command=self.start)
        self.btn.pack(side="left", padx=14)
        self.cbtn = ttk.Button(r1, text="Cancel", command=self.cancel, state="disabled")
        self.cbtn.pack(side="left")
        r2 = ttk.Frame(c, style="Page.TFrame")
        r2.pack(fill="x", pady=4)
        ttk.Label(r2, text="Expected sequence:").pack(side="left")
        self.mode = tk.StringVar(value="none")
        ttk.Radiobutton(r2, text="None (score each detected mudra)", variable=self.mode, value="none").pack(side="left", padx=6)
        ttk.Radiobutton(r2, text="Typed:", variable=self.mode, value="typed").pack(side="left")
        self.seq = tk.StringVar()
        ttk.Entry(r2, textvariable=self.seq, width=34).pack(side="left", padx=4)
        ttk.Radiobutton(r2, text="Reference video:", variable=self.mode, value="ref").pack(side="left", padx=(10, 0))
        self.ref = tk.StringVar()
        self.refcb = ttk.Combobox(r2, textvariable=self.ref, state="readonly", width=26)
        self.refcb.pack(side="left", padx=4)
        r3 = ttk.Frame(c, style="Page.TFrame")
        r3.pack(fill="x")
        self.bar = ttk.Progressbar(r3, maximum=100)
        self.bar.pack(side="left", fill="x", expand=True)
        self.status = ttk.Label(r3, text="Idle", style="Sub.TLabel", width=46)
        self.status.pack(side="left", padx=8)

        mid = ttk.Frame(self, style="Page.TFrame")
        mid.grid(row=2, column=0, sticky="nsew", padx=24, pady=6)
        mid.columnconfigure(0, weight=3)
        mid.columnconfigure(1, weight=2)
        mid.rowconfigure(0, weight=1)
        pl = ttk.Frame(mid, style="Page.TFrame")
        pl.grid(row=0, column=0, sticky="nsew")
        pl.rowconfigure(0, weight=1)
        pl.columnconfigure(1, weight=1)
        self.view = ImageView(pl)
        self.view.grid(row=0, column=0, columnspan=4, sticky="nsew")
        self.view.show_message("Select a video to preview it here")
        self.time_var = tk.StringVar(value="00:00.0 / 00:00")
        self.now_var = tk.StringVar()
        self.slider = ttk.Scale(pl, orient="horizontal", from_=0, to=1)
        self.player = VideoPlayer(self, self.view, self.slider, self.time_var, self.on_time)
        self.slider.configure(command=self.player.on_slider)
        ttk.Button(pl, text="Play / Pause", command=self.player.toggle).grid(row=1, column=0, pady=4, padx=(0, 6))
        self.slider.grid(row=1, column=1, sticky="ew")
        ttk.Label(pl, textvariable=self.time_var, width=18).grid(row=1, column=2, padx=6)
        ttk.Label(pl, textvariable=self.now_var, foreground=T.ACCENT, font=(T.FONT, 10, "bold")).grid(row=2, column=0, columnspan=4, sticky="w")

        rt = ttk.Frame(mid, style="Card.TFrame", padding=10)
        rt.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        rt.rowconfigure(1, weight=1)
        rt.columnconfigure(0, weight=1)
        self.summary = ttk.Label(rt, text="Results appear here after analysis.", style="Card.TLabel", justify="left", wraplength=420)
        self.summary.grid(row=0, column=0, sticky="ew")
        nb = ttk.Notebook(rt)
        nb.grid(row=1, column=0, sticky="nsew", pady=6)
        self.ch1, self.ch2 = ChartFrame(nb), ChartFrame(nb)
        nb.add(self.ch1, text="Per-mudra accuracy")
        nb.add(self.ch2, text="Timeline")
        self.notes = ttk.Label(rt, text="", style="CardSub.TLabel", justify="left", wraplength=420)
        self.notes.grid(row=2, column=0, sticky="ew")

        bot = ttk.Frame(self, style="Page.TFrame")
        bot.grid(row=3, column=0, sticky="nsew", padx=24, pady=(0, 14))
        bot.rowconfigure(0, weight=1)
        bot.columnconfigure(0, weight=1)
        fr, self.tree = make_tree(bot, ITEM_COLUMNS_VIDEO, height=6, stretch_col="tip")
        fr.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", self.jump)
        b = ttk.Frame(bot, style="Page.TFrame")
        b.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(b, text="Save PDF report", command=lambda: export_pdf_dialog(self, self.ctx, self.analysis)).pack(side="left")
        ttk.Button(b, text="Export CSV", command=lambda: export_csv_dialog(self, self.analysis)).pack(side="left", padx=8)
        ttk.Label(b, text="Double-click a row to jump to that moment in the video.", style="Sub.TLabel").pack(side="left", padx=10)

    # ------------------------------------------------------------------ events
    def on_show(self):
        self.ref_map = {f"#{r['id']} {r['title'] or Path(r['path']).name}": r
                        for r in self.ctx.db.list_references("video_sequence")}
        self.refcb.configure(values=list(self.ref_map))

    def on_hide(self):
        self.player.pause()

    def choose(self):
        p = filedialog.askopenfilename(parent=self, title="Select a dance video", filetypes=[
            ("Videos", " ".join("*" + e for e in C.VIDEO_EXTS)), ("All files", "*.*")])
        if not p:
            return
        self.path = p
        self.lbl.configure(text=Path(p).name)
        try:
            self.player.load(p)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Video", str(exc), parent=self)

    def on_time(self, t):
        if not self.analysis:
            return
        for it in self.analysis["items"]:
            if it["start_s"] <= t <= it["end_s"] and it["status"] != C.MISSED:
                self.now_var.set(f"Now: {it['detected']} - {it['status']}")
                return
        self.now_var.set("")

    def jump(self, _):
        sel = self.tree.selection()
        if sel and self.analysis:
            it = next((i for i in self.analysis["items"] if str(i["idx"]) == sel[0]), None)
            if it and self.path:
                self.player.pause()
                self.player.seek(max(0.0, (it["start_s"] or 0) - 0.3))

    # --------------------------------------------------------------- analysis
    def _expected(self):
        mode = self.mode.get()
        if mode == "typed":
            names = [n.strip() for n in self.seq.get().replace(">", ",").split(",") if n.strip()]
            if not names:
                raise ValueError("Type the expected mudra names separated by commas, e.g. Pataka, Tripataka, Mayura.")
            out, bad = [], []
            for n in names:
                c = self.ctx.db.canonical_name(n)
                (out if c else bad).append(c or n)
            if bad:
                raise ValueError("Unknown mudra name(s): " + ", ".join(bad) + ". Add them in the Reference Library first.")
            return out, None
        if mode == "ref":
            r = self.ref_map.get(self.ref.get())
            if not r:
                raise ValueError("Choose a reference sequence video (add one in the Reference Library).")
            return None, r
        return None, None

    def start(self):
        if not self.path:
            messagebox.showinfo("Analyse", "Select a video first.", parent=self)
            return
        if not self.ctx.classifier.ready:
            messagebox.showwarning("No model", "Train the recognition model first (Reference Library > Dataset & Training).", parent=self)
            return
        try:
            expected, ref = self._expected()
        except ValueError as exc:
            messagebox.showwarning("Expected sequence", str(exc), parent=self)
            return
        self.player.pause()
        self.ctx.reload_profiles()
        va = self.ctx.video_analyzer()
        self.q, self.cancel_evt = queue.Queue(), threading.Event()
        q, ev, path, hand = self.q, self.cancel_evt, self.path, self.hand.get()

        def cb(frac, text, preview):
            q.put(("p", frac, text, preview))

        def work():
            try:
                q.put(("done", va.analyze(path, expected, ref, hand, cb, ev)))
            except AnalysisCancelled:
                q.put(("cancel", None))
            except Exception as exc:  # noqa: BLE001
                log.exception("video analysis failed")
                q.put(("error", str(exc)))

        self.btn.configure(state="disabled")
        self.cbtn.configure(state="normal")
        self.bar["value"] = 0
        threading.Thread(target=work, daemon=True).start()
        self._poll()

    def cancel(self):
        if self.cancel_evt:
            self.cancel_evt.set()
            self.status.configure(text="Cancelling...")

    def _poll(self):
        last_preview = None
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "p":
                    self.bar["value"] = m[1] * 100
                    self.status.configure(text=m[2])
                    if m[3] is not None:
                        last_preview = m[3]
                else:
                    self._finish(m)
                    return
        except queue.Empty:
            pass
        if last_preview is not None:
            self.view.show_bgr(last_preview)
        self.after(80, self._poll)

    def _finish(self, m):
        self.btn.configure(state="normal")
        self.cbtn.configure(state="disabled")
        if m[0] == "cancel":
            self.status.configure(text="Cancelled")
            self.bar["value"] = 0
            return
        if m[0] == "error":
            self.status.configure(text="Failed")
            messagebox.showerror("Video analysis", m[1], parent=self)
            return
        self.bar["value"] = 100
        self.status.configure(text="Finished")
        aid = self.ctx.db.save_analysis(m[1], self.ctx.user_id)
        self.analysis = self.ctx.db.get_analysis(aid)
        self.show_result(self.analysis)
        try:
            self.player.load(self.path)
        except Exception:  # noqa: BLE001
            pass

    def show_result(self, a):
        s = a["summary"]
        acc = s.get("overall_accuracy")
        lines = [f"Duration {fmt_time(a['duration_s'])}   |   Mudras detected: {s.get('segments_detected', 0)}",
                 f"Correct: {s['correct']}   Needs improvement: {s['needs_improvement']}   "
                 f"Incorrect/missed: {s['incorrect'] + s.get('missed', 0)}   Unclear stretches: {s.get('unrecognised', 0)}",
                 "Overall mudra accuracy: " + ("not available" if acc is None else f"{acc:.1f}%")]
        if a.get("movement"):
            lines.append(f"Movement similarity (DTW): {a['movement']['similarity']:.0f}%")
        self.summary.configure(text="\n".join(lines))
        fill_items_tree(self.tree, a["items"], True)
        self.ch1.draw(per_mudra_chart, a.get("per_mudra", {}))
        self.ch2.draw(timeline_chart, a["items"], a["duration_s"])
        self.notes.configure(text="\n".join("- " + n for n in a.get("notes", [])[:4]))
