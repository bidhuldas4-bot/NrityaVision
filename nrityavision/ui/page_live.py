from __future__ import annotations

import threading
import time
import tkinter as tk
from collections import Counter, deque
from dataclasses import replace
from datetime import datetime
from tkinter import messagebox, ttk

import cv2
import numpy as np

from .. import config as C
from ..analyzer import STATUS_BGR
from ..hand_tracker import draw_hand
from ..media import open_camera
from . import theme as T
from .widgets import ImageView, Page


def smooth(dq):
    top, _ = Counter(a.label for a in dq).most_common(1)[0]
    pick = [a for a in dq if a.label == top]
    return replace(pick[-1], confidence=float(np.mean([a.confidence for a in pick])) if top else pick[-1].confidence)


class CameraWorker(threading.Thread):
    def __init__(self, ctx):
        super().__init__(daemon=True)
        self.ctx = ctx
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.target = None
        self.draw = True

    def run(self):
        cap = tracker = None
        try:
            S = self.ctx.S()
            cap = open_camera(S["camera_index"])
            tracker = self.ctx.tracker(False, 2)
            fa = self.ctx.frame_analyzer()
            sm = {"Left": deque(maxlen=5), "Right": deque(maxlen=5)}
            fails, fps, last = 0, 0.0, time.time()
            while not self.stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 100:
                        raise RuntimeError("The camera stopped delivering frames.")
                    time.sleep(0.02)
                    continue
                fails = 0
                frame = cv2.flip(frame, 1)
                results, seen = {}, set()
                for o in tracker.process(frame, mirrored=True):
                    a = fa.assess(o, self.target)
                    dq = sm[o.label]
                    dq.append(a)
                    shown = smooth(dq)
                    key = o.label if o.label not in results else o.label + "*"
                    results[key] = shown
                    seen.add(o.label)
                    if self.draw:
                        draw_hand(frame, o, shown.caption, STATUS_BGR.get(shown.status, (200, 200, 200)))
                for lab in sm:
                    if lab not in seen:
                        sm[lab].clear()
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
                last = now
                cv2.putText(frame, f"{fps:4.1f} fps", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                with self.lock:
                    self.latest = (frame, results)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
        finally:
            if tracker:
                tracker.close()
            if cap:
                cap.release()


class HandPanel(ttk.Frame):
    def __init__(self, master, title):
        super().__init__(master, style="Card.TFrame", padding=12)
        ttk.Label(self, text=title, style="CardSub.TLabel").pack(anchor="w")
        self.name = ttk.Label(self, text="No hand", style="Big.TLabel")
        self.name.pack(anchor="w")
        self.conf = ttk.Progressbar(self, maximum=100, length=240)
        self.conf.pack(anchor="w", pady=3)
        self.conf_lbl = ttk.Label(self, text="", style="CardSub.TLabel")
        self.conf_lbl.pack(anchor="w")
        self.sim = ttk.Label(self, text="", style="Card.TLabel")
        self.sim.pack(anchor="w")
        self.status = ttk.Label(self, text="", style="CardTitle.TLabel")
        self.status.pack(anchor="w", pady=(4, 0))
        self.tips = ttk.Label(self, text="", style="Card.TLabel", wraplength=300, justify="left")
        self.tips.pack(anchor="w", pady=(4, 0))

    def update_from(self, a):
        if a is None:
            self.name.configure(text="No hand", foreground=T.MUTED)
            self.conf["value"] = 0
            for w in (self.conf_lbl, self.sim, self.status, self.tips):
                w.configure(text="")
            return
        self.name.configure(text=a.label or C.UNRECOGNISED, foreground=T.ACCENT if a.label else T.MUTED)
        self.conf["value"] = a.confidence * 100 if a.usable else 0
        extra = f"  (closest, not accepted: {a.top_label})" if (a.usable and not a.label and a.top_label) else ""
        self.conf_lbl.configure(text=(f"Confidence {a.confidence:.0%}" if a.usable else "") + extra)
        self.sim.configure(text=f"Similarity to {a.sim_name}: {a.similarity:.0f}%" if a.similarity is not None else "")
        self.status.configure(text=a.status if a.usable else "Unusable view", foreground=T.STATUS_FG.get(a.status, T.TEXT))
        msgs = list(a.tips) if a.tips else ([a.reason] if a.reason and not a.label else [])
        self.tips.configure(text="\n".join("- " + m for m in msgs[:3]))


class LivePage(Page):
    def build(self):
        self.worker = None
        self._job = None
        self._last = None
        self.rowconfigure(1, weight=1)
        self.header("Live Camera Detection", "Both hands are tracked with 21 landmarks. Pick a practice target for live feedback.")
        body = ttk.Frame(self, style="Page.TFrame")
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 18))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, style="Page.TFrame")
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.view = ImageView(left)
        self.view.grid(row=0, column=0, sticky="nsew")
        self.view.show_message("Press 'Start camera'")
        bar = ttk.Frame(left, style="Page.TFrame")
        bar.grid(row=1, column=0, sticky="ew", pady=8)
        self.btn = ttk.Button(bar, text="Start camera", style="Accent.TButton", command=self.toggle)
        self.btn.pack(side="left")
        ttk.Button(bar, text="Snapshot", command=self.snapshot).pack(side="left", padx=8)
        self.show_lm = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Show landmarks", variable=self.show_lm, command=self._opts).pack(side="left", padx=8)
        ttk.Label(bar, text="Practice target:").pack(side="left", padx=(16, 4))
        self.target = tk.StringVar(value="(none)")
        self.cb = ttk.Combobox(bar, textvariable=self.target, state="readonly", width=18)
        self.cb.pack(side="left")
        self.cb.bind("<<ComboboxSelected>>", lambda e: self._opts())
        right = ttk.Frame(body, style="Page.TFrame")
        right.grid(row=0, column=1, sticky="ns", padx=(14, 0))
        self.banner = ttk.Label(right, text="", foreground="#b36b00", wraplength=320, justify="left")
        self.banner.pack(anchor="w", pady=(0, 6))
        self.panels = {"Left": HandPanel(right, "LEFT HAND"), "Right": HandPanel(right, "RIGHT HAND")}
        for p in self.panels.values():
            p.pack(fill="x", pady=(0, 10))
            p.update_from(None)
        ttk.Label(right, text="Similarity is geometric closeness to your reference samples;\nconfidence is the model's certainty.",
                  style="Sub.TLabel", justify="left").pack(anchor="w")

    def on_show(self):
        names = self.ctx.db.mudra_names()
        self.cb.configure(values=["(none)"] + names)
        if self.target.get() not in ["(none)"] + names:
            self.target.set("(none)")
        self.banner.configure(text="" if self.ctx.classifier.ready else
                              "No trained model: only landmarks are shown. Train it in the Reference Library.")

    def on_hide(self):
        self.stop()

    def _opts(self):
        if self.worker:
            t = self.target.get()
            self.worker.target = None if t == "(none)" else t
            self.worker.draw = self.show_lm.get()

    def toggle(self):
        self.stop() if self.worker else self.start()

    def start(self):
        self.ctx.reload_profiles()
        self.worker = CameraWorker(self.ctx)
        self._opts()
        self.worker.start()
        self.btn.configure(text="Stop camera")
        self.view.show_message("Opening camera...")
        self._poll()

    def stop(self):
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        if self.worker:
            self.worker.stop_evt.set()
            self.worker.join(timeout=2)
            self.worker = None
        self.btn.configure(text="Start camera")
        for p in self.panels.values():
            p.update_from(None)
        self.view.show_message("Camera stopped")

    def _poll(self):
        w = self.worker
        if not w:
            return
        if w.error:
            err = w.error
            self.stop()
            messagebox.showerror("Camera", err, parent=self)
            return
        with w.lock:
            latest = w.latest
        if latest is not None and latest is not self._last:
            self._last = latest
            frame, results = latest
            self.view.show_bgr(frame)
            shown = {"Left": None, "Right": None}
            for k, a in results.items():
                shown.setdefault(k.rstrip("*"), a)
                if shown[k.rstrip("*")] is None:
                    shown[k.rstrip("*")] = a
            for lab, p in self.panels.items():
                p.update_from(shown.get(lab))
            self._frame = frame
        self._job = self.after(30, self._poll)

    def snapshot(self):
        f = getattr(self, "_frame", None)
        if f is None:
            messagebox.showinfo("Snapshot", "Start the camera first.", parent=self)
            return
        p = C.SNAPSHOT_DIR / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
        cv2.imwrite(str(p), f)
        messagebox.showinfo("Snapshot", f"Saved to:\n{p}", parent=self)
