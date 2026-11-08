from __future__ import annotations

import logging
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from .. import config as C
from ..charts import confusion_chart
from ..classifier import MIN_PER_CLASS
from ..dataset import sample_from_image, samples_from_video, store_reference_file
from ..hand_tracker import draw_hand
from ..media import open_camera, open_video
from . import theme as T
from .widgets import ChartFrame, ImageView, Page, make_tree, run_with_progress

log = logging.getLogger("nrityavision")
IMG_TYPES = [("Images", " ".join("*" + e for e in C.IMAGE_EXTS)), ("All files", "*.*")]
VID_TYPES = [("Videos", " ".join("*" + e for e in C.VIDEO_EXTS)), ("All files", "*.*")]


class MudraDialog(tk.Toplevel):
    def __init__(self, parent, title, name="", meaning="", desc=""):
        super().__init__(parent)
        self.title(title)
        self.transient(parent.winfo_toplevel())
        self.result = None
        f = ttk.Frame(self, padding=16)
        f.pack(fill="both", expand=True)
        self.vars = [tk.StringVar(value=name), tk.StringVar(value=meaning)]
        for i, lab in enumerate(("Name", "English meaning")):
            ttk.Label(f, text=lab).grid(row=i, column=0, sticky="w", pady=3)
            ttk.Entry(f, textvariable=self.vars[i], width=40).grid(row=i, column=1, pady=3, padx=8)
        ttk.Label(f, text="Description").grid(row=2, column=0, sticky="nw", pady=3)
        self.desc = tk.Text(f, width=40, height=4, wrap="word")
        self.desc.insert("1.0", desc)
        self.desc.grid(row=2, column=1, pady=3, padx=8)
        b = ttk.Frame(f)
        b.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(b, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(b, text="Save", style="Accent.TButton", command=self.ok).pack(side="right")
        self.grab_set()
        self.wait_window()

    def ok(self):
        if not self.vars[0].get().strip():
            messagebox.showwarning("Mudra", "Name is required.", parent=self)
            return
        self.result = (self.vars[0].get(), self.vars[1].get(), self.desc.get("1.0", "end").strip())
        self.destroy()


class CaptureDialog(tk.Toplevel):
    """Captures labelled hand samples from the webcam (3 s countdown, then automatic sampling)."""

    def __init__(self, parent, ctx, mudra_name, mudra_id):
        super().__init__(parent)
        self.ctx, self.mudra_name, self.mudra_id = ctx, mudra_name, mudra_id
        self.title(f"Capture samples - {mudra_name}")
        self.geometry("900x700")
        self.transient(parent.winfo_toplevel())
        self.cap = self.tracker = None
        self.state, self.t_start, self.last = "idle", 0.0, 0.0
        self.samples, self.saved, self.group = [], 0, ctx.new_group()
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Hand:").pack(side="left")
        self.hand = tk.StringVar(value="Either")
        ttk.Combobox(top, textvariable=self.hand, values=["Either", "Left", "Right"], state="readonly", width=8).pack(side="left", padx=4)
        ttk.Label(top, text="Samples:").pack(side="left", padx=(12, 2))
        self.n = tk.IntVar(value=80)
        ttk.Spinbox(top, from_=10, to=500, textvariable=self.n, width=6).pack(side="left")
        self.btn = ttk.Button(top, text="Start capture", style="Accent.TButton", command=self.begin)
        self.btn.pack(side="left", padx=12)
        ttk.Label(top, text="Hold the mudra, then slowly turn/tilt it and vary the distance.", style="Sub.TLabel").pack(side="left")
        self.view = ImageView(self)
        self.view.pack(fill="both", expand=True, padx=8)
        self.bar = ttk.Progressbar(self, maximum=100)
        self.bar.pack(fill="x", padx=8, pady=4)
        self.msg = ttk.Label(self, text="Opening camera...")
        self.msg.pack(anchor="w", padx=8, pady=(0, 8))
        self.protocol("WM_DELETE_WINDOW", self.close)
        try:
            self.cap = open_camera(ctx.S()["camera_index"])
            self.tracker = ctx.tracker(False, 2)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Camera", str(exc), parent=self)
            self.after(10, self.close)
            return
        self.grab_set()
        self.after(30, self.tick)

    def begin(self):
        self.state, self.t_start, self.samples = "count", time.time(), []
        self.btn.configure(state="disabled")

    def tick(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            obs = self.tracker.process(frame, mirrored=True)
            want = self.hand.get()
            good = [o for o in obs if o.usable and (want == "Either" or o.label == want)]
            for o in obs:
                draw_hand(frame, o, o.label, (0, 200, 80))
            best = max(good, key=lambda o: o.palm_px) if good else None
            if self.state == "count":
                left = 3 - (time.time() - self.t_start)
                if left <= 0:
                    self.state, self.last = "capture", 0.0
                else:
                    cv2.putText(frame, str(int(left) + 1), (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 215, 255), 8)
                    self.msg.configure(text="Get ready...")
            elif self.state == "capture":
                if best is not None and time.time() - self.last >= 0.12:
                    self.samples.append((best.canon, best.label))
                    self.last = time.time()
                if best is None:
                    why = obs[0].reasons[0] if obs and obs[0].reasons else "show your hand to the camera"
                    self.msg.configure(text=f"Waiting for a clear hand: {why}")
                else:
                    self.msg.configure(text=f"Captured {len(self.samples)} / {self.n.get()}")
                self.bar["value"] = 100 * len(self.samples) / max(self.n.get(), 1)
                if len(self.samples) >= self.n.get():
                    self.finish()
            self.view.show_bgr(frame)
        if self.cap is not None:
            self.after(15, self.tick)

    def finish(self):
        self.state = "idle"
        hands = {}
        for c, h in self.samples:
            hands.setdefault(h, []).append(c)
        for h, lst in hands.items():
            self.ctx.db.add_samples(self.mudra_id, lst, h, "webcam", self.group)
        self.saved += len(self.samples)
        self.ctx.reload_profiles()
        self.btn.configure(state="normal")
        self.group = self.ctx.new_group()
        self.msg.configure(text=f"Saved {len(self.samples)} samples ({self.saved} this dialog). "
                                "Change angle/distance and capture another session for better validation.")

    def close(self):
        cap, self.cap = self.cap, None
        if cap:
            cap.release()
        if self.tracker:
            self.tracker.close()
        self.destroy()


class LibraryPage(Page):
    def build(self):
        self.rowconfigure(1, weight=1)
        self.header("Reference Mudra Library", "Manage mudras, reference media and the training dataset.")
        nb = ttk.Notebook(self)
        nb.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 16))
        self.t1, self.t2, self.t3 = (ttk.Frame(nb, style="Page.TFrame", padding=10) for _ in range(3))
        nb.add(self.t1, text="Mudras")
        nb.add(self.t2, text="Reference images & videos")
        nb.add(self.t3, text="Dataset & training")
        self._tab_mudras()
        self._tab_media()
        self._tab_training()

    def on_show(self):
        self.refresh()

    # ----------------------------------------------------------------- tab 1
    def _tab_mudras(self):
        t = self.t1
        t.rowconfigure(0, weight=1)
        t.columnconfigure(0, weight=1)
        fr, self.mtree = make_tree(t, [("id", "ID", 40), ("name", "Mudra", 150), ("meaning", "Meaning", 150),
                                       ("desc", "Description", 420), ("ns", "Samples", 70), ("nr", "Refs", 60)],
                                   stretch_col="desc")
        fr.grid(row=0, column=0, sticky="nsew")
        b = ttk.Frame(t, style="Page.TFrame")
        b.grid(row=1, column=0, sticky="ew", pady=8)
        ttk.Button(b, text="Add mudra", style="Accent.TButton", command=self.add_mudra).pack(side="left")
        ttk.Button(b, text="Edit", command=self.edit_mudra).pack(side="left", padx=6)
        ttk.Button(b, text="Delete", style="Danger.TButton", command=self.del_mudra).pack(side="left")
        ttk.Button(b, text="Add all 28 Asamyuta Hastas", command=self.add_all).pack(side="left", padx=14)
        ttk.Label(b, text="Descriptions follow common textual definitions; follow your guru's reference where they differ.",
                  style="Sub.TLabel").pack(side="left")

    def _sel(self, tree):
        s = tree.selection()
        return tree.item(s[0], "values") if s else None

    def add_mudra(self):
        d = MudraDialog(self, "Add mudra")
        if d.result:
            try:
                self.ctx.db.add_mudra(*d.result)
            except ValueError as exc:
                messagebox.showwarning("Mudra", str(exc), parent=self)
            self.refresh()

    def edit_mudra(self):
        v = self._sel(self.mtree)
        if not v:
            return
        d = MudraDialog(self, "Edit mudra", v[1], v[2], v[3])
        if d.result:
            try:
                self.ctx.db.update_mudra(int(v[0]), *d.result)
            except ValueError as exc:
                messagebox.showwarning("Mudra", str(exc), parent=self)
            self.refresh()

    def del_mudra(self):
        v = self._sel(self.mtree)
        if v and messagebox.askyesno("Delete", f"Delete '{v[1]}' with all its samples and reference media?\n"
                                     "Retrain the model afterwards.", parent=self):
            self.ctx.db.delete_mudra(int(v[0]))
            self.ctx.reload_profiles()
            self.refresh()

    def add_all(self):
        n = self.ctx.db.add_all_asamyuta()
        messagebox.showinfo("Asamyuta Hastas", f"Added {n} mudra names. They need labelled samples before they can be recognised.", parent=self)
        self.refresh()

    # ----------------------------------------------------------------- tab 2
    def _tab_media(self):
        t = self.t2
        t.rowconfigure(0, weight=1)
        t.columnconfigure(0, weight=1)
        fr, self.rtree = make_tree(t, [("id", "ID", 40), ("kind", "Type", 120), ("mudra", "Mudra", 130),
                                       ("title", "Title", 200), ("file", "File", 400)], stretch_col="file")
        fr.grid(row=0, column=0, sticky="nsew")
        b = ttk.Frame(t, style="Page.TFrame")
        b.grid(row=1, column=0, sticky="ew", pady=8)
        ttk.Button(b, text="Add reference image(s)", style="Accent.TButton", command=self.add_images).pack(side="left")
        ttk.Button(b, text="Add reference video", command=self.add_video).pack(side="left", padx=6)
        ttk.Button(b, text="Preview", command=self.preview).pack(side="left")
        ttk.Button(b, text="Delete", style="Danger.TButton", command=self.del_ref).pack(side="left", padx=6)
        ttk.Label(b, text="Images and single-mudra clips also become labelled training samples. "
                          "Sequence videos are used for DTW comparison.", style="Sub.TLabel").pack(side="left", padx=10)

    def _choose_mudra(self, title="Choose mudra"):
        names = self.ctx.db.mudra_names()
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.transient(self.winfo_toplevel())
        v = tk.StringVar(value=names[0] if names else "")
        f = ttk.Frame(dlg, padding=16)
        f.pack()
        ttk.Label(f, text="Mudra:").pack(side="left")
        ttk.Combobox(f, textvariable=v, values=names, state="readonly", width=24).pack(side="left", padx=8)
        out = {}
        ttk.Button(f, text="OK", style="Accent.TButton", command=lambda: (out.update(v=v.get()), dlg.destroy())).pack(side="left")
        dlg.grab_set()
        dlg.wait_window()
        return out.get("v") or None

    def add_images(self):
        name = self._choose_mudra("Mudra shown in the image(s)")
        if not name:
            return
        files = filedialog.askopenfilenames(parent=self, title="Reference images", filetypes=IMG_TYPES)
        if files:
            self._import_images(name, files, store=True)

    def _import_images(self, name, files, store=False):
        mid = self.ctx.db.get_mudra_id(name)

        def work(update, cancel):
            ok = bad = 0
            problems = []
            for i, f in enumerate(files):
                update(i / len(files), f"Processing {Path(f).name} ({i + 1}/{len(files)})")
                try:
                    canon, hand, why = sample_from_image(self.ctx, f)
                except Exception as exc:  # noqa: BLE001
                    canon, hand, why = None, None, str(exc)
                if canon is None:
                    bad += 1
                    problems.append(f"{Path(f).name}: {why}")
                    continue
                self.ctx.db.add_samples(mid, [canon], hand, "image", self.ctx.new_group())
                if store:
                    self.ctx.db.add_reference(mid, "image", str(store_reference_file(f, name)), Path(f).name)
                ok += 1
            return ok, bad, problems

        def done(res, err):
            if err:
                messagebox.showerror("Import", str(err), parent=self)
            else:
                ok, bad, problems = res
                msg = f"Added {ok} sample(s) to {name}. Skipped {bad}."
                if problems:
                    msg += "\n\n" + "\n".join(problems[:8])
                messagebox.showinfo("Import finished", msg, parent=self)
            self.ctx.reload_profiles()
            self.refresh()

        run_with_progress(self, "Importing images", work, done, determinate=True)

    def add_video(self):
        dlg = tk.Toplevel(self)
        dlg.title("Add reference video")
        dlg.transient(self.winfo_toplevel())
        f = ttk.Frame(dlg, padding=16)
        f.pack()
        kind = tk.StringVar(value="video_clip")
        ttk.Radiobutton(f, text="Single-mudra clip (adds training samples)", variable=kind, value="video_clip").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(f, text="Full sequence demonstration (for DTW comparison)", variable=kind, value="video_sequence").grid(row=1, column=0, columnspan=2, sticky="w")
        names = self.ctx.db.mudra_names()
        mv, tv = tk.StringVar(value=names[0] if names else ""), tk.StringVar()
        ttk.Label(f, text="Mudra (clips):").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(f, textvariable=mv, values=names, state="readonly", width=24).grid(row=2, column=1)
        ttk.Label(f, text="Title:").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=tv, width=27).grid(row=3, column=1)
        go = {}
        ttk.Button(f, text="Choose video and add", style="Accent.TButton",
                   command=lambda: (go.update(ok=True), dlg.destroy())).grid(row=4, column=0, columnspan=2, pady=(10, 0))
        dlg.grab_set()
        dlg.wait_window()
        if not go:
            return
        path = filedialog.askopenfilename(parent=self, title="Reference video", filetypes=VID_TYPES)
        if not path:
            return
        k, name, title = kind.get(), mv.get(), tv.get().strip() or Path(path).stem
        mid = self.ctx.db.get_mudra_id(name) if k == "video_clip" else None
        if k == "video_clip" and not mid:
            messagebox.showwarning("Reference video", "Choose a mudra for a single-mudra clip.", parent=self)
            return

        def work(update, cancel):
            stored = store_reference_file(path, name if k == "video_clip" else "sequences")
            n = 0
            if k == "video_clip":
                found = samples_from_video(self.ctx, stored, update, cancel)
                grp = self.ctx.new_group()
                for hand in ("Left", "Right"):
                    lst = [c for c, h in found if h == hand]
                    if lst:
                        self.ctx.db.add_samples(mid, lst, hand, "video_clip", grp)
                n = len(found)
            self.ctx.db.add_reference(mid, k, str(stored), title)
            return n

        def done(n, err):
            if err:
                messagebox.showerror("Reference video", str(err), parent=self)
            else:
                messagebox.showinfo("Reference video", "Video added." + (f" Extracted {n} training samples." if k == "video_clip" else ""), parent=self)
            self.ctx.reload_profiles()
            self.refresh()

        run_with_progress(self, "Adding video", work, done, determinate=True)

    def preview(self):
        v = self._sel(self.rtree)
        if not v:
            return
        ref = self.ctx.db.get_reference(int(v[0]))
        if not ref or not Path(ref["path"]).exists():
            messagebox.showwarning("Preview", "File not found.", parent=self)
            return
        from .widgets import open_file
        open_file(ref["path"])

    def del_ref(self):
        v = self._sel(self.rtree)
        if v and messagebox.askyesno("Delete", "Remove this reference file?\n(Samples already extracted stay in the dataset.)", parent=self):
            self.ctx.db.delete_reference(int(v[0]))
            self.refresh()

    # ----------------------------------------------------------------- tab 3
    def _tab_training(self):
        t = self.t3
        t.columnconfigure(0, weight=1)
        t.columnconfigure(1, weight=1)
        t.rowconfigure(0, weight=1)
        left = ttk.Frame(t, style="Page.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        fr, self.dtree = make_tree(left, [("name", "Mudra", 150), ("n", "Samples", 80), ("state", "Dataset status", 220)], stretch_col="state")
        fr.grid(row=0, column=0, sticky="nsew")
        b = ttk.Frame(left, style="Page.TFrame")
        b.grid(row=1, column=0, sticky="ew", pady=6)
        ttk.Button(b, text="Capture from webcam", style="Accent.TButton", command=self.capture).grid(row=0, column=0, padx=(0, 6), pady=2)
        ttk.Button(b, text="Import images", command=self.imp_images).grid(row=0, column=1, padx=6)
        ttk.Button(b, text="Import folder", command=self.imp_folder).grid(row=0, column=2, padx=6)
        ttk.Button(b, text="Import video clip", command=self.imp_clip).grid(row=1, column=0, padx=(0, 6), pady=2, sticky="w")
        ttk.Button(b, text="Clear samples", style="Danger.TButton", command=self.clear).grid(row=1, column=1, padx=6, sticky="w")
        ttk.Label(left, text=f"Need at least {MIN_PER_CLASS} samples per mudra; aim for 100+ from several sessions, "
                             "distances, angles and people. Real labelled data decides accuracy.",
                  style="Sub.TLabel", wraplength=520, justify="left").grid(row=2, column=0, sticky="w")
        right = ttk.Frame(t, style="Page.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)
        self.train_btn = ttk.Button(right, text="Train model", style="Accent.TButton", command=self.train)
        self.train_btn.grid(row=0, column=0, sticky="w")
        self.log = tk.Text(right, height=9, wrap="word", font=("Consolas", 9), state="disabled", relief="flat")
        self.log.grid(row=1, column=0, sticky="ew", pady=6)
        self.cm = ChartFrame(right, (4.5, 3.4))
        self.cm.grid(row=2, column=0, sticky="nsew")

    def _names_sel(self):
        v = self._sel(self.dtree)
        if not v:
            messagebox.showinfo("Select a mudra", "Select a mudra in the table first.", parent=self)
            return None
        return v[0]

    def capture(self):
        n = self._names_sel()
        if n:
            CaptureDialog(self, self.ctx, n, self.ctx.db.get_mudra_id(n)).wait_window()
            self.refresh()

    def imp_images(self):
        n = self._names_sel()
        if n:
            f = filedialog.askopenfilenames(parent=self, title=f"Images of {n}", filetypes=IMG_TYPES)
            if f:
                self._import_images(n, f)

    def imp_folder(self):
        n = self._names_sel()
        if n:
            d = filedialog.askdirectory(parent=self, title=f"Folder with images of {n}")
            if d:
                files = [str(p) for p in sorted(Path(d).iterdir()) if p.suffix.lower() in C.IMAGE_EXTS]
                if not files:
                    messagebox.showinfo("Import", "No images found in that folder.", parent=self)
                else:
                    self._import_images(n, files)

    def imp_clip(self):
        n = self._names_sel()
        if not n:
            return
        p = filedialog.askopenfilename(parent=self, title=f"Video clip of {n}", filetypes=VID_TYPES)
        if not p:
            return
        mid = self.ctx.db.get_mudra_id(n)

        def work(update, cancel):
            found = samples_from_video(self.ctx, p, update, cancel)
            grp = self.ctx.new_group()
            for hand in ("Left", "Right"):
                lst = [c for c, h in found if h == hand]
                if lst:
                    self.ctx.db.add_samples(mid, lst, hand, "video_clip", grp)
            return len(found)

        def done(k, err):
            messagebox.showinfo("Import", f"Added {k} samples." if not err else str(err), parent=self)
            self.ctx.reload_profiles()
            self.refresh()

        run_with_progress(self, "Extracting samples", work, done, determinate=True)

    def clear(self):
        n = self._names_sel()
        if n and messagebox.askyesno("Clear", f"Delete all samples of {n}?", parent=self):
            self.ctx.db.delete_samples(self.ctx.db.get_mudra_id(n))
            self.ctx.reload_profiles()
            self.refresh()

    def _log(self, s):
        self.log.configure(state="normal")
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def train(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.train_btn.configure(state="disabled")
        lines = []

        def work(update, cancel):
            return self.ctx.train(lambda s: (lines.append(s), update(None, s)))

        def done(meta, err):
            self.train_btn.configure(state="normal")
            for s in lines:
                self._log(s)
            if err:
                self._log("ERROR: " + str(err))
                messagebox.showerror("Training", str(err), parent=self)
                return
            self.ctx.reload_model()
            self._log(f"Cross-validated accuracy {meta['cv_accuracy']:.1%}, macro-F1 {meta['cv_macro_f1']:.2f}")
            for c in meta["labels"]:
                r = meta["report"][c]
                self._log(f"  {c}: precision {r['precision']:.2f} recall {r['recall']:.2f} (n={int(r['support'])})")
            self.cm.draw(confusion_chart, meta["labels"], meta["confusion"])
            self.refresh()

        run_with_progress(self, "Training model", work, done)

    # --------------------------------------------------------------- refresh
    def refresh(self):
        ms = self.ctx.db.list_mudras()
        for tree in (self.mtree, self.dtree, self.rtree):
            tree.delete(*tree.get_children())
        trained = set(self.ctx.classifier.classes)
        for m in ms:
            self.mtree.insert("", "end", values=(m["id"], m["name"], m["meaning"], m["description"], m["n_samples"], m["n_refs"]))
            n = m["n_samples"]
            state = ("Ready to train" if n >= MIN_PER_CLASS else f"Need {MIN_PER_CLASS - n} more") + (" - in model" if m["name"] in trained else "")
            self.dtree.insert("", "end", values=(m["name"], n, state))
        for r in self.ctx.db.list_references():
            self.rtree.insert("", "end", values=(r["id"], r["kind"].replace("_", " "), r["mudra"] or "-", r["title"], Path(r["path"]).name))
        meta = self.ctx.classifier.meta
        if self.ctx.classifier.ready and meta.get("confusion"):
            self.cm.draw(confusion_chart, meta["labels"], meta["confusion"])
