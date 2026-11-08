from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from .. import config as C
from .widgets import Page

# key, label, kind (entry|combo|check), (min, max) or options, help
FIELDS = [
    ("Camera & detection", [
        ("camera_index", "Camera index", "entry", (0, 10), "0 = default webcam"),
        ("det_conf", "Hand detection confidence", "entry", (0.1, 1.0), "MediaPipe threshold"),
        ("track_conf", "Hand tracking confidence", "entry", (0.1, 1.0), ""),
        ("model_complexity", "Landmark model complexity", "combo", ["0", "1"], "0 faster, 1 more accurate"),
        ("blur_threshold", "Blur threshold", "entry", (0, 500), "Raise if blurry hands are accepted"),
        ("min_brightness", "Minimum brightness", "entry", (0, 120), ""),
        ("max_brightness", "Maximum brightness", "entry", (120, 255), ""),
        ("min_palm_px", "Minimum hand size (px)", "entry", (5, 300), ""),
    ]),
    ("Recognition", [
        ("conf_threshold", "Minimum confidence", "entry", (0.3, 0.99), "Below this: 'Unable to recognize'"),
        ("margin_threshold", "Minimum margin to 2nd best", "entry", (0.0, 0.9), ""),
        ("ood_factor", "Outlier rejection factor", "entry", (0, 10), "0 disables; lower = stricter"),
    ]),
    ("Evaluation", [
        ("sim_correct", "'Correct' similarity >= (%)", "entry", (1, 100), ""),
        ("sim_needs", "'Needs improvement' similarity >= (%)", "entry", (1, 100), "Below = Incorrect"),
        ("sim_tau", "Similarity tolerance (tau)", "entry", (0.5, 6.0), "Larger = more lenient"),
    ]),
    ("Video analysis", [
        ("analysis_fps", "Frames analysed per second", "entry", (1, 30), "Higher = slower but finer"),
        ("min_segment_sec", "Minimum mudra hold (s)", "entry", (0.1, 3.0), "Shorter = transition"),
        ("gap_tol_sec", "Tracking gap tolerance (s)", "entry", (0.0, 2.0), ""),
        ("unclear_min_sec", "Report unclear stretches over (s)", "entry", (0.2, 10.0), ""),
        ("hand_mode", "Default hand", "combo", ["Either", "Left", "Right"], ""),
        ("multi_policy", "Frames with >2 hands", "combo", ["skip", "largest"], "skip = safer with several dancers"),
        ("mirror_uploads", "Uploads are mirrored (selfie) media", "check", None, "Leave off for normal recordings"),
    ]),
]


class SettingsPage(Page):
    def build(self):
        self.vars = {}
        self.rowconfigure(1, weight=1)
        self.header("Settings", "Changes apply to new analyses and camera sessions.")
        canvas = tk.Canvas(self, bg="#f5f2ee", highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew", padx=24)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        sb.grid(row=1, column=1, sticky="ns")
        canvas.configure(yscrollcommand=sb.set)
        inner = ttk.Frame(canvas, style="Page.TFrame")
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units") if self.winfo_ismapped() else None)

        row = 0
        for title, fields in FIELDS:
            card = ttk.Frame(inner, style="Card.TFrame", padding=14)
            card.grid(row=row, column=0, sticky="ew", pady=6)
            row += 1
            ttk.Label(card, text=title, style="CardTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
            for i, (key, label, kind, opt, hint) in enumerate(fields, start=1):
                ttk.Label(card, text=label, style="Card.TLabel").grid(row=i, column=0, sticky="w", pady=3)
                if kind == "check":
                    v = tk.BooleanVar()
                    ttk.Checkbutton(card, variable=v, style="Card.TCheckbutton").grid(row=i, column=1, sticky="w", padx=12)
                elif kind == "combo":
                    v = tk.StringVar()
                    ttk.Combobox(card, textvariable=v, values=opt, state="readonly", width=12).grid(row=i, column=1, sticky="w", padx=12)
                else:
                    v = tk.StringVar()
                    ttk.Entry(card, textvariable=v, width=14).grid(row=i, column=1, sticky="w", padx=12)
                ttk.Label(card, text=hint, style="CardSub.TLabel").grid(row=i, column=2, sticky="w")
                self.vars[key] = (v, kind, opt)
            card.columnconfigure(0, minsize=300)

        card = ttk.Frame(inner, style="Card.TFrame", padding=14)
        card.grid(row=row, column=0, sticky="ew", pady=6)
        ttk.Label(card, text="User profile", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(card, text="Current user", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=6)
        self.user = tk.StringVar()
        self.user_cb = ttk.Combobox(card, textvariable=self.user, state="readonly", width=24)
        self.user_cb.grid(row=1, column=1, padx=12)
        ttk.Button(card, text="Add user...", command=self.add_user).grid(row=1, column=2)
        card.columnconfigure(0, minsize=300)

        b = ttk.Frame(self, style="Page.TFrame")
        b.grid(row=2, column=0, sticky="ew", padx=24, pady=12)
        ttk.Button(b, text="Save settings", style="Accent.TButton", command=self.save).pack(side="left")
        ttk.Button(b, text="Reset to defaults", command=self.reset).pack(side="left", padx=8)
        ttk.Button(b, text="Toggle full screen (F11)", command=self.app.toggle_fullscreen).pack(side="left")

    def on_show(self):
        S = self.ctx.S()
        for key, (v, kind, _) in self.vars.items():
            v.set(bool(S[key]) if kind == "check" else str(S[key]))
        self._users()

    def _users(self):
        users = self.ctx.db.list_users()
        self.users = {f"{u['name']} ({u['role']})": u["id"] for u in users}
        self.user_cb.configure(values=list(self.users))
        cur = int(self.ctx.S()["current_user_id"])
        for k, i in self.users.items():
            if i == cur:
                self.user.set(k)

    def add_user(self):
        name = simpledialog.askstring("Add user", "Dancer / student name:", parent=self)
        if name:
            try:
                uid = self.ctx.db.add_user(name)
                self.ctx.db.set_settings({"current_user_id": uid})
            except ValueError as exc:
                messagebox.showwarning("User", str(exc), parent=self)
            self._users()
            self.app.refresh_user()

    def save(self):
        out = {}
        for key, (v, kind, opt) in self.vars.items():
            raw = v.get()
            default = C.DEFAULT_SETTINGS[key]
            try:
                if kind == "check":
                    out[key] = bool(raw)
                elif isinstance(default, int) and not isinstance(default, bool):
                    out[key] = int(float(raw))
                elif isinstance(default, float):
                    out[key] = float(raw)
                else:
                    out[key] = raw
                if kind == "entry" and not (opt[0] <= float(out[key]) <= opt[1]):
                    raise ValueError(f"must be between {opt[0]} and {opt[1]}")
            except ValueError as exc:
                messagebox.showwarning("Invalid value", f"{key}: {exc}", parent=self)
                return
        if out["sim_needs"] >= out["sim_correct"]:
            messagebox.showwarning("Invalid value", "'Needs improvement' threshold must be below 'Correct'.", parent=self)
            return
        out["current_user_id"] = self.users.get(self.user.get(), 1)
        self.ctx.db.set_settings(out)
        self.app.refresh_user()
        messagebox.showinfo("Settings", "Settings saved.", parent=self)

    def reset(self):
        if messagebox.askyesno("Reset", "Restore all settings to defaults?", parent=self):
            self.ctx.db.reset_settings()
            self.on_show()
