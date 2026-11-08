"""Main window: sidebar navigation and page switching."""
from __future__ import annotations

import logging
import tkinter as tk
import traceback
from tkinter import messagebox, ttk

from .. import config as C
from ..context import AppContext
from . import theme as T
from .page_dashboard import DashboardPage
from .page_image import ImagePage
from .page_library import LibraryPage
from .page_live import LivePage
from .page_reports import ReportsPage
from .page_settings import SettingsPage
from .page_video import VideoPage

log = logging.getLogger("nrityavision")

PAGES = [("dashboard", "Dashboard", DashboardPage), ("live", "Live Camera Detection", LivePage),
         ("image", "Upload Image", ImagePage), ("video", "Upload Dance Video", VideoPage),
         ("library", "Reference Mudra Library", LibraryPage), ("reports", "Performance Reports", ReportsPage),
         ("settings", "Settings", SettingsPage)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{C.APP_NAME} - Kuchipudi Mudra Recognition")
        self.geometry("1360x820")
        self.minsize(1100, 680)
        T.apply_theme(self)
        self.ctx = AppContext()
        self.report_callback_exception = self._on_error
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side = tk.Frame(self, bg=T.SIDEBAR, width=250)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        tk.Label(side, text="NrityaVision AI", bg=T.SIDEBAR, fg="white", font=(T.FONT, 17, "bold")).pack(anchor="w", padx=18, pady=(22, 0))
        tk.Label(side, text="Kuchipudi mudra intelligence", bg=T.SIDEBAR, fg=T.GOLD, font=(T.FONT, 9)).pack(anchor="w", padx=18, pady=(0, 18))
        self.nav = {}
        for key, text, _ in PAGES:
            b = tk.Button(side, text="  " + text, anchor="w", relief="flat", bd=0, bg=T.SIDEBAR, fg="white",
                          activebackground=T.SIDE_HOVER, activeforeground="white", font=(T.FONT, 11), padx=12, pady=10,
                          cursor="hand2", command=lambda k=key: self.show(k))
            b.pack(fill="x")
            b.bind("<Enter>", lambda e, w=b: w.configure(bg=T.SIDE_HOVER) if w["bg"] != T.SIDE_ACTIVE else None)
            b.bind("<Leave>", lambda e, w=b: w.configure(bg=T.SIDEBAR) if w["bg"] != T.SIDE_ACTIVE else None)
            self.nav[key] = b
        tk.Button(side, text="  Exit", anchor="w", relief="flat", bd=0, bg=T.SIDEBAR, fg="#ffb4b4", activebackground="#5a2230",
                  activeforeground="white", font=(T.FONT, 11), padx=12, pady=10, cursor="hand2", command=self.quit_app).pack(fill="x", side="bottom", pady=(0, 14))
        self.user_lbl = tk.Label(side, text="", bg=T.SIDEBAR, fg="#cdbfd0", font=(T.FONT, 9), wraplength=220, justify="left")
        self.user_lbl.pack(side="bottom", anchor="w", padx=18, pady=6)

        self.content = ttk.Frame(self, style="Page.TFrame")
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)
        self.pages = {}
        for key, _, cls in PAGES:
            p = cls(self.content, self.ctx, self)
            p.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = p
        self.current = None
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.refresh_user()
        self.show("dashboard")

    def refresh_user(self):
        u = self.ctx.db.get_user(self.ctx.user_id)
        self.user_lbl.configure(text=f"Signed in as\n{u['name'] if u else 'Guest'}")

    def show(self, key):
        if self.current:
            self.pages[self.current].on_hide()
        self.current = key
        for k, b in self.nav.items():
            b.configure(bg=T.SIDE_ACTIVE if k == key else T.SIDEBAR)
        self.pages[key].tkraise()
        self.pages[key].on_show()

    def toggle_fullscreen(self):
        self.attributes("-fullscreen", not bool(self.attributes("-fullscreen")))

    def quit_app(self):
        if messagebox.askyesno("Exit", "Close NrityaVision AI?", parent=self):
            for p in self.pages.values():
                try:
                    p.on_hide()
                except Exception:  # noqa: BLE001
                    pass
            self.destroy()

    def _on_error(self, exc, val, tb):
        log.error("".join(traceback.format_exception(exc, val, tb)))
        messagebox.showerror("Unexpected error", f"{val}\n\nDetails were written to data/logs/app.log", parent=self)
