"""Colours, fonts and ttk styling for the dashboard."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .. import config as C

BG = "#f5f2ee"
CARD = "#ffffff"
SIDEBAR = "#2b1a2f"
SIDE_HOVER = "#43294a"
SIDE_ACTIVE = "#7a1f3d"
ACCENT = "#7a1f3d"
ACCENT_HOVER = "#94284c"
GOLD = "#d9a441"
TEXT = "#2a2a2e"
MUTED = "#6b6b73"
DARK = "#14161a"
FONT = "Segoe UI"

STATUS_FG = {C.CORRECT: "#1d7a44", C.NEEDS: "#b36b00", C.INCORRECT: "#c0392b", C.MISSED: "#8e2a2a",
             C.EXTRA: "#6b6b73", C.UNRECOGNISED: "#6b6b73"}
STATUS_BG = {C.CORRECT: "#e3f5ea", C.NEEDS: "#fff2d9", C.INCORRECT: "#fde3e3", C.MISSED: "#f6d5d5",
             C.EXTRA: "#efefef", C.UNRECOGNISED: "#efefef"}


def apply_theme(root: tk.Misc) -> None:
    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass
    root.configure(bg=BG)
    st.configure(".", font=(FONT, 10), background=BG, foreground=TEXT)
    st.configure("Page.TFrame", background=BG)
    st.configure("Card.TFrame", background=CARD, relief="flat")
    st.configure("TLabel", background=BG, foreground=TEXT)
    st.configure("Card.TLabel", background=CARD, foreground=TEXT)
    st.configure("Title.TLabel", font=(FONT, 20, "bold"), foreground=ACCENT)
    st.configure("Sub.TLabel", foreground=MUTED)
    st.configure("CardSub.TLabel", background=CARD, foreground=MUTED)
    st.configure("CardTitle.TLabel", background=CARD, font=(FONT, 11, "bold"))
    st.configure("Big.TLabel", background=CARD, font=(FONT, 22, "bold"), foreground=ACCENT)
    st.configure("Stat.TLabel", background=CARD, font=(FONT, 24, "bold"), foreground=ACCENT)
    st.configure("TButton", padding=(12, 6), background="#e7e0d8", foreground=TEXT, borderwidth=0)
    st.map("TButton", background=[("active", "#d9d0c5"), ("disabled", "#eeeeee")])
    st.configure("Accent.TButton", background=ACCENT, foreground="white", font=(FONT, 10, "bold"))
    st.map("Accent.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#c9b3bb")])
    st.configure("Danger.TButton", background="#c0392b", foreground="white")
    st.map("Danger.TButton", background=[("active", "#a93226")])
    st.configure("TNotebook", background=BG, borderwidth=0)
    st.configure("TNotebook.Tab", padding=(14, 7), font=(FONT, 10))
    st.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", ACCENT)])
    st.configure("Treeview", rowheight=26, background=CARD, fieldbackground=CARD, borderwidth=0)
    st.configure("Treeview.Heading", font=(FONT, 10, "bold"), background="#ece6de", foreground=TEXT)
    st.map("Treeview", background=[("selected", "#e8d3da")], foreground=[("selected", TEXT)])
    st.configure("Horizontal.TProgressbar", troughcolor="#e7e0d8", background=ACCENT, thickness=14)
    st.configure("TCheckbutton", background=BG)
    st.configure("TRadiobutton", background=BG)
    st.configure("Card.TCheckbutton", background=CARD)
