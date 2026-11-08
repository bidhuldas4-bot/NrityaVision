"""Central configuration: folders, default settings and the built-in mudra lists."""
from __future__ import annotations

from pathlib import Path

APP_NAME = "NrityaVision AI"
APP_VERSION = "1.0.0"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REFERENCE_DIR = DATA_DIR / "reference"
REPORTS_DIR = DATA_DIR / "reports"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = DATA_DIR / "logs"
DB_PATH = DATA_DIR / "nrityavision.db"
MODEL_PATH = MODELS_DIR / "mudra_model.joblib"

for _d in (DATA_DIR, MODELS_DIR, REFERENCE_DIR, REPORTS_DIR, SNAPSHOT_DIR, CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

# Status vocabulary used everywhere (database, UI, reports)
CORRECT = "Correct"
NEEDS = "Needs improvement"
INCORRECT = "Incorrect"
MISSED = "Missed"
EXTRA = "Extra"
UNRECOGNISED = "Unable to recognize"

# Every setting can be changed in the Settings page; the type of the default decides the cast.
DEFAULT_SETTINGS = {
    "camera_index": 0,
    "det_conf": 0.6,            # MediaPipe min detection confidence
    "track_conf": 0.5,          # MediaPipe min tracking confidence
    "model_complexity": 1,      # 0 = faster, 1 = more accurate
    "conf_threshold": 0.65,     # below this classifier probability -> "Unable to recognize"
    "margin_threshold": 0.15,   # top-1 minus top-2 probability must exceed this
    "ood_factor": 1.8,          # reject hands far from every known mudra (0 = disabled)
    "sim_correct": 75.0,        # similarity >= this -> Correct
    "sim_needs": 55.0,          # similarity >= this -> Needs improvement, else Incorrect
    "sim_tau": 2.0,             # tolerance of the similarity function (larger = more lenient)
    "analysis_fps": 10.0,       # frames per second sampled from videos
    "min_segment_sec": 0.4,     # shorter holds are treated as transitions
    "gap_tol_sec": 0.35,        # tracking dropouts shorter than this are bridged
    "unclear_min_sec": 1.0,     # unclear stretches shorter than this are not reported
    "hand_mode": "Either",      # Either / Left / Right
    "multi_policy": "skip",     # skip / largest  (frames with more than two hands)
    "mirror_uploads": False,    # True if uploaded media was recorded as a mirrored selfie
    "blur_threshold": 12.0,
    "min_brightness": 40.0,
    "max_brightness": 235.0,
    "min_palm_px": 28.0,
    "current_user_id": 1,
    "fullscreen": False,
}

# (name, English meaning, description). Descriptions follow the common textual definitions;
# schools and gurus differ slightly, so always use your own teacher's reference images.
INITIAL_MUDRAS = [
    ("Pataka", "Flag", "Fingers extended and together; thumb bent against the palm."),
    ("Tripataka", "Three-part flag", "Like Pataka, but the ring finger is bent."),
    ("Ardhapataka", "Half flag", "Index and middle fingers extended; ring and little fingers bent; thumb bent."),
    ("Mushti", "Fist", "All fingers folded into the palm with the thumb over them."),
    ("Mayura", "Peacock", "Thumb tip touches the ring fingertip; the other fingers stay extended."),
]

ASAMYUTA_HASTAS = [
    "Pataka", "Tripataka", "Ardhapataka", "Kartarimukha", "Mayura", "Ardhachandra", "Arala",
    "Shukatunda", "Mushti", "Shikhara", "Kapitha", "Katakamukha", "Suchi", "Chandrakala",
    "Padmakosha", "Sarpashirsha", "Mrigashirsha", "Simhamukha", "Kangula", "Alapadma",
    "Chatura", "Bhramara", "Hamsasya", "Hamsapaksha", "Sandamsha", "Mukula", "Tamrachuda", "Trishula",
]
