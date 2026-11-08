"""Small helpers for cameras, images, videos and time formatting (Windows-path safe)."""
from __future__ import annotations

import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def open_camera(index: int = 0, width: int = 1280, height: int = 720):
    """Open a webcam. DirectShow opens much faster on Windows."""
    cap = None
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera {index}. Close other apps that use the webcam, "
            "check the Windows camera privacy settings, or try another camera index in Settings."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def open_video(path):
    """Return (capture, cleanup). Falls back to an ASCII temp copy for odd Unicode paths."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, (lambda: None)
    cap.release()
    tmp = Path(tempfile.gettempdir()) / f"nv_{uuid.uuid4().hex}{path.suffix}"
    try:
        shutil.copy2(path, tmp)
    except OSError as exc:
        raise ValueError(f"Cannot read the video file: {exc}") from exc
    cap = cv2.VideoCapture(str(tmp))
    if not cap.isOpened():
        cap.release()
        tmp.unlink(missing_ok=True)
        raise ValueError(
            "OpenCV could not decode this video. Convert it to H.264 MP4 "
            "(for example with HandBrake or VLC) and load it again."
        )

    def cleanup():
        tmp.unlink(missing_ok=True)

    return cap, cleanup


def read_image(path) -> np.ndarray:
    """Read an image as BGR (handles Unicode paths and EXIF rotation)."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cannot open image '{Path(path).name}': {exc}") from exc


def resize_max_width(frame: np.ndarray, max_w: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    s = max_w / w
    return cv2.resize(frame, (max_w, int(h * s)), interpolation=cv2.INTER_AREA)


def fmt_time(sec, decimals: bool = False) -> str:
    if sec is None:
        return "-"
    sec = max(0.0, float(sec))
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:04.1f}" if decimals else f"{int(m):02d}:{int(s):02d}"
