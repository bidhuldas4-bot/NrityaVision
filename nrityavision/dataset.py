"""Turning images / videos into labelled landmark samples for the training dataset."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import cv2

from . import config as C
from .media import open_video, read_image, resize_max_width


def best_usable(obs):
    usable = [o for o in obs if o.usable]
    return max(usable, key=lambda o: o.palm_px) if usable else None


def sample_from_image(ctx, path):
    """Returns (canon, hand_label, None) or (None, None, reason)."""
    img = resize_max_width(read_image(path), 1600)
    tr = ctx.tracker(static=True, max_hands=2)
    try:
        obs = tr.process(img, mirrored=ctx.S()["mirror_uploads"])
    finally:
        tr.close()
    if not obs:
        return None, None, "no hand detected"
    best = best_usable(obs)
    if best is None:
        return None, None, "; ".join(obs[0].reasons)
    return best.canon, best.label, None


def samples_from_video(ctx, path, progress=None, cancel=None, fps: float = 4.0, max_samples: int = 400):
    """Returns list of (canon, hand_label) taken from the largest usable hand at ~fps frames/second."""
    S = ctx.S()
    cap, cleanup = open_video(path)
    tr = ctx.tracker(static=False, max_hands=2)
    out = []
    try:
        vfps = cap.get(cv2.CAP_PROP_FPS)
        vfps = vfps if 1 <= vfps <= 240 else 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(vfps / fps)))
        i = 0
        while len(out) < max_samples:
            if cancel is not None and cancel.is_set():
                break
            if i % step:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            obs = tr.process(resize_max_width(frame, 960), mirrored=S["mirror_uploads"])
            if len(obs) <= 2:
                b = best_usable(obs)
                if b is not None:
                    out.append((b.canon, b.label))
            if progress and total:
                progress(min(i / total, 1.0), f"Extracting samples: {len(out)} found")
    finally:
        tr.close()
        cap.release()
        cleanup()
    return out


def store_reference_file(src, mudra_name: str) -> Path:
    """Copy a reference file into data/reference/<mudra>/ with a unique name."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (mudra_name or "sequences"))
    folder = C.REFERENCE_DIR / safe
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"{uuid.uuid4().hex[:8]}_{Path(src).name}"
    shutil.copy2(src, dst)
    return dst
