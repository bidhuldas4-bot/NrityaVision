"""MediaPipe hand tracking with per-hand quality checks and drawing helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .features import canonicalize, palm_size

try:
    import mediapipe as mp
    _mp_hands = mp.solutions.hands
    HAND_CONNECTIONS = list(_mp_hands.HAND_CONNECTIONS)
except Exception as exc:  # noqa: BLE001
    mp = None
    _mp_hands = None
    HAND_CONNECTIONS = []
    _MP_ERROR = exc


@dataclass
class HandObservation:
    label: str                    # the person's actual hand: "Left" or "Right"
    score: float                  # MediaPipe handedness confidence
    px: np.ndarray                # (21,3) landmarks in pixel units (z scaled like x)
    norm: np.ndarray              # (21,3) MediaPipe normalised landmarks
    canon: np.ndarray             # (21,3) wrist-centred, palm-scaled, left hands mirrored
    palm_px: float = 0.0
    bbox: tuple = (0, 0, 0, 0)
    blur: float = 0.0
    brightness: float = 0.0
    usable: bool = True
    reasons: list = field(default_factory=list)


class HandTracker:
    def __init__(self, static: bool = False, max_hands: int = 2, det_conf: float = 0.6,
                 track_conf: float = 0.5, model_complexity: int = 1, quality: dict | None = None):
        if _mp_hands is None:
            raise RuntimeError(
                "MediaPipe could not be imported. Install the tested version with "
                "'pip install mediapipe==0.10.14' (Python 3.10-3.12). Original error: "
                f"{_MP_ERROR}"
            )
        self.q = {"blur_threshold": 12.0, "min_brightness": 40.0,
                  "max_brightness": 235.0, "min_palm_px": 28.0}
        if quality:
            self.q.update({k: quality[k] for k in self.q if k in quality})
        self._hands = _mp_hands.Hands(
            static_image_mode=static, max_num_hands=max_hands,
            model_complexity=int(model_complexity),
            min_detection_confidence=float(det_conf), min_tracking_confidence=float(track_conf))

    def close(self):
        try:
            self._hands.close()
        except Exception:  # noqa: BLE001
            pass

    def process(self, bgr: np.ndarray, mirrored: bool = False) -> list[HandObservation]:
        """Detect hands. ``mirrored`` = the image is a selfie-style mirror image.

        MediaPipe assumes mirrored input; for normal (non-mirrored) images the handedness
        label has to be swapped, which is done here.
        """
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        out: list[HandObservation] = []
        if not res.multi_hand_landmarks:
            return out
        for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
            cls = hd.classification[0]
            label = cls.label
            if not mirrored:
                label = "Left" if label == "Right" else "Right"
            norm = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float64)
            px = np.stack([norm[:, 0] * w, norm[:, 1] * h, norm[:, 2] * w], axis=1)
            obs = HandObservation(label=label, score=float(cls.score), px=px, norm=norm,
                                  canon=canonicalize(px, label), palm_px=palm_size(px))
            self._assess_quality(bgr, obs)
            out.append(obs)
        return out

    def _assess_quality(self, bgr: np.ndarray, obs: HandObservation) -> None:
        h, w = bgr.shape[:2]
        reasons = []
        n = obs.norm
        outside = int(((n[:, 0] < 0.005) | (n[:, 0] > 0.995) | (n[:, 1] < 0.005) | (n[:, 1] > 0.995)).sum())
        if outside >= 3:
            reasons.append("hand partly outside the frame")
        if obs.palm_px < self.q["min_palm_px"]:
            reasons.append("hand too small / too far from the camera")
        xs, ys = obs.px[:, 0], obs.px[:, 1]
        pad = 0.2 * max(xs.max() - xs.min(), ys.max() - ys.min())
        x0, x1 = int(max(0, xs.min() - pad)), int(min(w, xs.max() + pad))
        y0, y1 = int(max(0, ys.min() - pad)), int(min(h, ys.max() + pad))
        obs.bbox = (x0, y0, x1, y1)
        crop = bgr[y0:y1, x0:x1]
        if crop.size > 0 and (x1 - x0) > 8 and (y1 - y0) > 8:
            g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (128, 128))
            obs.blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
            obs.brightness = float(g.mean())
            if obs.blur < self.q["blur_threshold"]:
                reasons.append("hand is blurry / out of focus")
            if obs.brightness < self.q["min_brightness"]:
                reasons.append("lighting too dark")
            elif obs.brightness > self.q["max_brightness"]:
                reasons.append("hand over-exposed")
        else:
            reasons.append("hand region invalid")
        obs.reasons = reasons
        obs.usable = not reasons


def select_hands(obs: list[HandObservation]) -> tuple[list[HandObservation], bool]:
    """Keep the two biggest hands. Returns (hands, multiple_people_suspected)."""
    if len(obs) <= 2:
        return obs, False
    return sorted(obs, key=lambda o: o.palm_px, reverse=True)[:2], True


def draw_hand(frame: np.ndarray, obs: HandObservation, caption: str = "", color=(0, 200, 80)) -> None:
    pts = obs.px[:, :2].astype(int)
    line_col = color if obs.usable else (0, 140, 255)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), line_col, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(p), 4, line_col, 1, cv2.LINE_AA)
    if caption:
        x0, y0, _, _ = obs.bbox
        y = max(18, y0 - 8)
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x0, y - th - 6), (x0 + tw + 8, y + 4), (20, 20, 20), -1)
        cv2.putText(frame, caption, (x0 + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
