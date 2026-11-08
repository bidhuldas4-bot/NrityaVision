"""Landmark normalisation and feature engineering.

Pipeline for every detected hand
--------------------------------
1. ``canonicalize``: pixel-space landmarks -> translated to the wrist, left hands mirrored so a
   single model serves both hands, and scaled by palm size (removes camera distance / hand size).
2. ``extract_features``: the hand is expressed in its own coordinate frame (rotation invariant),
   then joint angles, distances, finger spreads and the palm orientation are computed.

Feature layout (102 values) is described by the SL_* slices below.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

FINGERS = {
    "Thumb": (1, 2, 3, 4),
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Little": (17, 18, 19, 20),
}
FINGER_NAMES = list(FINGERS)
TIPS = (4, 8, 12, 16, 20)
_TIP_PAIRS = list(itertools.combinations(TIPS, 2))

SL_COORDS = slice(0, 63)     # hand-frame coordinates of the 21 landmarks
SL_ANGLES = slice(63, 78)    # 3 joint angles x 5 fingers (1.0 = straight)
SL_TIPD = slice(78, 83)      # wrist -> fingertip distances
SL_PAIRD = slice(83, 93)     # fingertip <-> fingertip distances (10 pairs)
SL_SPREAD = slice(93, 97)    # angle between neighbouring fingers
SL_ORIENT = slice(97, 102)   # palm normal (3) + hand direction in image (2)
N_FEATURES = 102

GROUP_SLICES = {
    "coords": SL_COORDS, "angles": SL_ANGLES, "tipd": SL_TIPD,
    "paird": SL_PAIRD, "spread": SL_SPREAD, "orient": SL_ORIENT,
}
# Relative importance of every feature group in the similarity score
GROUP_WEIGHTS = {"coords": 0.20, "angles": 0.35, "tipd": 0.10, "paird": 0.10, "spread": 0.10, "orient": 0.15}
# Minimum tolerance (in feature units) so that a single reference sample is not "too strict"
GROUP_FLOORS = {"coords": 0.20, "angles": 0.10, "tipd": 0.20, "paird": 0.20, "spread": 0.08, "orient": 0.30}


def floor_vector() -> np.ndarray:
    v = np.zeros(N_FEATURES)
    for g, sl in GROUP_SLICES.items():
        v[sl] = GROUP_FLOORS[g]
    return v


def palm_size(p: np.ndarray) -> float:
    pairs = ((0, 5), (0, 9), (0, 13), (0, 17), (5, 17))
    return float(np.mean([np.linalg.norm(p[i] - p[j]) for i, j in pairs]))


def canonicalize(px: np.ndarray, hand_label: str) -> np.ndarray:
    """Pixel landmarks (21,3) -> wrist-centred, palm-scaled, left hands mirrored to look right."""
    p = np.asarray(px, dtype=np.float64).copy()
    p -= p[0]
    if hand_label == "Left":
        p[:, 0] *= -1.0
    return p / max(palm_size(p), 1e-6)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


def local_frame(c: np.ndarray) -> np.ndarray:
    """Rows = x, y, z axes of the hand (y: wrist->middle knuckle, x: index->little knuckle)."""
    y = _unit(c[9] - c[0])
    v = c[17] - c[5]
    x = _unit(v - np.dot(v, y) * y)
    if not x.any() or not y.any():
        return np.eye(3)
    z = np.cross(x, y)
    return np.vstack([x, y, z])


def _vec_angle(u: np.ndarray, v: np.ndarray) -> float:
    n = np.linalg.norm(u) * np.linalg.norm(v)
    if n < 1e-9:
        return 0.0
    return math.acos(max(-1.0, min(1.0, float(np.dot(u, v) / n))))


def _joint_angle(a, b, c) -> float:
    return _vec_angle(a - b, c - b)


def extract_features(canon: np.ndarray) -> np.ndarray:
    c = np.asarray(canon, dtype=np.float64).reshape(21, 3)
    R = local_frame(c)
    loc = c @ R.T
    ang = []
    for ch in FINGERS.values():
        for a, b, d in ((0, ch[0], ch[1]), (ch[0], ch[1], ch[2]), (ch[1], ch[2], ch[3])):
            ang.append(_joint_angle(loc[a], loc[b], loc[d]) / math.pi)
    tipd = [np.linalg.norm(loc[t] - loc[0]) for t in TIPS]
    paird = [np.linalg.norm(loc[a] - loc[b]) for a, b in _TIP_PAIRS]
    dirs = [loc[4] - loc[1], loc[8] - loc[5], loc[12] - loc[9], loc[16] - loc[13], loc[20] - loc[17]]
    spread = [_vec_angle(dirs[i], dirs[i + 1]) / math.pi for i in range(4)]
    orient = list(R[2]) + list(R[1][:2])
    return np.concatenate([loc.ravel(), ang, tipd, paird, spread, orient]).astype(np.float32)


def feature_matrix(canon_array: np.ndarray) -> np.ndarray:
    """(N,63) canonical landmarks -> (N,102) features."""
    return np.array([extract_features(r.reshape(21, 3)) for r in canon_array], dtype=np.float32)


def descriptors(feat: np.ndarray) -> dict:
    a = np.asarray(feat[SL_ANGLES]).reshape(5, 3)
    curl = np.clip((1.0 - a).mean(axis=1) / 0.5, 0.0, 1.0)  # 0 = straight, 1 = tightly curled
    return {
        "curl": dict(zip(FINGER_NAMES, curl.tolist())),
        "spread_deg": (np.asarray(feat[SL_SPREAD]) * 180.0).tolist(),
        "normal": np.asarray(feat[97:100]),
        "dir": np.asarray(feat[100:102]),
    }


_SPREAD_NAMES = [("thumb", "index finger"), ("index finger", "middle finger"),
                 ("middle finger", "ring finger"), ("ring finger", "little finger")]


def explain_differences(user: np.ndarray, ref: np.ndarray, max_tips: int = 3) -> list[str]:
    """Plain-language corrections derived from measured differences to the reference mean."""
    du, dr = descriptors(user), descriptors(ref)
    tips: list[tuple[float, str]] = []
    for f in FINGER_NAMES:
        diff = du["curl"][f] - dr["curl"][f]
        if abs(diff) > 0.22:
            if diff > 0:
                tips.append((abs(diff), f"{f} finger: straighten it more (it is more bent than the reference)."))
            else:
                tips.append((abs(diff), f"{f} finger: bend it more (it is straighter than the reference)."))
    for (a, b), u, r in zip(_SPREAD_NAMES, du["spread_deg"], dr["spread_deg"]):
        d = u - r
        if abs(d) > 14:
            word = "Bring the" if d > 0 else "Open the"
            tail = "closer together." if d > 0 else "slightly wider apart."
            tips.append((abs(d) / 60.0, f"{word} {a} and {b} {tail}"))
    nu, nr = du["normal"], dr["normal"]
    ang = math.degrees(_vec_angle(nu, nr))
    if ang > 45:
        tips.append((ang / 90.0, f"Palm orientation differs from the reference by about {ang:.0f} degrees; "
                                 "check how your palm faces the camera."))
    tilt = math.degrees(_vec_angle(np.append(du["dir"], 0), np.append(dr["dir"], 0)))
    if tilt > 35:
        tips.append((tilt / 90.0, f"Hand tilt differs from the reference by about {tilt:.0f} degrees."))
    tips.sort(key=lambda t: -t[0])
    return [t[1] for t in tips[:max_tips]]
