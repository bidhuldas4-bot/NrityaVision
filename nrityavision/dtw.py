"""Dynamic Time Warping used for (a) aligning detected mudra sequences with an expected
sequence and (b) comparing movement trajectories of a learner video and a reference video."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .features import GROUP_FLOORS, SL_ANGLES, SL_PAIRD, SL_SPREAD, SL_TIPD


def dtw(cost: np.ndarray, band: int | None = None) -> tuple[float, list[tuple[int, int]]]:
    """Classic DTW on a precomputed (n, m) cost matrix. Returns (total_cost, path of (i, j))."""
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    if n == 0 or m == 0:
        return 0.0, []
    if band is not None:
        band = max(band, abs(n - m)) + 1
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    C = cost.tolist()
    for i in range(1, n + 1):
        Di, Dp, Ci = D[i], D[i - 1], C[i - 1]
        if band is None:
            lo, hi = 1, m
        else:
            ctr = i * m / n
            lo, hi = max(1, int(ctr - band)), min(m, int(ctr + band) + 1)
        for j in range(lo, hi + 1):
            a, b, c = Dp[j], Di[j - 1], Dp[j - 1]
            best = a if a < b else b
            if c < best:
                best = c
            Di[j] = Ci[j - 1] + best
    i, j = n, m
    path = []
    while i > 1 or j > 1:
        path.append((i - 1, j - 1))
        _, i, j = min((D[i - 1][j - 1], i - 1, j - 1), (D[i - 1][j], i - 1, j), (D[i][j - 1], i, j - 1))
    path.append((0, 0))
    path.reverse()
    return D[n][m], path


# ---------------------------------------------------------------- movement comparison
@dataclass
class MovementResult:
    similarity: float
    mean_cost: float
    deviations: list = field(default_factory=list)   # [(t_start, t_end, mean_local_cost)]
    n_learner: int = 0
    n_reference: int = 0

    def to_dict(self):
        return {"similarity": self.similarity, "mean_cost": self.mean_cost,
                "deviations": self.deviations, "n_learner": self.n_learner, "n_reference": self.n_reference}


def _motion_features(F: np.ndarray) -> np.ndarray:
    """Shape-only features (angles, distances, spread) scaled by their tolerances."""
    cols = np.r_[np.arange(*SL_ANGLES.indices(F.shape[1])), np.arange(*SL_TIPD.indices(F.shape[1])),
                 np.arange(*SL_PAIRD.indices(F.shape[1])), np.arange(*SL_SPREAD.indices(F.shape[1]))]
    scale = np.concatenate([
        np.full(15, GROUP_FLOORS["angles"]), np.full(5, GROUP_FLOORS["tipd"]),
        np.full(10, GROUP_FLOORS["paird"]), np.full(4, GROUP_FLOORS["spread"])])
    return F[:, cols] / scale


def _downsample(t: np.ndarray, X: np.ndarray, max_len: int):
    if len(t) <= max_len:
        return t, X
    idx = np.linspace(0, len(t) - 1, max_len).astype(int)
    return t[idx], X[idx]


def movement_similarity(t_learner, F_learner, t_ref, F_ref, tau: float = 2.0,
                        max_len: int = 500, dev_threshold: float = 2.5,
                        dev_min_sec: float = 0.6) -> MovementResult | None:
    """Compare two per-frame feature trajectories with banded DTW.

    The per-step cost is the RMS of tolerance-scaled feature differences; ``similarity`` maps the
    mean warped cost c to 100*exp(-(c/tau)^2). Stretches where the learner's local cost stays
    above ``dev_threshold`` are returned as timestamps of movement deviation.
    """
    t_l, t_r = np.asarray(t_learner, float), np.asarray(t_ref, float)
    if len(t_l) < 5 or len(t_r) < 5:
        return None
    A = _motion_features(np.asarray(F_learner, float))
    B = _motion_features(np.asarray(F_ref, float))
    t_l, A = _downsample(t_l, A, max_len)
    t_r, B = _downsample(t_r, B, max_len)
    diff = A[:, None, :] - B[None, :, :]
    cost = np.sqrt((diff ** 2).mean(axis=2))
    _, path = dtw(cost, band=int(0.3 * max(len(A), len(B))))
    local = np.zeros(len(A))
    cnt = np.zeros(len(A))
    for i, j in path:
        local[i] += cost[i, j]
        cnt[i] += 1
    local = local / np.maximum(cnt, 1)
    mean_cost = float(np.mean([cost[i, j] for i, j in path]))
    sim = 100.0 * math.exp(-((mean_cost / tau) ** 2))
    devs, start, last = [], None, None
    dt = float(np.median(np.diff(t_l))) if len(t_l) > 1 else 0.1
    for k in range(len(t_l)):
        if local[k] > dev_threshold:
            if start is None:
                start = k
            last = k
        elif start is not None:
            if t_l[last] - t_l[start] + dt >= dev_min_sec:
                devs.append((float(t_l[start]), float(t_l[last] + dt), float(local[start:last + 1].mean())))
            start = None
    if start is not None and t_l[last] - t_l[start] + dt >= dev_min_sec:
        devs.append((float(t_l[start]), float(t_l[last] + dt), float(local[start:last + 1].mean())))
    return MovementResult(sim, mean_cost, devs, len(A), len(B))
