"""Reference profiles and the geometric similarity score.

Similarity is NOT the classifier's confidence. It measures how close a hand's measured geometry
(joint angles, distances, finger spread, palm orientation) is to the mean of the labelled
reference samples, using a per-feature tolerance:

    tolerance_i = sqrt(reference_std_i^2 + floor_i^2)         (floor = minimum tolerance)
    d           = sqrt( sum_g w_g * mean_i_in_g( ((x_i - mean_i) / tolerance_i)^2 ) )
    similarity  = 100 * exp(-(d / tau)^2)                      (tau = 2.0 by default)

It is only as good as the reference samples behind it: have your teacher's reference images
validated, and capture several samples per mudra.
"""
from __future__ import annotations

import math

import numpy as np

from .features import (GROUP_SLICES, GROUP_WEIGHTS, explain_differences, extract_features,
                       floor_vector)


class ReferenceProfiles:
    def __init__(self, tau: float = 2.0):
        self.tau = tau
        self.profiles: dict[str, dict] = {}
        self._floor = floor_vector()

    def rebuild(self, db) -> None:
        self.profiles = {}
        for name, lms in db.samples_by_mudra().items():
            if len(lms) == 0:
                continue
            F = np.array([extract_features(l.reshape(21, 3)) for l in lms], dtype=np.float64)
            self.profiles[name] = {"mean": F.mean(axis=0),
                                   "std": F.std(axis=0) if len(F) > 1 else np.zeros(F.shape[1]),
                                   "n": len(F)}

    def has(self, name) -> bool:
        return name in self.profiles

    def count(self, name) -> int:
        return self.profiles.get(name, {}).get("n", 0)

    def mean(self, name) -> np.ndarray:
        return self.profiles[name]["mean"]

    def distance(self, name: str, feat: np.ndarray) -> float:
        p = self.profiles[name]
        tol = np.minimum(np.sqrt(p["std"] ** 2 + self._floor ** 2), 3.0 * self._floor)
        z = (np.asarray(feat, dtype=np.float64) - p["mean"]) / tol
        num = sum(GROUP_WEIGHTS[g] * float(np.mean(z[sl] ** 2)) for g, sl in GROUP_SLICES.items())
        return math.sqrt(num / sum(GROUP_WEIGHTS.values()))

    def similarity(self, name: str, feat: np.ndarray) -> float:
        d = self.distance(name, feat)
        return 100.0 * math.exp(-((d / self.tau) ** 2))

    def explain(self, name: str, feat: np.ndarray) -> list[str]:
        return explain_differences(feat, self.mean(name))
