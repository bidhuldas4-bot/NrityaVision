"""Mudra classifier (scikit-learn) with confidence, margin and out-of-distribution rejection.

Training compares several model families by cross-validation and keeps the best one.
Cross-validation is grouped by capture session whenever there are enough sessions, because
consecutive webcam frames are near-duplicates and plain random splits would overstate accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import MODEL_PATH

MIN_PER_CLASS = 10


@dataclass
class Prediction:
    label: Optional[str]          # None => "Unable to recognize"
    top_label: Optional[str]      # closest class (diagnostic only; NOT accepted when label is None)
    confidence: float
    margin: float
    reason: str = ""
    probs: dict = field(default_factory=dict)


def _candidates() -> dict:
    sc = lambda: StandardScaler()  # noqa: E731
    return {
        "SVM (RBF)": Pipeline([("sc", sc()), ("m", SVC(C=10, gamma="scale", probability=True,
                                                       class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("sc", sc()), ("m", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1))]),
        "Logistic Regression": Pipeline([("sc", sc()), ("m", LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced"))]),
    }


class MudraClassifier:
    def __init__(self, path=MODEL_PATH):
        self.path = path
        self.pipe = None
        self.classes: list[str] = []
        self.meta: dict = {}
        self._ood_scaler = None
        self._centroids: dict = {}
        self._ood_p99: dict = {}

    @property
    def ready(self) -> bool:
        return self.pipe is not None and len(self.classes) >= 2

    @property
    def stamp(self) -> str:
        return self.meta.get("trained_at", "none")

    def load(self) -> bool:
        self.pipe = None
        self.classes, self.meta = [], {}
        if not self.path.exists():
            return False
        try:
            blob = joblib.load(self.path)
            self.pipe = blob["pipe"]
            self.classes = list(blob["classes"])
            self.meta = blob["meta"]
            self._ood_scaler = blob["ood_scaler"]
            self._centroids = blob["centroids"]
            self._ood_p99 = blob["ood_p99"]
            return True
        except Exception as exc:  # noqa: BLE001  (corrupt / version mismatch)
            self.pipe = None
            self.meta = {"load_error": str(exc)}
            return False

    # ------------------------------------------------------------------ training
    def train(self, X: np.ndarray, y: list[str], groups: Optional[list[str]] = None,
              log: Callable[[str], None] = print) -> dict:
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        dropped = {c: int(n) for c, n in zip(classes, counts) if n < MIN_PER_CLASS}
        keep = [c for c, n in zip(classes, counts) if n >= MIN_PER_CLASS]
        for c, n in dropped.items():
            log(f"Skipping '{c}': only {n} samples (need at least {MIN_PER_CLASS}).")
        if len(keep) < 2:
            raise ValueError(f"Need at least 2 mudras with {MIN_PER_CLASS}+ samples each to train. "
                             "Collect more samples in the Reference Library.")
        mask = np.isin(y, keep)
        X, y = X[mask], y[mask]
        g = np.asarray(groups)[mask] if groups is not None else None
        per_class = {c: int((y == c).sum()) for c in keep}
        log(f"Training on {len(y)} samples from {len(keep)} mudras: {per_class}")

        cv, optimistic, cv_desc = self._make_cv(y, g)
        log(f"Validation: {cv_desc}")
        results = {}
        best_name, best_f1, best_pred = None, -1.0, None
        for name, model in _candidates().items():
            try:
                if g is not None and not optimistic:
                    pred = cross_val_predict(model, X, y, cv=cv, groups=g)
                else:
                    pred = cross_val_predict(model, X, y, cv=cv)
            except Exception as exc:  # noqa: BLE001
                log(f"  {name}: failed ({exc})")
                continue
            acc = accuracy_score(y, pred)
            f1 = f1_score(y, pred, average="macro", zero_division=0)
            results[name] = {"accuracy": float(acc), "macro_f1": float(f1)}
            log(f"  {name}: accuracy {acc:.3f}, macro-F1 {f1:.3f}")
            if f1 > best_f1:
                best_name, best_f1, best_pred = name, f1, pred
        if best_name is None:
            raise RuntimeError("All candidate models failed during validation.")
        log(f"Selected model: {best_name}")

        final = _candidates()[best_name]
        final.fit(X, y)
        self.pipe, self.classes = final, list(final.classes_)

        ood = StandardScaler().fit(X)
        Z = ood.transform(X)
        dim = Z.shape[1]
        self._ood_scaler, self._centroids, self._ood_p99 = ood, {}, {}
        for c in self.classes:
            Zc = Z[y == c]
            cen = Zc.mean(axis=0)
            d = np.linalg.norm(Zc - cen, axis=1) / math.sqrt(dim)
            self._centroids[c] = cen
            self._ood_p99[c] = float(np.percentile(d, 99))

        report = classification_report(y, best_pred, labels=self.classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, best_pred, labels=self.classes)
        self.meta = {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": best_name, "n_samples": int(len(y)), "per_class": per_class,
            "dropped_classes": dropped, "cv_description": cv_desc, "optimistic": bool(optimistic),
            "cv_accuracy": float(accuracy_score(y, best_pred)), "cv_macro_f1": float(best_f1),
            "candidates": results, "report": report, "confusion": cm.tolist(), "labels": self.classes,
        }
        joblib.dump({"pipe": self.pipe, "classes": self.classes, "meta": self.meta,
                     "ood_scaler": self._ood_scaler, "centroids": self._centroids,
                     "ood_p99": self._ood_p99}, self.path)
        log(f"Saved model to {self.path}")
        if optimistic:
            log("WARNING: validation was not grouped by session, so the accuracy above is optimistic. "
                "Record each mudra in several separate sessions (different days, distances, people) "
                "for a trustworthy estimate.")
        return self.meta

    @staticmethod
    def _make_cv(y, groups):
        classes = np.unique(y)
        min_count = min(int((y == c).sum()) for c in classes)
        if groups is not None:
            n_groups = min(len({gg for gg, yy in zip(groups, y) if yy == c}) for c in classes)
            if n_groups >= 3:
                n = min(5, n_groups)
                return (StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=42), False,
                        f"{n}-fold cross-validation grouped by capture session")
        n = max(2, min(5, min_count))
        return (StratifiedKFold(n_splits=n, shuffle=True, random_state=42), True,
                f"{n}-fold stratified cross-validation (sessions too few to group - optimistic)")

    # ---------------------------------------------------------------- prediction
    def predict(self, feat: np.ndarray, conf_thr: float = 0.65, margin_thr: float = 0.15,
                ood_factor: float = 1.8) -> Prediction:
        if not self.ready:
            return Prediction(None, None, 0.0, 0.0, "no trained model")
        x = np.asarray(feat, dtype=np.float64).reshape(1, -1)
        proba = self.pipe.predict_proba(x)[0]
        order = np.argsort(proba)[::-1]
        top = self.classes[order[0]]
        conf = float(proba[order[0]])
        margin = conf - float(proba[order[1]]) if len(order) > 1 else conf
        probs = {c: float(p) for c, p in zip(self.classes, proba)}
        reason = ""
        if conf < conf_thr:
            reason = f"low confidence ({conf:.0%})"
        elif margin < margin_thr:
            reason = "ambiguous between two mudras"
        elif ood_factor and ood_factor > 0:
            z = self._ood_scaler.transform(x)[0]
            d = np.linalg.norm(z - self._centroids[top]) / math.sqrt(z.size)
            if d > max(self._ood_p99[top] * ood_factor, 0.25):
                reason = "hand shape unlike any known mudra"
        return Prediction(None if reason else top, top, conf, margin, reason, probs)
