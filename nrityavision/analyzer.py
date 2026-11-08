"""Frame, image and video analysis plus the performance-evaluation logic.

Evaluation rules (documented in the PDF report as well)
-------------------------------------------------------
* A hand is only classified when it passes quality checks (visible, sharp, lit, big enough) and the
  classifier is confident, unambiguous and not an outlier; otherwise it is "Unable to recognize".
* Consecutive recognised frames become mudra *segments*; holds shorter than ``min_segment_sec``
  are treated as transitions and ignored.
* Item score = geometric similarity (0-100) to the reference profile of the target mudra
  (expected mudra if a sequence was supplied, otherwise the detected mudra). A wrong mudra or a
  missed one scores 0. Overall accuracy = mean item score.
* Status: similarity >= sim_correct -> Correct; >= sim_needs -> Needs improvement; else Incorrect.
* Expected sequences are aligned to the detected segments with Dynamic Time Warping.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import config as C
from .classifier import MudraClassifier
from .config import CACHE_DIR
from .dtw import dtw, movement_similarity
from .features import extract_features
from .hand_tracker import HandObservation, HandTracker, draw_hand, select_hands
from .media import open_video, read_image, resize_max_width
from .similarity import ReferenceProfiles

UNVERIFIED = "Unverified (no reference samples)"
SCORED = {C.CORRECT, C.NEEDS, C.INCORRECT, C.MISSED}
STATUS_BGR = {C.CORRECT: (80, 175, 76), C.NEEDS: (0, 170, 255), C.INCORRECT: (60, 60, 230),
              C.UNRECOGNISED: (160, 160, 160)}


class AnalysisCancelled(Exception):
    pass


# =============================================================== single-hand assessment
@dataclass
class HandAssessment:
    hand: str
    usable: bool
    reasons: list
    label: Optional[str] = None
    top_label: Optional[str] = None
    confidence: float = 0.0
    margin: float = 0.0
    reason: str = ""
    similarity: Optional[float] = None
    sim_name: Optional[str] = None
    status: str = C.UNRECOGNISED
    tips: list = field(default_factory=list)
    feat: Optional[np.ndarray] = None
    probs: dict = field(default_factory=dict)

    @property
    def caption(self) -> str:
        if self.label:
            return f"{self.label} {self.confidence:.0%}"
        return C.UNRECOGNISED


def status_from_similarity(sim: Optional[float], S: dict) -> str:
    if sim is None:
        return UNVERIFIED
    if sim >= S["sim_correct"]:
        return C.CORRECT
    if sim >= S["sim_needs"]:
        return C.NEEDS
    return C.INCORRECT


class FrameAnalyzer:
    def __init__(self, classifier: MudraClassifier, profiles: ReferenceProfiles, S: dict):
        self.clf, self.profiles, self.S = classifier, profiles, S

    def assess(self, obs: HandObservation, target: Optional[str] = None) -> HandAssessment:
        S = self.S
        a = HandAssessment(hand=obs.label, usable=obs.usable, reasons=list(obs.reasons))
        if not obs.usable:
            a.reason = "; ".join(obs.reasons)
            return a
        feat = extract_features(obs.canon)
        a.feat = feat
        if self.clf.ready:
            p = self.clf.predict(feat, S["conf_threshold"], S["margin_threshold"], S["ood_factor"])
            a.label, a.top_label, a.confidence, a.margin = p.label, p.top_label, p.confidence, p.margin
            a.reason, a.probs = p.reason, p.probs
        else:
            a.reason = "no trained model yet"
        name = target or a.label
        if name and self.profiles.has(name):
            a.sim_name = name
            a.similarity = self.profiles.similarity(name, feat)
        if a.label is None:
            a.status = C.UNRECOGNISED
        elif target and a.label != target:
            a.status = C.INCORRECT
        else:
            a.status = status_from_similarity(a.similarity, S)
        if a.sim_name and a.status in (C.NEEDS, C.INCORRECT) or (target and a.sim_name and a.label is None):
            a.tips = self.profiles.explain(a.sim_name, feat)
        return a


# ====================================================================== video records
@dataclass
class FrameRecord:
    t: float
    status: str                         # ok | unknown | unusable | nohand | multi
    label: Optional[str] = None
    conf: float = 0.0
    hand: str = ""
    feat: Optional[np.ndarray] = None
    reason: str = ""


@dataclass
class Collected:
    records: list
    fps: float
    duration: float
    frame_dt: float
    counters: Counter
    reasons: Counter
    notes: list


@dataclass
class Segment:
    label: str
    start: float
    end: float
    conf: float
    hand: str
    feats: np.ndarray
    _sims: dict = field(default_factory=dict)

    def sim_to(self, name: str, profiles: ReferenceProfiles) -> Optional[float]:
        if name not in self._sims:
            self._sims[name] = (float(np.median([profiles.similarity(name, f) for f in self.feats]))
                                if profiles.has(name) else None)
        return self._sims[name]

    @property
    def mean_feat(self) -> np.ndarray:
        return self.feats.mean(axis=0)


def smooth_labels(labels: list, win: int = 5) -> list:
    out, half = [], win // 2
    for i, own in enumerate(labels):
        c = Counter(labels[max(0, i - half): i + half + 1])
        top, cnt = c.most_common(1)[0]
        out.append(own if list(c.values()).count(cnt) > 1 else top)
    return out


def build_segments(records: list, min_dur: float, gap_tol: float, frame_dt: float) -> list[Segment]:
    sm = smooth_labels([r.label for r in records])
    runs, cur = [], None
    for i, (r, lab) in enumerate(zip(records, sm)):
        if lab is None:
            continue
        if cur and cur["label"] == lab and r.t - cur["last"] <= gap_tol + frame_dt * 1.01:
            cur["idx"].append(i)
            cur["last"] = r.t
        else:
            if cur:
                runs.append(cur)
            cur = {"label": lab, "idx": [i], "first": r.t, "last": r.t}
    if cur:
        runs.append(cur)
    segs = []
    for run in runs:
        if run["last"] - run["first"] + frame_dt < min_dur:
            continue
        fr = [records[i] for i in run["idx"] if records[i].label == run["label"] and records[i].feat is not None]
        if not fr:
            continue
        hand = Counter(f.hand for f in fr).most_common(1)[0][0]
        segs.append(Segment(run["label"], run["first"], run["last"] + frame_dt,
                            float(np.median([f.conf for f in fr])), hand, np.array([f.feat for f in fr])))
    return segs


def find_unclear_stretches(records: list, frame_dt: float, min_len: float, merge_gap: float = 0.5) -> list[dict]:
    """Stretches where a hand was visible but could not be recognised (blur, dark, transition...)."""
    runs, cur = [], None
    for r in records:
        if r.status in ("unknown", "unusable"):
            if cur and r.t - cur["end"] <= merge_gap + frame_dt * 1.01:
                cur["end"] = r.t
                cur["reasons"].append(r.reason or "unclear hand shape")
            else:
                if cur:
                    runs.append(cur)
                cur = {"start": r.t, "end": r.t, "reasons": [r.reason or "unclear hand shape"]}
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        if r["end"] - r["start"] + frame_dt >= min_len:
            out.append({"start": r["start"], "end": r["end"] + frame_dt,
                        "reason": Counter(r["reasons"]).most_common(1)[0][0]})
    return out


# ========================================================================= evaluation
def _default_tip(status: str) -> str:
    return {C.CORRECT: "Good form - keep this shape consistent.",
            C.NEEDS: "Close, but compare with the reference image and hold the shape more precisely.",
            C.INCORRECT: "Shape differs clearly from the reference; practise this mudra slowly in front of the reference.",
            }.get(status, "")


def evaluate_free(segs: list[Segment], profiles: ReferenceProfiles, S: dict) -> list[dict]:
    items = []
    for i, s in enumerate(segs):
        sim = s.sim_to(s.label, profiles)
        status = status_from_similarity(sim, S)
        tips = profiles.explain(s.label, s.mean_feat) if status in (C.NEEDS, C.INCORRECT) else []
        items.append(dict(idx=i + 1, start_s=s.start, end_s=s.end, hand=s.hand, expected="", detected=s.label,
                          similarity=sim, confidence=s.conf, status=status,
                          suggestion=" ".join(tips) or _default_tip(status),
                          score=sim if status in SCORED else None))
    return items


def evaluate_expected(segs: list[Segment], expected: list[str], profiles: ReferenceProfiles, S: dict) -> list[dict]:
    n, m = len(segs), len(expected)
    rows: list[tuple[float, int, dict]] = []
    seq = 0
    used: set[int] = set()
    if n:
        sims = np.zeros((n, m))
        cost = np.zeros((n, m))
        for i, s in enumerate(segs):
            for j, e in enumerate(expected):
                v = s.sim_to(e, profiles)
                sims[i, j] = v if v is not None else 0.0
                cost[i, j] = 0.6 * (s.label != e) + 0.4 * (1 - sims[i, j] / 100.0)
        _, path = dtw(cost)
        # Resolve the DTW alignment into one-to-one matches, cheapest (best) pairs first, so that a
        # skipped mudra does not steal the segment that belongs to the following expected mudra.
        match: dict[int, int] = {}
        taken: set[int] = set()
        for i, j in sorted(set(path), key=lambda p: (cost[p[0], p[1]], p[1])):
            if j not in match and i not in taken:
                match[j] = i
                taken.add(i)
    prev_end = 0.0
    for j, e in enumerate(expected):
        if not n or j not in match:
            rows.append((prev_end - 1e-3, seq, dict(start_s=prev_end, end_s=prev_end, hand="", expected=e,
                         detected="-", similarity=None, confidence=None, status=C.MISSED, score=0.0,
                         suggestion=f"{e} was not found at this point of the sequence. Make sure the hand is "
                                    "visible and the mudra is held for at least a moment.")))
            seq += 1
            continue
        i = match[j]
        used.add(i)
        s = segs[i]
        sim = s.sim_to(e, profiles)
        if s.label == e:
            status = status_from_similarity(sim, S)
            tips = profiles.explain(e, s.mean_feat) if status in (C.NEEDS, C.INCORRECT) else []
            sug, score = " ".join(tips) or _default_tip(status), (sim if status in SCORED else None)
        else:
            status, score = C.INCORRECT, 0.0
            tips = profiles.explain(e, s.mean_feat) if profiles.has(e) else []
            sug = f"You showed {s.label} where {e} was expected. " + " ".join(tips)
        rows.append((s.start, seq, dict(start_s=s.start, end_s=s.end, hand=s.hand, expected=e, detected=s.label,
                     similarity=sim, confidence=s.conf, status=status, suggestion=sug.strip(), score=score)))
        seq += 1
        prev_end = s.end
    for i, s in enumerate(segs):
        if i not in used:
            rows.append((s.start, seq, dict(start_s=s.start, end_s=s.end, hand=s.hand, expected="", detected=s.label,
                         similarity=None, confidence=s.conf, status=C.EXTRA, score=None,
                         suggestion="Extra movement that is not part of the expected sequence.")))
            seq += 1
    rows.sort(key=lambda r: (r[0], r[1]))
    items = []
    for k, (_, _, it) in enumerate(rows):
        it["idx"] = k + 1
        items.append(it)
    return items


def summarize(items: list[dict]) -> tuple[dict, dict]:
    cnt = Counter(it["status"] for it in items)
    scored = [it for it in items if it["status"] in SCORED and it.get("score") is not None]
    sims = [it["similarity"] for it in items if it["status"] in SCORED and it.get("similarity") is not None]
    summary = {
        "total_items": len([it for it in items if it["status"] not in (C.EXTRA, C.UNRECOGNISED)]),
        "segments_detected": len([it for it in items if it["detected"] not in ("", "-", C.UNRECOGNISED)]),
        "correct": cnt[C.CORRECT], "needs_improvement": cnt[C.NEEDS], "incorrect": cnt[C.INCORRECT],
        "missed": cnt[C.MISSED], "extra": cnt[C.EXTRA], "unrecognised": cnt[C.UNRECOGNISED],
        "overall_accuracy": float(np.mean([it["score"] for it in scored])) if scored else None,
        "mean_similarity": float(np.mean(sims)) if sims else None,
    }
    per: dict[str, dict] = {}
    for it in scored:
        key = it["expected"] or it["detected"]
        d = per.setdefault(key, {"n": 0, "correct": 0, "needs": 0, "incorrect": 0, "scores": []})
        d["n"] += 1
        d["scores"].append(it["score"])
        d[{C.CORRECT: "correct", C.NEEDS: "needs"}.get(it["status"], "incorrect")] += 1
    per_mudra = {k: {"n": d["n"], "correct": d["correct"], "needs": d["needs"], "incorrect": d["incorrect"],
                     "accuracy": float(np.mean(d["scores"]))} for k, d in per.items()}
    return summary, per_mudra


def mistakes_from_items(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        st = it["status"]
        sim = it.get("similarity")
        simtxt = f" (similarity {sim:.0f}%)" if sim is not None else ""
        if st == C.MISSED:
            desc, sev = f"Expected {it['expected']} was not found in the performance", "error"
        elif st == C.INCORRECT and it["expected"] and it["detected"] != it["expected"]:
            desc, sev = f"Expected {it['expected']} but detected {it['detected']}{simtxt}", "error"
        elif st == C.INCORRECT:
            desc, sev = f"{it['detected']} shape is clearly off{simtxt}", "error"
        elif st == C.NEEDS:
            desc, sev = f"{it['expected'] or it['detected']} needs improvement{simtxt}", "warning"
        else:
            continue
        out.append({"timestamp_s": it["start_s"], "end_s": it["end_s"], "severity": sev,
                    "description": desc, "suggestion": it.get("suggestion", "")})
    return out


def evaluate_records(col: Collected, S: dict, profiles: ReferenceProfiles,
                     expected: Optional[list[str]] = None) -> dict:
    segs = build_segments(col.records, S["min_segment_sec"], S["gap_tol_sec"], col.frame_dt)
    items = evaluate_expected(segs, expected, profiles, S) if expected else evaluate_free(segs, profiles, S)
    unclear = find_unclear_stretches(col.records, col.frame_dt, S["unclear_min_sec"])
    mistakes = mistakes_from_items(items)
    for u in unclear:
        mistakes.append({"timestamp_s": u["start"], "end_s": u["end"], "severity": "info",
                         "description": f"Unable to recognize the mudra ({u['reason']})",
                         "suggestion": "Check lighting, keep the hand fully in frame and hold the mudra steadily."})
    mistakes.sort(key=lambda m: m["timestamp_s"])
    summary, per_mudra = summarize(items)
    summary["unrecognised"] = len(unclear)
    n = max(col.counters["sampled"], 1)
    quality = {"frames_sampled": col.counters["sampled"], "hands_recognised_pct": 100.0 * col.counters["ok"] / n,
               "no_hand_pct": 100.0 * col.counters["nohand"] / n, "unclear_pct": 100.0 * (col.counters["unknown"] + col.counters["unusable"]) / n,
               "multi_person_pct": 100.0 * col.counters["multi"] / n, "reasons": dict(col.reasons.most_common(6))}
    notes = list(col.notes)
    if quality["multi_person_pct"] > 10:
        notes.append(f"{quality['multi_person_pct']:.0f}% of sampled frames contained more than two hands "
                     "(possible multiple dancers). These frames were "
                     + ("skipped." if S["multi_policy"] == "skip" else "analysed using the two largest hands.")
                     + " Crop the video to one dancer for reliable results.")
    if quality["no_hand_pct"] > 70:
        notes.append(f"Hands were visible in only {100 - quality['no_hand_pct']:.0f}% of sampled frames.")
    blur = sum(v for k, v in col.reasons.items() if "blurry" in k)
    if blur / n > 0.15:
        notes.append("Many frames were too blurry to analyse; use a steadier camera or higher shutter speed.")
    if sum(v for k, v in col.reasons.items() if "dark" in k) / n > 0.15:
        notes.append("Many frames were too dark; improve the lighting.")
    if sum(v for k, v in col.reasons.items() if "outside" in k) / n > 0.15:
        notes.append("The hands were often cut off by the frame edge.")
    names = {it["expected"] or it["detected"] for it in items if (it["expected"] or it["detected"]) not in ("", "-")}
    for nm in sorted(names):
        if profiles.has(nm) and profiles.count(nm) < 10:
            notes.append(f"Reference profile for {nm} uses only {profiles.count(nm)} samples; "
                         "add more samples for a more reliable similarity score.")
    for nm in sorted(set(expected or [])):
        if not profiles.has(nm):
            notes.append(f"No reference samples exist for {nm}, so its similarity could not be calculated.")
    if not items:
        notes.append("No mudra could be recognised with enough confidence in this video.")
    return {"items": items, "mistakes": mistakes, "summary": summary, "per_mudra": per_mudra,
            "quality": quality, "notes": notes}


# ======================================================================= video analyzer
ProgressCB = Callable[[float, str, Optional[np.ndarray]], None]


class VideoAnalyzer:
    def __init__(self, classifier: MudraClassifier, profiles: ReferenceProfiles, S: dict):
        self.clf, self.profiles, self.S = classifier, profiles, S

    def collect(self, path, progress: Optional[ProgressCB] = None, cancel: Optional[threading.Event] = None,
                hand_mode: Optional[str] = None, p0: float = 0.0, p1: float = 1.0) -> Collected:
        S = self.S
        hand_mode = hand_mode or S["hand_mode"]
        cap, cleanup = open_video(path)
        notes: list[str] = []
        tracker = None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if not (1.0 <= fps <= 240.0):
                fps = 25.0
                notes.append("Video frame-rate metadata was missing; 25 fps was assumed for timestamps.")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, int(round(fps / max(S["analysis_fps"], 1.0))))
            frame_dt = step / fps
            tracker = HandTracker(False, 4, S["det_conf"], S["track_conf"], S["model_complexity"], S)
            fa = FrameAnalyzer(self.clf, self.profiles, S)
            records: list[FrameRecord] = []
            counters, reasons = Counter(), Counter()
            idx, shown, t0 = 0, 0, time.time()
            while True:
                if cancel is not None and idx % 15 == 0 and cancel.is_set():
                    raise AnalysisCancelled()
                if idx % step:
                    if not cap.grab():
                        break
                    idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                t = idx / fps
                idx += 1
                counters["sampled"] += 1
                frame = resize_max_width(frame, 960)
                obs, multi = select_hands(tracker.process(frame, mirrored=S["mirror_uploads"]))
                rec, annotated = None, None
                if multi and S["multi_policy"] == "skip":
                    rec = FrameRecord(t, "multi", reason="more than two hands visible")
                else:
                    if hand_mode in ("Left", "Right"):
                        obs = [o for o in obs if o.label == hand_mode]
                    assessed = [(o, fa.assess(o)) for o in obs]
                    if not assessed:
                        rec = FrameRecord(t, "nohand")
                    else:
                        good = [a for _, a in assessed if a.usable and a.label]
                        usable = [a for _, a in assessed if a.usable]
                        if good:
                            b = max(good, key=lambda a: a.confidence)
                            rec = FrameRecord(t, "ok", b.label, b.confidence, b.hand, b.feat)
                        elif usable:
                            rec = FrameRecord(t, "unknown", reason=usable[0].reason or "unclear hand shape")
                        else:
                            rec = FrameRecord(t, "unusable", reason=assessed[0][1].reason)
                    if progress and counters["sampled"] % 6 == 1:
                        annotated = frame.copy()
                        for o, a in assessed:
                            draw_hand(annotated, o, a.caption, STATUS_BGR.get(a.status, (200, 200, 200)))
                counters[rec.status] += 1
                if rec.status in ("unknown", "unusable") and rec.reason:
                    for part in rec.reason.split("; "):
                        reasons[part] += 1
                elif rec.status == "multi":
                    reasons["multiple dancers / more than two hands"] += 1
                records.append(rec)
                if progress:
                    frac = (idx / total) if total > 0 else 0.0
                    elapsed = time.time() - t0
                    eta = f", about {int(elapsed / max(frac, 1e-3) * (1 - frac))} s left" if frac > 0.03 else ""
                    progress(p0 + (p1 - p0) * min(frac, 1.0),
                             f"Analysing {t:5.1f}s of {total / fps:.0f}s{eta}" if total else f"Analysing {t:5.1f}s",
                             annotated)
            if not records:
                raise ValueError("No frames could be read from this video.")
            duration = max(total / fps if total > 0 else 0.0, (idx / fps))
            return Collected(records, fps, duration, frame_dt, counters, reasons, notes)
        finally:
            if tracker:
                tracker.close()
            cap.release()
            cleanup()

    # ------------------------------------------------------------ reference videos
    def reference_sequence(self, ref: dict, progress=None, cancel=None, p0=0.0, p1=0.4) -> dict:
        """Mudra label sequence + feature trajectory of a reference demonstration video (cached)."""
        path = Path(ref["path"])
        key = f"ref_{ref['id']}_{int(path.stat().st_mtime)}_{self.clf.stamp.replace(':', '').replace(' ', '')}.npz"
        cache = CACHE_DIR / key
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            return {"labels": [str(x) for x in z["labels"]], "t": z["t"], "F": z["F"]}
        col = self.collect(path, progress, cancel, "Either", p0, p1)
        segs = build_segments(col.records, self.S["min_segment_sec"], self.S["gap_tol_sec"], col.frame_dt)
        labels = [s.label for s in segs]
        good = [r for r in col.records if r.status == "ok" and r.feat is not None]
        t = np.array([r.t for r in good], float)
        F = np.array([r.feat for r in good], np.float32).reshape(-1, 102)
        np.savez_compressed(cache, labels=np.array(labels, dtype=str), t=t, F=F)
        return {"labels": labels, "t": t, "F": F}

    # --------------------------------------------------------------------- main API
    def analyze(self, path, expected: Optional[list[str]] = None, ref: Optional[dict] = None,
                hand_mode: Optional[str] = None, progress: Optional[ProgressCB] = None,
                cancel: Optional[threading.Event] = None) -> dict:
        if not self.clf.ready:
            raise ValueError("No trained model found. Collect samples and train the model in the "
                             "Reference Mudra Library first.")
        expected_source = "None"
        ref_seq = None
        p_start = 0.0
        if ref is not None:
            if not Path(ref["path"]).exists():
                raise ValueError("The selected reference video file no longer exists.")
            ref_seq = self.reference_sequence(ref, progress, cancel, 0.0, 0.4)
            if not ref_seq["labels"]:
                raise ValueError("No mudras could be recognised in the reference video, so it cannot be "
                                 "used as an expected sequence.")
            expected = ref_seq["labels"]
            expected_source = f"Reference video: {ref.get('title') or Path(ref['path']).name}"
            p_start = 0.4
        elif expected:
            expected_source = "Typed sequence: " + ", ".join(expected)
        col = self.collect(path, progress, cancel, hand_mode, p_start, 1.0)
        res = evaluate_records(col, self.S, self.profiles, expected)
        movement = None
        if ref_seq is not None:
            good = [r for r in col.records if r.status == "ok" and r.feat is not None]
            if good:
                mv = movement_similarity([r.t for r in good], np.array([r.feat for r in good]),
                                         ref_seq["t"], ref_seq["F"], tau=self.S["sim_tau"])
                if mv:
                    movement = mv.to_dict()
                    for (a, b, c) in mv.deviations:
                        res["mistakes"].append({
                            "timestamp_s": a, "end_s": b, "severity": "warning",
                            "description": "Hand movement deviates from the reference video",
                            "suggestion": "Compare this passage with the demonstration; check finger shapes "
                                          "and the timing of the transitions."})
                    res["mistakes"].sort(key=lambda m: m["timestamp_s"])
                    res["notes"].append(f"Movement similarity to the reference video is {mv.similarity:.0f}% "
                                        "(DTW on hand-shape trajectories, ignores speed differences).")
        res["notes"].append("Scores compare your hand geometry with labelled reference samples; they are "
                            "not a substitute for a teacher's judgement.")
        res.update(kind="video", source_name=Path(path).name, source_path=str(path), duration_s=col.duration,
                   expected_source=expected_source, movement=movement,
                   meta={"fps": col.fps, "analysis_interval_s": col.frame_dt, "model": self.clf.meta.get("model"),
                         "model_trained_at": self.clf.stamp})
        return res


# ======================================================================= image analyzer
class ImageAnalyzer:
    def __init__(self, classifier: MudraClassifier, profiles: ReferenceProfiles, S: dict):
        self.clf, self.profiles, self.S = classifier, profiles, S

    def analyze(self, path, expected: Optional[str] = None):
        """Returns (result_dict, annotated_bgr_image)."""
        S = self.S
        img = resize_max_width(read_image(path), 1600)
        obs = []
        for conf in (S["det_conf"], 0.3):                       # retry with a lower threshold
            tr = HandTracker(True, 2, conf, S["track_conf"], S["model_complexity"], S)
            try:
                obs = tr.process(img, mirrored=S["mirror_uploads"])
            finally:
                tr.close()
            if obs:
                break
        fa = FrameAnalyzer(self.clf, self.profiles, S)
        annotated = img.copy()
        items, notes = [], []
        reasons: Counter = Counter()
        for k, o in enumerate(sorted(obs, key=lambda x: x.label)):
            a = fa.assess(o, expected)
            draw_hand(annotated, o, f"{o.label}: {a.caption}", STATUS_BGR.get(a.status, (200, 200, 200)))
            if not a.usable:
                sug = "Retake the photo: " + "; ".join(a.reasons) + "."
                reasons.update(a.reasons)
            elif a.status == C.UNRECOGNISED:
                sug = f"Hand shape could not be matched confidently ({a.reason}). Try a clearer, front-on photo."
            else:
                sug = " ".join(a.tips) or _default_tip(a.status)
            if expected and a.label and a.label != expected:
                sug = f"Detected {a.label} instead of {expected}. " + sug
            score = None
            if a.status in SCORED:
                score = 0.0 if (expected and a.label != expected) else a.similarity
            items.append(dict(idx=k + 1, start_s=0.0, end_s=0.0, hand=o.label, expected=expected or "",
                              detected=a.label or C.UNRECOGNISED, similarity=a.similarity,
                              confidence=a.confidence if a.usable else None, status=a.status,
                              suggestion=sug.strip(), score=score))
        if not self.clf.ready:
            notes.append("No trained model yet - only hand landmarks were detected. Train the model first.")
        if not obs:
            notes.append("No hand was detected. Make sure the whole hand is inside the frame, well lit and in focus.")
        notes.append("Similarity compares your hand geometry with labelled reference samples; it is not a "
                     "substitute for a teacher's judgement.")
        summary, per_mudra = summarize(items)
        res = {"kind": "image", "source_name": Path(path).name, "source_path": str(path), "duration_s": 0.0,
               "expected_source": expected or "None", "items": items, "mistakes": mistakes_from_items(items),
               "summary": summary, "per_mudra": per_mudra, "notes": notes, "movement": None,
               "quality": {"frames_sampled": 1, "reasons": dict(reasons)},
               "meta": {"model": self.clf.meta.get("model"), "model_trained_at": self.clf.stamp}}
        return res, annotated
