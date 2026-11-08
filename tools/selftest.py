"""Developer self-test (no camera needed).

Uses *synthetic* kinematic hand skeletons to verify that the pipeline runs end to end:
feature extraction -> training -> prediction/rejection -> segmentation -> DTW alignment ->
database -> PDF/CSV. The synthetic hands are NOT real mudras: the accuracy printed here says
nothing about real-world performance. Run:  python tools/selftest.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrityavision import config as C  # noqa: E402
from nrityavision.analyzer import (Collected, FrameRecord, evaluate_records,  # noqa: E402
                                   build_segments)
from nrityavision.classifier import MudraClassifier  # noqa: E402
from nrityavision.database import Database  # noqa: E402
from nrityavision.dtw import dtw, movement_similarity  # noqa: E402
from nrityavision.features import canonicalize, extract_features, feature_matrix  # noqa: E402
from nrityavision.report import build_csv, build_pdf  # noqa: E402
from nrityavision.similarity import ReferenceProfiles  # noqa: E402

RNG = np.random.default_rng(7)
MCP = {1: (0.45, 0.30), 5: (0.35, 1.0), 9: (0.0, 1.05), 13: (-0.3, 1.0), 17: (-0.6, 0.9)}
PATTERNS = {   # curl per finger: thumb, index, middle, ring, little (0 straight .. 1 curled)
    "Pataka": [0.6, 0.0, 0.0, 0.0, 0.0],
    "Tripataka": [0.6, 0.0, 0.0, 1.0, 0.0],
    "Ardhapataka": [0.6, 0.0, 0.0, 1.0, 1.0],
    "Mushti": [0.9, 1.0, 1.0, 1.0, 1.0],
    "Mayura": [0.8, 0.0, 0.0, 0.9, 0.0],
}


def make_hand(curls, noise=0.03):
    pts = np.zeros((21, 3))
    for k, base in enumerate((1, 5, 9, 13, 17)):
        c = curls[k] + RNG.normal(0, 0.06)
        mx, my = MCP[base]
        pts[base] = (mx, my, 0)
        pos, phi = np.array([mx, my, 0.0]), 0.0
        sx = 0.5 if base == 1 else (0.1 * (base - 9) / 4)
        for n, (ln, fl) in enumerate(((0.5, 1.2), (0.3, 1.6), (0.25, 1.0))):
            phi += fl * np.clip(c, 0, 1.1)
            d = np.array([sx, math.cos(phi), -math.sin(phi)])
            pos = pos + ln * d / np.linalg.norm(d)
            pts[base + n + 1] = pos
    return pts + RNG.normal(0, noise * 0.2, pts.shape)


def random_view(pts, label):
    a, b, c = RNG.uniform(-0.6, 0.6, 3)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    p = pts @ (Rz @ Ry @ Rx).T * RNG.uniform(60, 220)
    if label == "Left":
        p[:, 0] *= -1
    return p + np.array([300, 300, 0])


def sample(name):
    label = "Right" if RNG.random() < 0.5 else "Left"
    return canonicalize(random_view(make_hand(PATTERNS[name]), label), label)


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        check.failed += 1


check.failed = 0


def main():
    tmp = Path(tempfile.mkdtemp())
    db = Database(tmp / "t.db")
    db.initialize()
    check(len(db.list_mudras()) == 5, "database seeded with 5 initial mudras")
    check(db.add_all_asamyuta() == 23, "28 Asamyuta Hastas available (23 added on top of the initial 5)")
    S = db.get_settings()

    # ---- features: invariance
    base = make_hand(PATTERNS["Pataka"], 0)
    f1 = extract_features(canonicalize(random_view(base, "Right"), "Right"))
    f2 = extract_features(canonicalize(random_view(base, "Right"), "Right"))
    check(np.allclose(f1[63:97], f2[63:97], atol=0.02), "shape features invariant to rotation/scale")
    fl = extract_features(canonicalize(random_view(base, "Left"), "Left"))
    check(np.allclose(f1[63:97], fl[63:97], atol=0.02), "left hand mirrored to match right hand")

    # ---- dataset + training
    for name in PATTERNS:
        mid = db.get_mudra_id(name)
        for session in range(4):
            db.add_samples(mid, [sample(name) for _ in range(25)], "Right", "synthetic", f"{name}{session}")
    canon, y, g = db.load_training_data()
    clf = MudraClassifier(tmp / "m.joblib")
    meta = clf.train(feature_matrix(canon), y, g, log=lambda s: None)
    print(f"      model={meta['model']} grouped-CV acc={meta['cv_accuracy']:.2f} optimistic={meta['optimistic']}")
    check(meta["cv_accuracy"] > 0.8 and not meta["optimistic"], "classifier trains with session-grouped validation")
    prof = ReferenceProfiles(S["sim_tau"])
    prof.rebuild(db)
    clf2 = MudraClassifier(tmp / "m.joblib")
    check(clf2.load() and clf2.ready, "model saved and reloaded")

    # ---- prediction, rejection and similarity
    ok = sum(clf2.predict(extract_features(sample(n))).label == n for n in PATTERNS for _ in range(10))
    print(f"      held-out accuracy on synthetic hands: {ok}/50")
    check(ok >= 40, "held-out synthetic hands recognised")
    weird = make_hand([0.0, 1.0, 0.0, 1.0, 0.0])
    p = clf2.predict(extract_features(canonicalize(random_view(weird, "Right"), "Right")))
    print(f"      odd shape -> label={p.label} closest={p.top_label} conf={p.confidence:.2f} reason={p.reason}")
    same = prof.similarity("Pataka", extract_features(sample("Pataka")))
    other = prof.similarity("Pataka", extract_features(sample("Mushti")))
    check(same > 70 and other < 40, f"similarity separates true ({same:.0f}%) from wrong mudra ({other:.0f}%)")
    mushti_as_pataka = prof.explain("Pataka", extract_features(sample("Mushti")))
    check(len(mushti_as_pataka) > 0, "corrective tips generated: " + (mushti_as_pataka[0] if mushti_as_pataka else ""))

    # ---- synthetic video timeline: Pataka, Tripataka, (blur gap), Mayura, Mushti
    dt, t = 0.1, 0.0
    recs = []
    script = [("Pataka", 2.0), (None, 0.8), ("Tripataka", 2.0), ("Mayura", 1.5), ("Mushti", 2.0)]
    for name, dur in script:
        for _ in range(int(dur / dt)):
            if name is None:
                recs.append(FrameRecord(t, "unusable", reason="hand is blurry / out of focus"))
            else:
                f = extract_features(sample(name))
                pr = clf2.predict(f)
                recs.append(FrameRecord(t, "ok" if pr.label else "unknown", pr.label, pr.confidence, "Right", f,
                                        pr.reason))
            t += dt
    col = Collected(recs, 30.0, t, dt, Counter(r.status for r in recs) + Counter(sampled=len(recs)),
                    Counter({"hand is blurry / out of focus": 8}), [])
    segs = build_segments(recs, 0.4, 0.35, dt)
    print("      segments:", [(s.label, round(s.start, 1)) for s in segs])
    check([s.label for s in segs][:2] == ["Pataka", "Tripataka"], "frames grouped into mudra segments")
    free = evaluate_records(col, S, prof)
    check(free["summary"]["overall_accuracy"] is not None, "free evaluation produces accuracy")
    exp = evaluate_records(col, S, prof, ["Pataka", "Tripataka", "Ardhapataka", "Mayura", "Mushti"])
    sts = [(it["expected"], it["detected"], it["status"]) for it in exp["items"]]
    print("      expected-sequence rows:", sts)
    check(any(e == "Ardhapataka" and st in (C.MISSED, C.INCORRECT) for e, d, st in sts),
          "deliberately skipped mudra is flagged")
    check(any(e == "Mayura" and st == C.CORRECT for e, d, st in sts) and any(e == "Ardhapataka" and st == C.MISSED for e, d, st in sts),
          "alignment attributes Mayura/Mushti correctly around the skipped mudra")
    check(any(m["severity"] == "info" for m in exp["mistakes"]), "blurry stretch reported as unclear, not guessed")

    # ---- DTW
    cost, path = dtw(np.abs(np.arange(6)[:, None] - np.array([0, 1, 1, 2, 3, 4, 5, 5])[None, :]).astype(float))
    check(cost == 0.0 and path[0] == (0, 0) and path[-1] == (5, 7), "DTW aligns stretched sequences")
    tl = np.array([r.t for r in recs if r.feat is not None])
    Fl = np.array([r.feat for r in recs if r.feat is not None])
    mv = movement_similarity(tl, Fl, tl * 1.3, Fl, tau=S["sim_tau"])
    check(mv is not None and mv.similarity > 85, f"movement similarity ignores speed (got {mv.similarity:.0f}%)")

    # ---- persistence + report
    res = dict(exp, kind="video", source_name="synthetic.mp4", source_path="", duration_s=t,
               expected_source="Typed sequence", movement=mv.to_dict(), meta={})
    aid = db.save_analysis(res, 1)
    back = db.get_analysis(aid)
    check(len(back["items"]) == len(res["items"]) and back["summary"]["correct"] == res["summary"]["correct"],
          "analysis saved to / loaded from SQLite")
    pdf = build_pdf(back, tmp / "r.pdf")
    csv_ = build_csv(back, tmp / "r.csv")
    check(Path(pdf).stat().st_size > 5000 and Path(csv_).exists(), f"PDF ({Path(pdf).stat().st_size} bytes) and CSV written")
    print("\nSELF-TEST", "FAILED" if check.failed else "PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
