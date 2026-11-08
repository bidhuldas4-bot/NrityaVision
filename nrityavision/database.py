"""SQLite storage: users, mudras, reference media, training samples, analyses, reports, settings."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import ASAMYUTA_HASTAS, DB_PATH, DEFAULT_SETTINGS, INITIAL_MUDRAS


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
    email TEXT DEFAULT '', role TEXT DEFAULT 'Student', created_at TEXT);
CREATE TABLE IF NOT EXISTS mudras(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
    meaning TEXT DEFAULT '', description TEXT DEFAULT '', created_at TEXT);
CREATE TABLE IF NOT EXISTS reference_media(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mudra_id INTEGER REFERENCES mudras(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,            -- image | video_clip | video_sequence
    path TEXT NOT NULL, title TEXT DEFAULT '', notes TEXT DEFAULT '', created_at TEXT);
CREATE TABLE IF NOT EXISTS samples(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mudra_id INTEGER NOT NULL REFERENCES mudras(id) ON DELETE CASCADE,
    landmarks TEXT NOT NULL,       -- 63 canonical floats (JSON)
    handedness TEXT, source TEXT, group_id TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS idx_samples_mudra ON samples(mudra_id);
CREATE TABLE IF NOT EXISTS training_runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, model_name TEXT,
    n_samples INTEGER, n_classes INTEGER, cv_accuracy REAL, optimistic INTEGER, meta_json TEXT);
CREATE TABLE IF NOT EXISTS analyses(
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER REFERENCES users(id),
    kind TEXT, source_name TEXT, source_path TEXT, duration_s REAL, created_at TEXT,
    total_items INTEGER, correct INTEGER, needs_improvement INTEGER, incorrect INTEGER,
    unrecognised INTEGER, overall_accuracy REAL, mean_similarity REAL,
    expected_source TEXT, movement_similarity REAL, payload_json TEXT);
CREATE TABLE IF NOT EXISTS analysis_items(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    idx INTEGER, start_s REAL, end_s REAL, hand TEXT, expected TEXT, detected TEXT,
    similarity REAL, confidence REAL, status TEXT, suggestion TEXT, score REAL);
CREATE TABLE IF NOT EXISTS mistakes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    timestamp_s REAL, end_s REAL, severity TEXT, description TEXT, suggestion TEXT);
CREATE TABLE IF NOT EXISTS reports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE,
    pdf_path TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
"""


class Database:
    def __init__(self, path=DB_PATH):
        self.path = Path(path)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    def initialize(self) -> None:
        with self.conn() as c:
            c.executescript(SCHEMA)
            if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
                c.execute("INSERT INTO users(name,role,created_at) VALUES('Guest Dancer','Student',?)", (now(),))
            if c.execute("SELECT COUNT(*) FROM mudras").fetchone()[0] == 0:
                for name, meaning, desc in INITIAL_MUDRAS:
                    c.execute("INSERT INTO mudras(name,meaning,description,created_at) VALUES(?,?,?,?)",
                              (name, meaning, desc, now()))

    # ------------------------------------------------------------------ settings
    def get_settings(self) -> dict:
        out = dict(DEFAULT_SETTINGS)
        with self.conn() as c:
            for r in c.execute("SELECT key,value FROM settings"):
                if r["key"] in out:
                    out[r["key"]] = self._cast(r["key"], r["value"])
        return out

    @staticmethod
    def _cast(key, value):
        default = DEFAULT_SETTINGS[key]
        try:
            if isinstance(default, bool):
                return str(value).lower() in ("1", "true", "yes")
            if isinstance(default, int):
                return int(float(value))
            if isinstance(default, float):
                return float(value)
        except (TypeError, ValueError):
            return default
        return str(value)

    def set_settings(self, values: dict) -> None:
        with self.conn() as c:
            for k, v in values.items():
                if k in DEFAULT_SETTINGS:
                    c.execute("INSERT INTO settings(key,value) VALUES(?,?) "
                              "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, str(v)))

    def reset_settings(self) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM settings")

    # --------------------------------------------------------------------- users
    def list_users(self):
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM users ORDER BY id")]

    def add_user(self, name: str, email: str = "", role: str = "Student") -> int:
        name = name.strip()
        if not name:
            raise ValueError("Name cannot be empty.")
        try:
            with self.conn() as c:
                return c.execute("INSERT INTO users(name,email,role,created_at) VALUES(?,?,?,?)",
                                 (name, email.strip(), role, now())).lastrowid
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"User '{name}' already exists.") from exc

    def get_user(self, user_id: int):
        with self.conn() as c:
            r = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            return dict(r) if r else None

    # -------------------------------------------------------------------- mudras
    def list_mudras(self):
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT m.*, (SELECT COUNT(*) FROM samples s WHERE s.mudra_id=m.id) AS n_samples,"
                " (SELECT COUNT(*) FROM reference_media r WHERE r.mudra_id=m.id) AS n_refs "
                "FROM mudras m ORDER BY m.id")]

    def mudra_names(self) -> list[str]:
        return [m["name"] for m in self.list_mudras()]

    def get_mudra_id(self, name: str):
        with self.conn() as c:
            r = c.execute("SELECT id FROM mudras WHERE lower(name)=lower(?)", (name.strip(),)).fetchone()
            return r["id"] if r else None

    def canonical_name(self, name: str):
        with self.conn() as c:
            r = c.execute("SELECT name FROM mudras WHERE lower(name)=lower(?)", (name.strip(),)).fetchone()
            return r["name"] if r else None

    def add_mudra(self, name: str, meaning: str = "", description: str = "") -> int:
        name = name.strip()
        if not name:
            raise ValueError("Mudra name cannot be empty.")
        try:
            with self.conn() as c:
                return c.execute("INSERT INTO mudras(name,meaning,description,created_at) VALUES(?,?,?,?)",
                                 (name, meaning.strip(), description.strip(), now())).lastrowid
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A mudra named '{name}' already exists.") from exc

    def update_mudra(self, mudra_id: int, name: str, meaning: str, description: str) -> None:
        try:
            with self.conn() as c:
                c.execute("UPDATE mudras SET name=?,meaning=?,description=? WHERE id=?",
                          (name.strip(), meaning.strip(), description.strip(), mudra_id))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A mudra named '{name}' already exists.") from exc

    def delete_mudra(self, mudra_id: int) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM mudras WHERE id=?", (mudra_id,))

    def add_all_asamyuta(self) -> int:
        added = 0
        with self.conn() as c:
            for name in ASAMYUTA_HASTAS:
                if not c.execute("SELECT 1 FROM mudras WHERE lower(name)=lower(?)", (name,)).fetchone():
                    c.execute("INSERT INTO mudras(name,created_at) VALUES(?,?)", (name, now()))
                    added += 1
        return added

    # ------------------------------------------------------------ reference media
    def add_reference(self, mudra_id, kind: str, path: str, title: str = "", notes: str = "") -> int:
        with self.conn() as c:
            return c.execute("INSERT INTO reference_media(mudra_id,kind,path,title,notes,created_at) "
                             "VALUES(?,?,?,?,?,?)", (mudra_id, kind, str(path), title, notes, now())).lastrowid

    def list_references(self, kind: str | None = None):
        q = ("SELECT r.*, m.name AS mudra FROM reference_media r "
             "LEFT JOIN mudras m ON m.id=r.mudra_id")
        args = ()
        if kind:
            q += " WHERE r.kind=?"
            args = (kind,)
        with self.conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY r.id DESC", args)]

    def get_reference(self, ref_id: int):
        with self.conn() as c:
            r = c.execute("SELECT r.*, m.name AS mudra FROM reference_media r "
                          "LEFT JOIN mudras m ON m.id=r.mudra_id WHERE r.id=?", (ref_id,)).fetchone()
            return dict(r) if r else None

    def delete_reference(self, ref_id: int) -> None:
        ref = self.get_reference(ref_id)
        with self.conn() as c:
            c.execute("DELETE FROM reference_media WHERE id=?", (ref_id,))
        if ref:
            try:
                Path(ref["path"]).unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------- samples
    def add_samples(self, mudra_id: int, canon_list, handedness: str, source: str, group_id: str) -> int:
        rows = [(mudra_id, json.dumps(np.round(np.asarray(c, float).ravel(), 5).tolist()),
                 handedness, source, group_id, now()) for c in canon_list]
        with self.conn() as c:
            c.executemany("INSERT INTO samples(mudra_id,landmarks,handedness,source,group_id,created_at) "
                          "VALUES(?,?,?,?,?,?)", rows)
        return len(rows)

    def samples_by_mudra(self) -> dict:
        out: dict[str, list] = {}
        with self.conn() as c:
            for r in c.execute("SELECT m.name, s.landmarks FROM samples s JOIN mudras m ON m.id=s.mudra_id"):
                out.setdefault(r["name"], []).append(json.loads(r["landmarks"]))
        return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}

    def load_training_data(self):
        """Returns (canonical landmarks (N,63), labels, group ids)."""
        X, y, g = [], [], []
        with self.conn() as c:
            for r in c.execute("SELECT m.name, s.landmarks, s.group_id FROM samples s "
                               "JOIN mudras m ON m.id=s.mudra_id"):
                X.append(json.loads(r["landmarks"]))
                y.append(r["name"])
                g.append(r["group_id"] or "")
        return np.asarray(X, dtype=np.float64).reshape(-1, 63), y, g

    def delete_samples(self, mudra_id: int) -> int:
        with self.conn() as c:
            return c.execute("DELETE FROM samples WHERE mudra_id=?", (mudra_id,)).rowcount

    def count_samples(self) -> int:
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    def add_training_run(self, meta: dict) -> None:
        light = {k: v for k, v in meta.items() if k not in ("report", "confusion")}
        with self.conn() as c:
            c.execute("INSERT INTO training_runs(created_at,model_name,n_samples,n_classes,cv_accuracy,"
                      "optimistic,meta_json) VALUES(?,?,?,?,?,?,?)",
                      (now(), meta["model"], meta["n_samples"], len(meta["per_class"]),
                       meta["cv_accuracy"], int(meta["optimistic"]), json.dumps(light)))

    # ------------------------------------------------------------------ analyses
    def save_analysis(self, res: dict, user_id: int) -> int:
        s = res.get("summary", {})
        payload = {k: res.get(k) for k in ("summary", "per_mudra", "quality", "notes", "movement", "meta")}
        with self.conn() as c:
            aid = c.execute(
                "INSERT INTO analyses(user_id,kind,source_name,source_path,duration_s,created_at,total_items,"
                "correct,needs_improvement,incorrect,unrecognised,overall_accuracy,mean_similarity,"
                "expected_source,movement_similarity,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (user_id, res["kind"], res["source_name"], res.get("source_path", ""), res.get("duration_s", 0),
                 now(), s.get("total_items", 0), s.get("correct", 0), s.get("needs_improvement", 0),
                 s.get("incorrect", 0) + s.get("missed", 0), s.get("unrecognised", 0),
                 s.get("overall_accuracy"), s.get("mean_similarity"), res.get("expected_source", "None"),
                 (res.get("movement") or {}).get("similarity"), json.dumps(payload, default=float))).lastrowid
            for it in res.get("items", []):
                c.execute("INSERT INTO analysis_items(analysis_id,idx,start_s,end_s,hand,expected,detected,"
                          "similarity,confidence,status,suggestion,score) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                          (aid, it["idx"], it.get("start_s"), it.get("end_s"), it.get("hand", ""),
                           it.get("expected", ""), it.get("detected", ""), it.get("similarity"),
                           it.get("confidence"), it["status"], it.get("suggestion", ""), it.get("score")))
            for m in res.get("mistakes", []):
                c.execute("INSERT INTO mistakes(analysis_id,timestamp_s,end_s,severity,description,suggestion) "
                          "VALUES(?,?,?,?,?,?)", (aid, m["timestamp_s"], m.get("end_s"), m.get("severity", "error"),
                                                  m["description"], m.get("suggestion", "")))
        return aid

    def list_analyses(self, user_id: int | None = None):
        q = ("SELECT a.*, u.name AS user_name FROM analyses a LEFT JOIN users u ON u.id=a.user_id")
        args = ()
        if user_id is not None:
            q += " WHERE a.user_id=?"
            args = (user_id,)
        with self.conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY a.id DESC", args)]

    def get_analysis(self, analysis_id: int):
        with self.conn() as c:
            a = c.execute("SELECT a.*, u.name AS user_name FROM analyses a LEFT JOIN users u ON u.id=a.user_id "
                          "WHERE a.id=?", (analysis_id,)).fetchone()
            if not a:
                return None
            res = dict(a)
            payload = json.loads(res.pop("payload_json") or "{}")
            res.update({k: v for k, v in payload.items() if v is not None})
            res["items"] = [dict(r) for r in c.execute(
                "SELECT * FROM analysis_items WHERE analysis_id=? ORDER BY idx", (analysis_id,))]
            res["mistakes"] = [dict(r) for r in c.execute(
                "SELECT * FROM mistakes WHERE analysis_id=? ORDER BY timestamp_s", (analysis_id,))]
            res.setdefault("per_mudra", {})
            res.setdefault("notes", [])
            return res

    def delete_analysis(self, analysis_id: int) -> None:
        with self.conn() as c:
            paths = [r["pdf_path"] for r in c.execute("SELECT pdf_path FROM reports WHERE analysis_id=?",
                                                      (analysis_id,))]
            c.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass

    def add_report(self, analysis_id: int, pdf_path: str) -> int:
        with self.conn() as c:
            return c.execute("INSERT INTO reports(analysis_id,pdf_path,created_at) VALUES(?,?,?)",
                             (analysis_id, str(pdf_path), now())).lastrowid

    def stats(self, user_id: int | None = None) -> dict:
        with self.conn() as c:
            n_an = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            avg = c.execute("SELECT AVG(overall_accuracy) FROM analyses WHERE overall_accuracy IS NOT NULL"
                            + (" AND user_id=?" if user_id else ""), ((user_id,) if user_id else ())).fetchone()[0]
            return {"mudras": c.execute("SELECT COUNT(*) FROM mudras").fetchone()[0],
                    "samples": c.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
                    "analyses": n_an, "avg_accuracy": avg}
