"""Atomic episode telemetry in a local SQLite database; no model loading required."""

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

DATABASE = "episodes.sqlite3"


def pack(arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def unpack(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class EpisodeStore:
    """One writer per run; WAL allows the dashboard to read during training.

    Household arrays are compressed per step. Starting histories and reference
    profiles are shared between repeated occurrences of the same episode inputs.
    All episodes are retained, including those later rolled back on resumption.
    """

    def __init__(self, folder, manifest):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / DATABASE
        self.db = sqlite3.connect(self.path, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS manifest (value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS contexts (
                id TEXT PRIMARY KEY, arrays BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context TEXT NOT NULL REFERENCES contexts(id),
                origin TEXT NOT NULL, kind TEXT NOT NULL,
                started TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
                discarded INTEGER NOT NULL DEFAULT 0,
                reward REAL NOT NULL DEFAULT 0, steps INTEGER NOT NULL DEFAULT 0,
                first_decision INTEGER, last_decision INTEGER,
                metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS steps (
                episode INTEGER NOT NULL REFERENCES episodes(id),
                hour INTEGER NOT NULL, decision INTEGER NOT NULL,
                action INTEGER NOT NULL, price REAL NOT NULL, reward REAL NOT NULL,
                components TEXT NOT NULL, diagnostics TEXT NOT NULL,
                arrays BLOB NOT NULL, PRIMARY KEY (episode, hour));
            CREATE INDEX IF NOT EXISTS episode_kind ON episodes(kind, discarded, id);
            """
        )
        saved = self.db.execute("SELECT value FROM manifest").fetchone()
        if saved and saved[0] != encode(manifest):
            self.db.close()
            raise ValueError("Recording provenance changed; choose a new run folder.")
        if not saved:
            self.db.execute("INSERT INTO manifest VALUES (?)", (encode(manifest),))
        # An interrupted writer may have left a partial episode. Keep its evidence.
        self.db.execute(
            "UPDATE episodes SET status='interrupted' WHERE status='active'"
        )
        self.db.commit()

    @property
    def cursor(self):
        return self.db.execute("SELECT COALESCE(MAX(id), 0) FROM episodes").fetchone()[
            0
        ]

    def start(self, metadata, arrays, kind):
        blob = pack(arrays)
        context = hashlib.sha256(blob).hexdigest()
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO contexts VALUES (?, ?)", (context, blob)
            )
            result = self.db.execute(
                "INSERT INTO episodes(context, origin, kind, started, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (context, metadata["origin"], kind, utc_now(), encode(metadata)),
            )
        return result.lastrowid

    def append(
        self,
        episode,
        hour,
        decision,
        action,
        price,
        reward,
        components,
        diagnostics,
        arrays,
        finished,
        context_arrays=None,
    ):
        """Commit the step and episode totals in the same transaction."""
        with self.db:
            if context_arrays is not None:
                blob = pack(context_arrays)
                context = hashlib.sha256(blob).hexdigest()
                self.db.execute(
                    "INSERT OR IGNORE INTO contexts VALUES (?, ?)", (context, blob)
                )
                self.db.execute(
                    "UPDATE episodes SET context=? WHERE id=?", (context, episode)
                )
            self.db.execute(
                "INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    episode,
                    hour,
                    decision,
                    action,
                    price,
                    reward,
                    encode(components),
                    encode(diagnostics),
                    pack(arrays),
                ),
            )
            self.db.execute(
                "UPDATE episodes SET reward=reward+?, steps=steps+1, "
                "first_decision=COALESCE(first_decision, ?), last_decision=?, "
                "status=? WHERE id=?",
                (
                    reward,
                    decision,
                    decision,
                    "complete" if finished else "active",
                    episode,
                ),
            )

    def rollback_after(self, cursor):
        """Flag interactions absent from the recovered learner; never delete them."""
        with self.db:
            self.db.execute("UPDATE episodes SET discarded=1 WHERE id>?", (cursor,))

    def close(self):
        with self.db:
            self.db.execute(
                "UPDATE episodes SET status='interrupted' WHERE status='active'"
            )
        self.db.close()


def reader(path):
    return sqlite3.connect(
        Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5
    )


def episodes(path):
    with closing(reader(path)) as db:
        return pd.read_sql_query("SELECT * FROM episodes WHERE steps>0 ORDER BY id", db)


def load_episode(path, episode_id):
    with closing(reader(path)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")  # Keep metadata and step arrays at one reader snapshot.
        row = db.execute(
            "SELECT * FROM episodes WHERE id=?", (int(episode_id),)
        ).fetchone()
        if row is None:
            raise ValueError("Episode does not exist.")
        context = db.execute(
            "SELECT arrays FROM contexts WHERE id=?", (row["context"],)
        ).fetchone()
        steps = db.execute(
            "SELECT * FROM steps WHERE episode=? ORDER BY hour", (int(episode_id),)
        ).fetchall()
        manifest = json.loads(db.execute("SELECT value FROM manifest").fetchone()[0])
    result = dict(row)
    result["metadata"] = json.loads(result["metadata"])
    result["manifest"] = manifest
    result["context"] = unpack(context[0])
    if "reference_known" in result["context"]:
        known = result["context"]["reference_known"].astype(bool)
        # Unknown reference/market values must not appear as real zero forecasts.
        result["context"]["reference"][~known] = np.nan
        result["context"]["market_prices"][:, ~known] = np.nan
    frames, arrays = [], []
    for step in steps:
        item = dict(step)
        arrays.append(unpack(item.pop("arrays")))
        item.update(
            {f"reward_{k}": v for k, v in json.loads(item.pop("components")).items()}
        )
        item.update(json.loads(item.pop("diagnostics")))
        frames.append(item)
    result["steps"] = pd.DataFrame(frames)
    result["arrays"] = (
        {name: np.stack([a[name] for a in arrays]) for name in arrays[0]}
        if arrays
        else {}
    )
    return result


def discover(root):
    """Find recorded runs and older SB3 fits without opening checkpoint pickles."""
    root = Path(root).expanduser().resolve()
    folders = {p.parent for p in root.rglob(DATABASE)} if root.exists() else set()
    folders.update(p.parent for p in root.rglob("best.zip"))
    for path in root.rglob("settings.json"):
        try:
            settings = json.loads(path.read_text())
            if str(settings.get("version", "")).startswith("regional_stress_"):
                folders.add(path.parent)
        except (OSError, ValueError):
            continue
    return sorted(folders, key=lambda p: p.relative_to(root).as_posix())


def validation_history(folder):
    path = Path(folder) / "history.json"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.DataFrame(json.loads(path.read_text()))
    column = "mean_cost" if "mean_cost" in frame else "loss"
    if column in frame:
        frame["validation_reward"] = -100 * frame[column]
    return frame


def diagnostics(episode):
    """Describe suspicious patterns; these are flags, not proof of reward hacking."""
    frame, arrays = episode["steps"], episode["arrays"]
    if frame.empty:
        return []
    messages = []
    components = frame.filter(regex="^reward_").sum(axis=1)
    mismatch = np.max(np.abs(components - frame.reward))
    if not np.isfinite(mismatch) or mismatch > 1e-6:
        messages.append(
            ("error", f"Reward components disagree by up to {mismatch:.3g}.")
        )
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            messages.append(("error", f"{name}: nonfinite household values."))
        if (values < 0).any():
            messages.append(("error", f"{name}: negative household demand/energy."))
    if frame.action.nunique() == 1:
        messages.append(("info", "One price action throughout the observed episode."))
    if len(frame) > 1:
        flat = np.mean(np.ptp(arrays["requested"], axis=0) <= 1e-5)
        messages.append(
            ("info", f"Households with a flat demand trajectory: {flat:.1%}.")
        )
    magnitudes = frame.filter(regex="^reward_").abs().sum()
    if magnitudes.sum() > 1e-9 and magnitudes.max() / magnitudes.sum() > 0.9:
        messages.append(
            ("warning", f"{magnitudes.idxmax()} supplies over 90% of reward magnitude.")
        )
    if (frame.reward.iloc[:-1] == 0).all() and frame.reward.iloc[-1] != 0:
        messages.append(
            (
                "info",
                "All reward is booked at termination; it evaluates the whole episode.",
            )
        )
    return messages
