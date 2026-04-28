import json
import sqlite3
from pathlib import Path


class SQLiteStorage:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "optimizer.sqlite"
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._schema()

    def _schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, fingerprints TEXT NOT NULL, created_at INTEGER DEFAULT (strftime('%s','now')));
        CREATE TABLE IF NOT EXISTS trials(id INTEGER PRIMARY KEY, status TEXT, params_hash TEXT UNIQUE, params TEXT, metrics TEXT, objective_value REAL, payload TEXT);
        CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);
        CREATE INDEX IF NOT EXISTS idx_trials_params_hash ON trials(params_hash);
        CREATE TABLE IF NOT EXISTS result_profiles(name TEXT PRIMARY KEY, trial_id INTEGER, reason TEXT, score_name TEXT, score_value REAL, payload TEXT);
        """)
        self.conn.commit()

    def init_run(self, fingerprints):
        self.conn.execute(
            "INSERT INTO runs(id,fingerprints) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET fingerprints=excluded.fingerprints",
            (json.dumps(fingerprints, sort_keys=True, default=str),),
        )
        self.conn.commit()

    def load_meta(self):
        row = self.conn.execute("SELECT fingerprints FROM runs WHERE id=1").fetchone()
        return json.loads(row[0]) if row else None

    def save_trial(self, trial):
        d = trial.to_dict()
        ph = (
            d.get("params_hash")
            or __import__("hashlib")
            .sha256(json.dumps(d["params"], sort_keys=True, default=str).encode())
            .hexdigest()
        )
        self.conn.execute(
            "INSERT INTO trials(id,status,params_hash,params,metrics,objective_value,payload) VALUES(?,?,?,?,?,?,?) ON CONFLICT(params_hash) DO UPDATE SET id=excluded.id,status=excluded.status,params=excluded.params,metrics=excluded.metrics,objective_value=excluded.objective_value,payload=excluded.payload",
            (
                trial.id,
                trial.status,
                ph,
                json.dumps(trial.params, sort_keys=True, default=str),
                json.dumps(trial.metrics, sort_keys=True, default=str),
                trial.objective_value,
                json.dumps(d, sort_keys=True, default=str),
            ),
        )
        self.conn.commit()

    def save_profile(self, profile):
        tid = profile.trial.id if profile.trial else None
        self.conn.execute(
            "INSERT OR REPLACE INTO result_profiles(name,trial_id,reason,score_name,score_value,payload) VALUES(?,?,?,?,?,?)",
            (
                profile.name,
                tid,
                profile.reason,
                profile.score_name,
                profile.score_value,
                json.dumps(profile.to_dict(), sort_keys=True, default=str),
            ),
        )
        self.conn.commit()

    def load_trials_raw(self):
        return [
            json.loads(r[0]) for r in self.conn.execute("SELECT payload FROM trials ORDER BY id")
        ]

    def close(self):
        self.conn.close()
