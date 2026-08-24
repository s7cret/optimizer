import json
import sqlite3
from pathlib import Path

from optimizer.core.trial_key import validate_trial_identity_payload
from optimizer.errors import StorageError

_TERMINAL = {"completed", "failed", "timeout", "canceled"}


class SQLiteStorage:
    def __init__(self, output_dir):
        self._closed = True
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "optimizer.sqlite"
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._closed = False
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._schema()

    def _schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, fingerprints TEXT NOT NULL, created_at INTEGER DEFAULT (strftime('%s','now')));
        CREATE TABLE IF NOT EXISTS trials(
          id INTEGER PRIMARY KEY,
          status TEXT,
          params_hash TEXT,
          params TEXT,
          metrics TEXT,
          objective_value REAL,
          payload TEXT,
          trial_key TEXT,
          lifecycle TEXT,
          identity_payload TEXT,
          constraints_payload TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);
        CREATE INDEX IF NOT EXISTS idx_trials_params_hash ON trials(params_hash);
        CREATE TABLE IF NOT EXISTS result_profiles(name TEXT PRIMARY KEY, trial_id INTEGER, reason TEXT, score_name TEXT, score_value REAL, payload TEXT);
        CREATE TABLE IF NOT EXISTS champions(
          name TEXT PRIMARY KEY,
          trial_key TEXT NOT NULL,
          selected_profile TEXT,
          selection_policy TEXT NOT NULL,
          constraints_payload TEXT NOT NULL,
          payload TEXT NOT NULL
        );
        """)
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(trials)")}
        for name, declaration in (
            ("trial_key", "TEXT"),
            ("lifecycle", "TEXT"),
            ("identity_payload", "TEXT"),
            ("constraints_payload", "TEXT"),
        ):
            if name not in columns:
                self.conn.execute(f"ALTER TABLE trials ADD COLUMN {name} {declaration}")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_trials_trial_key "
            "ON trials(trial_key) WHERE trial_key IS NOT NULL"
        )
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

    @staticmethod
    def _verify_identity(trial_key: str, identity_json: str | None) -> dict:
        if not identity_json:
            raise StorageError("trial identity payload is missing")
        try:
            payload = json.loads(identity_json)
        except json.JSONDecodeError as exc:
            raise StorageError("trial identity payload is malformed") from exc
        if not isinstance(payload, dict) or payload.get("content_hash") != trial_key:
            raise StorageError("trial identity payload does not match trial_key")
        try:
            validate_trial_identity_payload(trial_key, payload)
        except ValueError as exc:
            raise StorageError(f"trial identity schema is invalid: {exc}") from exc
        return payload

    def reserve_trial(self, trial, resume=True):
        if not trial.trial_key or not trial.identity_payload:
            raise StorageError("trial reservation requires TrialKey identity")
        identity_json = json.dumps(
            trial.identity_payload, sort_keys=True, separators=(",", ":")
        )
        self._verify_identity(trial.trial_key, identity_json)
        payload_json = json.dumps(trial.to_dict(), sort_keys=True, default=str)
        constraints_json = json.dumps(
            trial.constraints_snapshot, sort_keys=True, default=str
        )
        with self.conn:
            cursor = self.conn.execute(
                "INSERT OR IGNORE INTO trials("
                "id,status,params_hash,params,metrics,objective_value,payload,"
                "trial_key,lifecycle,identity_payload,constraints_payload"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    trial.id,
                    trial.status,
                    trial.params_hash,
                    json.dumps(trial.params, sort_keys=True, default=str),
                    "{}",
                    None,
                    payload_json,
                    trial.trial_key,
                    "pending",
                    identity_json,
                    constraints_json,
                ),
            )
            if cursor.rowcount == 1:
                return trial
            row = self.conn.execute(
                "SELECT lifecycle,identity_payload,payload FROM trials WHERE trial_key=?",
                (trial.trial_key,),
            ).fetchone()
            if row is None:
                raise StorageError("trial reservation conflict could not be read")
            stored_identity = self._verify_identity(trial.trial_key, row[1])
            if stored_identity != trial.identity_payload:
                raise StorageError(
                    "trial identity payload conflicts with persisted identity"
                )
            if row[0] in _TERMINAL:
                return None
            if not resume:
                return None
            self.conn.execute(
                "UPDATE trials SET id=?,status=?,lifecycle='pending',payload=? WHERE trial_key=?",
                (trial.id, trial.status, payload_json, trial.trial_key),
            )
            return trial

    def load_trial_by_key(self, trial_key):
        row = self.conn.execute(
            "SELECT payload,identity_payload FROM trials WHERE trial_key=?",
            (trial_key,),
        ).fetchone()
        if row is None:
            return None
        self._verify_identity(trial_key, row[1])
        return json.loads(row[0])

    def save_trial(self, trial):
        d = trial.to_dict()
        ph = (
            trial.params_hash
            or __import__("hashlib")
            .sha256(json.dumps(trial.params, sort_keys=True, default=str).encode())
            .hexdigest()
        )
        lifecycle = trial.lifecycle or (
            "completed" if trial.status == "completed" else "failed"
        )
        identity_json = (
            json.dumps(trial.identity_payload, sort_keys=True, separators=(",", ":"))
            if trial.identity_payload
            else None
        )
        if trial.trial_key and identity_json:
            self._verify_identity(trial.trial_key, identity_json)
        self.conn.execute(
            "INSERT INTO trials(id,status,params_hash,params,metrics,objective_value,payload,"
            "trial_key,lifecycle,identity_payload,constraints_payload) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status,params_hash=excluded.params_hash,"
            "params=excluded.params,metrics=excluded.metrics,objective_value=excluded.objective_value,"
            "payload=excluded.payload,trial_key=excluded.trial_key,lifecycle=excluded.lifecycle,"
            "identity_payload=excluded.identity_payload,constraints_payload=excluded.constraints_payload",
            (
                trial.id,
                trial.status,
                ph,
                json.dumps(trial.params, sort_keys=True, default=str),
                json.dumps(trial.metrics, sort_keys=True, default=str),
                trial.objective_value,
                json.dumps(d, sort_keys=True, default=str),
                trial.trial_key,
                lifecycle,
                identity_json,
                json.dumps(trial.constraints_snapshot, sort_keys=True, default=str),
            ),
        )
        self.conn.commit()

    def cancel_trial(self, trial_key, reason):
        row = self.conn.execute(
            "SELECT payload,identity_payload FROM trials WHERE trial_key=?",
            (trial_key,),
        ).fetchone()
        if row is None:
            raise StorageError("trial not found")
        self._verify_identity(trial_key, row[1])
        payload = json.loads(row[0])
        payload["lifecycle"] = "canceled"
        payload["status"] = "failed"
        payload["error_message"] = reason
        self.conn.execute(
            "UPDATE trials SET status='failed',lifecycle='canceled',payload=? WHERE trial_key=?",
            (json.dumps(payload, sort_keys=True, default=str), trial_key),
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

    def save_champion(self, trial, profile_name, selection_policy, constraints):
        if not trial.trial_key:
            raise StorageError("champion trial has no TrialKey")
        self.conn.execute(
            "INSERT OR REPLACE INTO champions(name,trial_key,selected_profile,selection_policy,"
            "constraints_payload,payload) VALUES('recommended',?,?,?,?,?)",
            (
                trial.trial_key,
                profile_name,
                json.dumps(selection_policy, sort_keys=True, default=str),
                json.dumps(constraints, sort_keys=True, default=str),
                json.dumps(trial.to_dict(), sort_keys=True, default=str),
            ),
        )
        self.conn.commit()

    def load_trials_raw(self):
        rows = self.conn.execute(
            "SELECT payload,trial_key,identity_payload FROM trials ORDER BY id"
        ).fetchall()
        result = []
        for payload_json, trial_key, identity_json in rows:
            if trial_key is not None:
                self._verify_identity(trial_key, identity_json)
            result.append(json.loads(payload_json))
        return result

    def close(self):
        if not self._closed:
            try:
                self.conn.close()
            finally:
                self._closed = True

    def __del__(self):
        try:
            self.close()
        except (AttributeError, sqlite3.Error):
            return
