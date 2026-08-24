import json
import os
from pathlib import Path

from optimizer.core.trial_key import validate_trial_identity_payload
from optimizer.errors import StorageError


class JsonStorage:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "trials.jsonl"
        self.meta_path = self.output_dir / "run.json"

    def _atomic_write(self, path: Path, text: str):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    def init_run(self, fingerprints):
        self._atomic_write(
            self.meta_path,
            json.dumps(fingerprints, sort_keys=True, indent=2, default=str),
        )

    def load_meta(self):
        return (
            json.loads(self.meta_path.read_text()) if self.meta_path.exists() else None
        )

    def save_trial(self, trial):
        rows = self.load_trials_raw()
        d = trial.to_dict()
        ph = d.get("params_hash")
        out = []
        replaced = False
        for row in rows:
            if (ph and row.get("params_hash") == ph) or row.get("id") == d.get("id"):
                if not replaced:
                    out.append(d)
                    replaced = True
            else:
                out.append(row)
        if not replaced:
            out.append(d)
        text = "".join(json.dumps(x, default=str, sort_keys=True) + "\n" for x in out)
        self._atomic_write(self.path, text)

    def reserve_trial(self, trial, resume=True):
        if not trial.trial_key or not trial.identity_payload:
            raise StorageError("trial reservation requires TrialKey identity")
        try:
            validate_trial_identity_payload(trial.trial_key, trial.identity_payload)
        except ValueError as exc:
            raise StorageError(f"trial identity schema is invalid: {exc}") from exc
        for row in self.load_trials_raw():
            if row.get("trial_key") != trial.trial_key:
                continue
            identity = row.get("identity_payload")
            if not isinstance(identity, dict):
                raise StorageError("trial identity payload is invalid")
            try:
                validate_trial_identity_payload(trial.trial_key, identity)
            except ValueError as exc:
                raise StorageError(f"trial identity schema is invalid: {exc}") from exc
            if identity != trial.identity_payload:
                raise StorageError(
                    "trial identity payload conflicts with persisted identity"
                )
            if row.get("lifecycle") in {"completed", "failed", "timeout", "canceled"}:
                return None
            if not resume:
                return None
            break
        self.save_trial(trial)
        return trial

    def load_trial_by_key(self, trial_key):
        row = next(
            (
                row
                for row in self.load_trials_raw()
                if row.get("trial_key") == trial_key
            ),
            None,
        )
        if row is not None:
            identity = row.get("identity_payload")
            if not isinstance(identity, dict):
                raise StorageError("trial identity payload is missing")
            try:
                validate_trial_identity_payload(trial_key, identity)
            except ValueError as exc:
                raise StorageError(f"trial identity schema is invalid: {exc}") from exc
        return row

    def cancel_trial(self, trial_key, reason):
        row = self.load_trial_by_key(trial_key)
        if row is None:
            raise StorageError("trial not found")
        row["status"] = "failed"
        row["lifecycle"] = "canceled"
        row["error_message"] = reason
        rows = [
            row if item.get("trial_key") == trial_key else item
            for item in self.load_trials_raw()
        ]
        self._atomic_write(
            self.path,
            "".join(
                json.dumps(item, default=str, sort_keys=True) + "\n" for item in rows
            ),
        )

    def save_champion(self, trial, profile_name, selection_policy, constraints):
        self._atomic_write(
            self.output_dir / "champion.json",
            json.dumps(
                {
                    "trial_key": trial.trial_key,
                    "selected_profile": profile_name,
                    "selection_policy": selection_policy,
                    "constraints": constraints,
                    "trial": trial.to_dict(),
                },
                sort_keys=True,
                indent=2,
                default=str,
            ),
        )

    def load_trials_raw(self):
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
