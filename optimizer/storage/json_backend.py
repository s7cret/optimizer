import json
import os
from pathlib import Path


class JsonStorage:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / 'trials.jsonl'; self.meta_path = self.output_dir / 'run.json'

    def _atomic_write(self, path: Path, text: str):
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(text)
        os.replace(tmp, path)

    def init_run(self, fingerprints):
        self._atomic_write(self.meta_path, json.dumps(fingerprints, sort_keys=True, indent=2, default=str))

    def load_meta(self): return json.loads(self.meta_path.read_text()) if self.meta_path.exists() else None

    def save_trial(self, trial):
        rows = self.load_trials_raw()
        d = trial.to_dict()
        ph = d.get('params_hash')
        out = []
        replaced = False
        for row in rows:
            if (ph and row.get('params_hash') == ph) or row.get('id') == d.get('id'):
                if not replaced:
                    out.append(d); replaced = True
            else:
                out.append(row)
        if not replaced: out.append(d)
        text = ''.join(json.dumps(x, default=str, sort_keys=True) + '\n' for x in out)
        self._atomic_write(self.path, text)

    def load_trials_raw(self):
        if not self.path.exists(): return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
