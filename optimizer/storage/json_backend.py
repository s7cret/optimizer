import json
from pathlib import Path
from optimizer.results.trial import Trial
class JsonStorage:
    def __init__(self, output_dir): self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True,exist_ok=True); self.path=self.output_dir/'trials.jsonl'; self.meta_path=self.output_dir/'run.json'
    def init_run(self, fingerprints): self.meta_path.write_text(json.dumps(fingerprints,sort_keys=True,indent=2,default=str))
    def load_meta(self): return json.loads(self.meta_path.read_text()) if self.meta_path.exists() else None
    def save_trial(self, trial):
        with self.path.open('a') as f: f.write(json.dumps(trial.to_dict(),default=str,sort_keys=True)+'\n'); f.flush()
    def load_trials_raw(self):
        if not self.path.exists(): return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
