import json
from pathlib import Path


def to_json(result, path=None):
    data = result.to_dict()
    text = json.dumps(data, indent=2, default=str, sort_keys=True)
    if path:
        Path(path).write_text(text, encoding="utf-8")
    return text
