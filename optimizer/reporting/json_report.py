import json


def to_json(result, path=None):
    data = result.to_dict()
    text = json.dumps(data, indent=2, default=str, sort_keys=True)
    if path:
        open(path, "w").write(text)
    return text
