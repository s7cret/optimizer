from pathlib import Path

from openpine_contracts import list_schema_ids


def test_contracts_pin_and_catalog() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert (
        "openpine-contracts @ git+https://github.com/s7cret/openpine-contracts.git@"
        "af9ecbc455e9af83cdc609f6b6ff85c40fb6c8bb" in text
    )
    assert "openpine.trial.v2" in list_schema_ids()
