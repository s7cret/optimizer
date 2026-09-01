from pathlib import Path

from openpine_contracts import list_schema_ids


def test_contracts_pin_and_catalog() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert '"openpine-contracts==5.0.0rc6"' in text
    assert "git+" not in text
    schema_ids = set(list_schema_ids())
    assert "openpine.generated_artifact.v3" in schema_ids
    assert "openpine.execution_context.v1" in schema_ids
    assert "openpine.trial.identity.v1" in schema_ids
    assert "openpine.trial.v2" in schema_ids
    assert "ref: 6b5e67445e2772057cd877e158c7aa0c58bdfe37" in workflow
    assert "push:\n    branches: [main]" in workflow
    assert "pull_request:\n    branches: [main]" in workflow
    assert "github.event.pull_request.number || github.ref" in workflow
