from pathlib import Path

from openpine_contracts import list_schema_ids

RC5_CONTRACTS_SHA = "6b5e67445e2772057cd877e158c7aa0c58bdfe37"
RC6_CONTRACTS_SHA = "904e8f660834a10d3382cd1b2ed7380c24b73072"


def test_contracts_dependency_and_workflow_are_pinned_to_rc6() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert '"openpine-contracts==5.0.0rc6"' in text
    assert "git+" not in text
    assert workflow.count(f"ref: {RC6_CONTRACTS_SHA}") == 1
    assert RC5_CONTRACTS_SHA not in workflow


def test_contracts_catalog() -> None:
    schema_ids = set(list_schema_ids())
    assert "openpine.generated_artifact.v3" in schema_ids
    assert "openpine.execution_context.v1" in schema_ids
    assert "openpine.trial.identity.v1" in schema_ids
    assert "openpine.trial.v2" in schema_ids


def test_ci_triggers_and_concurrency_key() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "push:\n    branches: [main]" in workflow
    assert "pull_request:\n    branches: [main]" in workflow
    assert "github.event.pull_request.number || github.ref" in workflow
