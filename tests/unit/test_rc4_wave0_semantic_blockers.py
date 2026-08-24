import json
from pathlib import Path
from typing import Any

import pytest
from openpine_contracts import (
    SchemaValidationError,
    list_schema_ids,
    seal_content_hash,
    validate_payload,
    verify_content_hash,
)

import optimizer.engine as engine_module
from optimizer import OptimizerConfig, Parameter, optimize
from optimizer.core.diagnostic import Diagnostic
from optimizer.core.durable_trial import bind_trial_identity, pending_trial
from optimizer.core.trial_key import (
    TRIAL_SCHEMA_ID,
    TrialIdentity,
    _policy_value,
    contract_schema_hashes,
    validate_critical_identity_hashes,
    validate_trial_identity_payload,
)
from optimizer.engine import _run_reserved
from optimizer.errors import ParameterValidationError, StorageError
from optimizer.results.trial import Trial
from optimizer.storage.json_backend import JsonStorage
from optimizer.storage.sqlite_backend import SQLiteStorage


def _sha256(hex_digit: str) -> str:
    return f"sha256:{hex_digit * 64}"


def _identity() -> TrialIdentity:
    return TrialIdentity(
        generated_artifact_hash=_sha256("1"),
        data_snapshot_series_hash=_sha256("2"),
        parameters={"length": 10},
        engine_build_hash=_sha256("3"),
        engine_config_hash=_sha256("4"),
        semantic_profile="strict_5x",
        finality_policy={"bars": "FINAL"},
        warmup_policy={"mode": "CALC_ONLY"},
        score_policy={"window": "closed"},
        end_policy={"mode": "liquidate"},
        contract_schema_hashes=contract_schema_hashes(),
        stack_manifest_hash=_sha256("5"),
        deterministic_seed=42,
        fold_identity=None,
        walk_forward_identity=None,
        objective_version="objective.v1",
        constraints_version="constraints.v1",
        producer_commit="a" * 40,
        optimizer_id="optimizer-1",
        strategy_id="strategy-1",
        source_hash=_sha256("6"),
        emitted_module_hash=_sha256("7"),
        numeric_policy="decimal-string.v1",
        fill_policy="backtest-engine.v1",
    )


def _config(output_dir: Path, **changes: object) -> OptimizerConfig:
    values: dict[str, Any] = {
        "output_dir": output_dir,
        "timeout_per_trial_sec": 0,
        "generated_artifact_hash": _sha256("1"),
        "data_snapshot_series_hash": _sha256("2"),
        "data_fingerprint": _sha256("6"),
        "engine_build_hash": _sha256("3"),
        "engine_config_hash": _sha256("4"),
        "stack_manifest_hash": _sha256("5"),
        "optimizer_commit": "a" * 40,
        "optimizer_id": "optimizer-1",
        "strategy_id": "strategy-1",
        "source_hash": _sha256("7"),
        "emitted_module_hash": _sha256("8"),
    }
    values.update(changes)
    return OptimizerConfig(**values)


class _RecordingStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reserve_trial(self, trial: Trial, resume: bool = True) -> Trial:
        self.events.append("reserve_trial")
        return trial

    def save_trial(self, trial: Trial) -> None:
        self.events.append("save_trial")


def _hash_valid_schema_invalid_trial() -> Trial:
    identity_payload = seal_content_hash(
        {"schema_id": TRIAL_SCHEMA_ID, "marker": "schema-invalid"},
        schema_id=TRIAL_SCHEMA_ID,
    )
    assert verify_content_hash(identity_payload, schema_id=TRIAL_SCHEMA_ID)
    return Trial.pending(
        1,
        {"length": 10},
        trial_key=identity_payload["content_hash"],
        identity_payload=identity_payload,
        params_hash=_sha256("7"),
        objective_direction="maximize",
        parameter_space_hash=_sha256("8"),
        optimizer_config_hash=_sha256("9"),
        constraints_snapshot={},
    )


def _storage(kind: str, output_dir: Path) -> JsonStorage | SQLiteStorage:
    if kind == "json":
        return JsonStorage(output_dir)
    return SQLiteStorage(output_dir)


def _close(store: JsonStorage | SQLiteStorage) -> None:
    if isinstance(store, SQLiteStorage):
        store.close()


def _persist_without_validation(
    store: JsonStorage | SQLiteStorage, trial: Trial
) -> None:
    if isinstance(store, JsonStorage):
        store.path.write_text(json.dumps(trial.to_dict()) + "\n", encoding="utf-8")
        return
    store.conn.execute(
        "INSERT INTO trials(id,payload,trial_key,identity_payload) VALUES(?,?,?,?)",
        (
            trial.id,
            json.dumps(trial.to_dict(), sort_keys=True),
            trial.trial_key,
            json.dumps(trial.identity_payload, sort_keys=True),
        ),
    )
    store.conn.commit()


def test_trial_identity_seal_validates_against_packaged_identity_schema() -> None:
    catalog = set(list_schema_ids(include_aliases=False))
    assert (
        TRIAL_SCHEMA_ID in catalog
    ), f"packaged contracts catalog must provide TrialIdentity schema {TRIAL_SCHEMA_ID!r}"

    sealed = _identity().seal()
    try:
        validate_payload(TRIAL_SCHEMA_ID, sealed.payload)
    except SchemaValidationError as exc:
        pytest.fail(
            "TrialIdentity.seal() payload does not satisfy its packaged identity schema: "
            f"{exc}"
        )


@pytest.mark.parametrize(
    "field",
    [
        "generated_artifact_hash",
        "data_snapshot_series_hash",
        "engine_build_hash",
        "engine_config_hash",
        "stack_manifest_hash",
    ],
)
@pytest.mark.parametrize(
    "invalid_hash",
    [None, "", "md5:not-a-sha256"],
    ids=["none", "empty", "malformed"],
)
def test_critical_identity_hash_is_rejected_before_reserve_store_or_runner(
    tmp_path: Path, field: str, invalid_hash: object
) -> None:
    events: list[str] = []
    rejection: ParameterValidationError | None = None

    def runner(_params: dict[str, object]) -> dict[str, float]:
        events.append("runner")
        return {"net_profit": 1.0}

    try:
        config = _config(tmp_path, **{field: invalid_hash})
        _run_reserved(
            1,
            {"length": 10},
            runner,
            config,
            _sha256("8"),
            _sha256("9"),
            _RecordingStore(events),
        )
    except ParameterValidationError as exc:
        rejection = exc

    assert rejection is not None and events == [], (
        f"critical identity {field}={invalid_hash!r} must be rejected before side effects; "
        f"rejection={rejection!r}, events={events!r}"
    )


@pytest.mark.parametrize("storage_kind", ["json", "sqlite"])
def test_storage_rejects_hash_valid_schema_invalid_identity_on_reserve(
    tmp_path: Path, storage_kind: str
) -> None:
    store = _storage(storage_kind, tmp_path / storage_kind)
    try:
        with pytest.raises(StorageError, match="schema|identity"):
            store.reserve_trial(_hash_valid_schema_invalid_trial())
    finally:
        _close(store)


@pytest.mark.parametrize("storage_kind", ["json", "sqlite"])
def test_storage_rejects_hash_valid_schema_invalid_identity_on_load(
    tmp_path: Path, storage_kind: str
) -> None:
    store = _storage(storage_kind, tmp_path / storage_kind)
    trial = _hash_valid_schema_invalid_trial()
    _persist_without_validation(store, trial)
    try:
        with pytest.raises(StorageError, match="schema|identity"):
            store.load_trial_by_key(trial.trial_key)
    finally:
        _close(store)


def test_rc4_identity_validation_edge_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _policy_value({"mode": 1}, "mode") == ""
    assert _policy_value("DIRECT", "mode") == "DIRECT"
    assert _policy_value(1, "mode") == ""

    with pytest.raises(ValueError, match="not configured"):
        pending_trial(1, {"x": 1}, OptimizerConfig(output_dir=tmp_path), "s", "c")

    sealed = _identity().seal()
    with pytest.raises(ValueError, match="trial_key"):
        validate_trial_identity_payload(_sha256("f"), sealed.payload)
    monkeypatch.setattr(
        "optimizer.core.trial_key.verify_content_hash", lambda *_a, **_k: False
    )
    with pytest.raises(ValueError, match="content hash"):
        validate_trial_identity_payload(sealed.trial_key, sealed.payload)

    invalid_commit = _config(tmp_path, optimizer_commit="not-a-commit")
    with pytest.raises(ParameterValidationError, match="optimizer_commit"):
        validate_critical_identity_hashes(invalid_commit)


def test_rc4_timeout_binding_and_json_storage_edge_branches(tmp_path: Path) -> None:
    pending = pending_trial(1, {"x": 1}, _config(tmp_path), "space", "config")
    timed = Trial.pending(
        1,
        {"x": 1},
        trial_key=pending.trial_key or "",
        identity_payload=pending.identity_payload or {},
        params_hash="params",
        objective_direction="maximize",
        parameter_space_hash="space",
        optimizer_config_hash="config",
        constraints_snapshot={},
    )
    timed.diagnostics = [Diagnostic("TRIAL_TIMEOUT", "timeout", "error")]
    assert bind_trial_identity(timed, pending).lifecycle == "timeout"

    champion_store = JsonStorage(tmp_path / "champion")
    champion_store.save_champion(pending, "best", {}, {})
    assert (tmp_path / "champion" / "champion.json").is_file()

    missing_store = JsonStorage(tmp_path / "missing")
    missing_row = pending.to_dict()
    missing_row["identity_payload"] = None
    missing_store.path.write_text(json.dumps(missing_row) + "\n", encoding="utf-8")
    with pytest.raises(StorageError, match="missing"):
        missing_store.load_trial_by_key(pending.trial_key)

    invalid_store = JsonStorage(tmp_path / "invalid")
    invalid_row = pending.to_dict()
    invalid_row["identity_payload"] = []
    invalid_store.path.write_text(json.dumps(invalid_row) + "\n", encoding="utf-8")
    with pytest.raises(StorageError, match="invalid"):
        invalid_store.reserve_trial(pending)


def test_adaptive_refinement_executes_one_new_parameter_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        engine_module.adaptive_grid, "refine", lambda *_a, **_k: [{"x": 2}]
    )
    seen: list[int] = []
    result = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        lambda params: seen.append(int(params["x"]))
        or {"net_profit": float(params["x"])},
        OptimizerConfig(
            algorithm="adaptive_grid",
            max_trials=2,
            grid_max_combinations=1,
            output_dir=tmp_path,
            storage_backend="json",
            report_profiles=False,
            use_profile_auto_constraints=False,
            timeout_per_trial_sec=0,
        ),
    )
    assert seen == [1, 2]
    assert not isinstance(result, dict)
    assert len(result.trials) == 2
