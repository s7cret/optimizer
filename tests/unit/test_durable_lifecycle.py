import json
import sqlite3
import gc
from contextlib import closing
from pathlib import Path

import pytest

from optimizer import OptimizerConfig, Parameter, optimize
from optimizer.engine import _pending_trial, _run_jobs, _run_reserved
from optimizer.runners.backtest_engine import _metric_dict
from optimizer.core.trial_runner import run_one
from optimizer.errors import StorageError
from optimizer.results.leaderboard import rank_trials
from optimizer.results.trial import Trial
from optimizer.storage.json_backend import JsonStorage
from optimizer.storage.sqlite_backend import SQLiteStorage


def _sha256(digit: str) -> str:
    return "sha256:" + digit * 64


def _config(output_dir: Path, **changes: object) -> OptimizerConfig:
    values: dict[str, object] = {
        "output_dir": output_dir,
        "storage_backend": "sqlite",
        "max_trials": 1,
        "report_profiles": False,
        "use_profile_auto_constraints": False,
        "timeout_per_trial_sec": 0,
        "runner_fingerprint": _sha256("1"),
        "generated_artifact_hash": _sha256("2"),
        "data_fingerprint": _sha256("3"),
        "data_snapshot_series_hash": _sha256("4"),
        "engine_build_hash": _sha256("5"),
        "engine_config_hash": _sha256("6"),
        "stack_manifest_hash": _sha256("7"),
        "optimizer_commit": "a" * 40,
        "optimizer_id": "optimizer-1",
        "strategy_id": "strategy-1",
        "source_hash": _sha256("8"),
        "emitted_module_hash": _sha256("9"),
        "semantic_profile": "strict_5x",
        "finality_policy": {"bars": "FINAL"},
        "warmup_policy": {"mode": "CALC_ONLY"},
        "score_policy": {"window": "closed"},
        "end_policy": {"mode": "liquidate"},
        "objective_version": "net-profit.v2",
        "constraints_version": "risk.v3",
    }
    values.update(changes)
    return OptimizerConfig(**values)


def test_sqlite_storage_finalizer_is_idempotent(tmp_path: Path) -> None:
    store = SQLiteStorage(tmp_path)
    store.close()
    store.close()
    store.__del__()
    assert store._closed is True


def test_sqlite_storage_partial_init_finalizer_is_silent(tmp_path: Path) -> None:
    invalid_output = tmp_path / "not-a-directory"
    invalid_output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        SQLiteStorage(invalid_output)
    gc.collect()


def test_sqlite_storage_finalizer_suppresses_cleanup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteStorage(tmp_path)

    def fail_close() -> None:
        raise sqlite3.ProgrammingError("wrong thread")

    monkeypatch.setattr(store, "close", fail_close)
    store.__del__()
    store.conn.close()
    store._closed = True


def _winner(trial_id: int, trial_key: str) -> Trial:
    return Trial(
        trial_id,
        {"x": trial_id},
        {"net_profit": 10.0},
        10.0,
        "maximize",
        None,
        True,
        {},
        0,
        1.0,
        None,
        None,
        None,
        None,
        0.0,
        "completed",
        trial_key=trial_key,
        lifecycle="completed",
    )


def test_sqlite_reserves_trial_before_runner_and_unique_key_blocks_reexecution(
    tmp_path: Path,
) -> None:
    calls = 0

    def runner(params: dict[str, object]) -> dict[str, float]:
        nonlocal calls
        calls += 1
        with closing(sqlite3.connect(tmp_path / "optimizer.sqlite")) as conn, conn:
            row = conn.execute(
                "SELECT trial_key, lifecycle, identity_payload FROM trials"
            ).fetchone()
        assert row is not None
        assert row[0].startswith("sha256:")
        assert row[1] == "pending"
        assert json.loads(row[2])["content_hash"] == row[0]
        return {"net_profit": float(params["x"])}

    cfg = _config(tmp_path)
    first = optimize([Parameter("x", "int", 1, 1, 1, 1)], runner, cfg)
    second = optimize([Parameter("x", "int", 1, 1, 1, 1)], runner, cfg)

    assert calls == 1
    assert first.trials[0].trial_key == second.trials[0].trial_key
    with closing(sqlite3.connect(tmp_path / "optimizer.sqlite")) as conn, conn:
        rows = conn.execute(
            "SELECT trial_key, lifecycle, status, COUNT(*) FROM trials GROUP BY trial_key"
        ).fetchall()
        unique_indexes = [
            row for row in conn.execute("PRAGMA index_list(trials)") if row[2] == 1
        ]
    assert len(rows) == 1
    assert rows[0][1:] == ("completed", "completed", 1)
    assert unique_indexes


def test_resume_reexecutes_only_unfinished_exact_identity(tmp_path: Path) -> None:
    calls = 0

    def runner(params: dict[str, object]) -> dict[str, float]:
        nonlocal calls
        calls += 1
        return {"net_profit": float(params["x"])}

    cfg = _config(tmp_path)
    optimize([Parameter("x", "int", 1, 1, 1, 1)], runner, cfg)
    with closing(sqlite3.connect(tmp_path / "optimizer.sqlite")) as conn, conn:
        payload = json.loads(conn.execute("SELECT payload FROM trials").fetchone()[0])
        payload["status"] = "running"
        payload["lifecycle"] = "running"
        conn.execute(
            "UPDATE trials SET status='running', lifecycle='running', payload=?",
            (json.dumps(payload, sort_keys=True),),
        )
        conn.commit()

    resumed = optimize([Parameter("x", "int", 1, 1, 1, 1)], runner, cfg)
    assert calls == 2
    assert resumed.trials[0].lifecycle == "completed"

    with closing(sqlite3.connect(tmp_path / "optimizer.sqlite")) as conn, conn:
        payload = json.loads(
            conn.execute("SELECT identity_payload FROM trials").fetchone()[0]
        )
        payload["engine_build_hash"] = "sha256:tampered"
        conn.execute(
            "UPDATE trials SET status='pending', lifecycle='pending', identity_payload=?",
            (json.dumps(payload, sort_keys=True),),
        )
        conn.commit()
    with pytest.raises(StorageError, match="identity schema"):
        optimize([Parameter("x", "int", 1, 1, 1, 1)], runner, cfg)


def test_failed_timeout_and_canceled_lifecycles_are_terminal(tmp_path: Path) -> None:
    calls = 0

    def broken(_params: dict[str, object]) -> dict[str, float]:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    cfg = _config(tmp_path / "failed")
    first = optimize([Parameter("x", "int", 1, 1, 1, 1)], broken, cfg)
    second = optimize([Parameter("x", "int", 1, 1, 1, 1)], broken, cfg)
    assert calls == 1
    assert first.trials[0].lifecycle == second.trials[0].lifecycle == "failed"

    timed = run_one(
        1,
        {"x": 1},
        lambda _params: (_ for _ in ()).throw(TimeoutError()),
        _config(tmp_path / "timeout"),
        "space",
        "config",
    )
    assert timed.status == "failed"
    assert timed.lifecycle == "timeout"

    store = SQLiteStorage(tmp_path / "canceled")
    pending = _pending_trial(
        1,
        {"x": 1},
        _config(tmp_path / "canceled"),
        "space",
        "config",
    )
    assert store.reserve_trial(pending, resume=False) is not None
    store.cancel_trial(pending.trial_key, "fail-fast")
    assert store.reserve_trial(pending, resume=True) is None
    assert store.load_trials_raw()[0]["lifecycle"] == "canceled"
    store.close()


def test_champion_ties_and_persistence_are_deterministic(tmp_path: Path) -> None:
    low_key = _winner(20, "sha256:aaa")
    high_key = _winner(1, "sha256:bbb")
    cfg = _config(
        tmp_path,
        selection_mode="best_after_constraints",
        constraints={"net_profit": {"min": 0, "hard": True}},
    )

    assert rank_trials([high_key, low_key], cfg)[0].trial_key == "sha256:aaa"
    assert rank_trials([low_key, high_key], cfg)[0].trial_key == "sha256:aaa"

    result = optimize(
        [Parameter("x", "int", 1, 1, 2, 1)],
        lambda _params: {"net_profit": 10.0},
        cfg,
    )
    assert result.recommended_trial is not None
    with closing(sqlite3.connect(tmp_path / "optimizer.sqlite")) as conn, conn:
        champion = conn.execute(
            "SELECT trial_key, selected_profile, selection_policy, constraints_payload "
            "FROM champions WHERE name='recommended'"
        ).fetchone()
    assert champion is not None
    assert champion[0] == result.recommended_trial.trial_key
    assert champion[1] == result.recommended_profile
    assert json.loads(champion[2])["selection_mode"] == "best_after_constraints"
    assert json.loads(champion[3]) == {"net_profit": {"hard": True, "min": 0}}


def test_trial_stores_reject_unsealed_incoming_identity(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    valid = _pending_trial(1, {"x": 1}, cfg, "space", "config")
    tampered = Trial.pending(
        1,
        {"x": 1},
        trial_key=valid.trial_key or "",
        identity_payload={**(valid.identity_payload or {}), "tampered": True},
        params_hash="params",
        objective_direction="maximize",
        parameter_space_hash="space",
        optimizer_config_hash="config",
        constraints_snapshot={},
    )

    for store in (JsonStorage(tmp_path / "json"), SQLiteStorage(tmp_path / "sqlite")):
        with pytest.raises(StorageError, match="identity"):
            store.reserve_trial(tampered)
        close = getattr(store, "close", None)
        if callable(close):
            close()


def test_json_storage_identity_resume_conflict_and_cancel_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    pending = _pending_trial(1, {"x": 1}, cfg, "space", "config")
    store = JsonStorage(tmp_path)

    empty = Trial.pending(
        9,
        {},
        trial_key="",
        identity_payload={},
        params_hash="params",
        objective_direction="maximize",
        parameter_space_hash="space",
        optimizer_config_hash="config",
        constraints_snapshot={},
    )
    with pytest.raises(StorageError, match="requires TrialKey"):
        store.reserve_trial(empty)
    with pytest.raises(StorageError, match="not found"):
        store.cancel_trial("sha256:missing", "missing")

    assert store.reserve_trial(pending) is not None
    assert store.reserve_trial(pending, resume=False) is None
    assert store.reserve_trial(pending, resume=True) is not None

    bad_row = pending.to_dict()
    bad_row["identity_payload"] = {"content_hash": pending.trial_key}
    store.path.write_text(json.dumps(bad_row) + "\n")
    with pytest.raises(StorageError, match="invalid"):
        store.reserve_trial(pending)

    store.path.unlink()
    assert store.reserve_trial(pending) is not None
    conflicting = Trial.pending(
        pending.id,
        pending.params,
        trial_key=pending.trial_key or "",
        identity_payload={**(pending.identity_payload or {}), "collision": True},
        params_hash=pending.params_hash or "",
        objective_direction=pending.objective_direction,
        parameter_space_hash=pending.parameter_space_hash or "",
        optimizer_config_hash=pending.optimizer_config_hash or "",
        constraints_snapshot={},
    )
    monkeypatch.setattr(
        "optimizer.storage.json_backend.validate_trial_identity_payload",
        lambda *_args, **_kwargs: True,
    )
    with pytest.raises(StorageError, match="conflicts"):
        store.reserve_trial(conflicting)

    store.cancel_trial(pending.trial_key, "stopped")
    canceled = store.load_trial_by_key(pending.trial_key)
    assert canceled is not None
    assert canceled["lifecycle"] == "canceled"


def test_sqlite_storage_migration_identity_conflict_and_missing_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "migrate"
    root.mkdir()
    db = root / "optimizer.sqlite"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.executescript(
            "CREATE TABLE trials(id INTEGER PRIMARY KEY,status TEXT,params_hash TEXT,"
            "params TEXT,metrics TEXT,objective_value REAL,payload TEXT);"
        )
    migrated = SQLiteStorage(root)
    columns = {row[1] for row in migrated.conn.execute("PRAGMA table_info(trials)")}
    assert {
        "trial_key",
        "lifecycle",
        "identity_payload",
        "constraints_payload",
    } <= columns
    migrated.close()

    for raw, message in (
        (None, "missing"),
        ("not-json", "malformed"),
        (json.dumps([]), "does not match"),
        (json.dumps({"content_hash": "sha256:other"}), "does not match"),
        (json.dumps({"content_hash": "sha256:key"}), "identity schema is invalid"),
    ):
        with pytest.raises(StorageError, match=message):
            SQLiteStorage._verify_identity("sha256:key", raw)

    store = SQLiteStorage(tmp_path / "sqlite")
    cfg = _config(tmp_path)
    first = _pending_trial(1, {"x": 1}, cfg, "space", "config")
    with pytest.raises(StorageError, match="requires TrialKey"):
        store.reserve_trial(
            Trial.pending(
                9,
                {},
                trial_key="",
                identity_payload={},
                params_hash="params",
                objective_direction="maximize",
                parameter_space_hash="space",
                optimizer_config_hash="config",
                constraints_snapshot={},
            )
        )
    assert store.reserve_trial(first) is not None
    assert store.reserve_trial(first, resume=False) is None
    assert store.reserve_trial(first, resume=True) is not None
    assert store.load_trial_by_key("sha256:missing") is None
    with pytest.raises(StorageError, match="not found"):
        store.cancel_trial("sha256:missing", "missing")
    with pytest.raises(StorageError, match="no TrialKey"):
        store.save_champion(_winner(2, ""), "best", {}, {})

    second = _pending_trial(1, {"x": 2}, cfg, "space", "config")
    with pytest.raises(StorageError, match="could not be read"):
        store.reserve_trial(second)

    conflicting = Trial.pending(
        first.id,
        first.params,
        trial_key=first.trial_key or "",
        identity_payload={**(first.identity_payload or {}), "collision": True},
        params_hash=first.params_hash or "",
        objective_direction=first.objective_direction,
        parameter_space_hash=first.parameter_space_hash or "",
        optimizer_config_hash=first.optimizer_config_hash or "",
        constraints_snapshot={},
    )
    monkeypatch.setattr(
        "optimizer.storage.sqlite_backend.validate_trial_identity_payload",
        lambda *_args, **_kwargs: True,
    )
    with pytest.raises(StorageError, match="conflicts"):
        store.reserve_trial(conflicting)
    store.close()


def test_engine_legacy_parallel_dedup_and_missing_reservation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, max_parallel=1)

    class LegacyStore:
        def __init__(self) -> None:
            self.saved: list[Trial] = []

        def save_trial(self, trial: Trial) -> None:
            self.saved.append(trial)

    def runner(params: dict[str, object]) -> dict[str, float]:
        value = params["x"]
        assert isinstance(value, (int, float))
        return {"net_profit": float(value)}

    legacy = LegacyStore()
    one = _run_reserved(1, {"x": 1}, runner, cfg, "space", "config", legacy)
    assert one is not None
    assert one.status == "completed"
    assert legacy.saved == [one]

    class MissingReservation:
        def reserve_trial(self, _trial: Trial, resume: bool = True) -> None:
            return None

        def load_trial_by_key(self, _trial_key: str) -> None:
            return None

    assert (
        _run_jobs(
            [(1, {"x": 1}), (2, {"x": 2})],
            runner,
            _config(tmp_path / "missing", max_parallel=2),
            "space",
            "config",
            MissingReservation(),
        )
        == []
    )
    assert (
        _run_jobs(
            [(1, {"x": 1})],
            runner,
            cfg,
            "space",
            "config",
            MissingReservation(),
        )
        == []
    )

    parallel_cfg = _config(tmp_path / "parallel", max_parallel=2)
    store = JsonStorage(tmp_path / "parallel")
    calls: list[int] = []

    def counted(params: dict[str, object]) -> dict[str, float]:
        value = params["x"]
        assert isinstance(value, (int, float))
        calls.append(int(value))
        return {"net_profit": float(value)}

    _run_reserved(1, {"x": 1}, counted, parallel_cfg, "space", "config", store)
    trials = _run_jobs(
        [(2, {"x": 1}), (3, {"x": 2})],
        counted,
        parallel_cfg,
        "space",
        "config",
        store,
    )
    assert sorted(t.params["x"] for t in trials) == [1, 2]
    assert calls == [1, 2]

    real_run_reserved = _run_reserved

    def skip_baseline(*args, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("is_baseline") or (len(args) > 7 and args[7] is True):
            return None
        return real_run_reserved(*args, **kwargs)

    monkeypatch.setattr("optimizer.engine._run_reserved", skip_baseline)
    result = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        counted,
        _config(tmp_path / "baseline", baseline_params={"x": 1}),
    )
    assert not isinstance(result, dict)
    assert result.baseline_trial is None

    assert _metric_dict(
        {"finite": 1, "infinite": float("inf")}, {"finite", "infinite"}
    ) == {"finite": 1.0}
