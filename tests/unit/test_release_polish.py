from __future__ import annotations

import builtins
import json
import runpy
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from optimizer.cli import main as cli_main
from optimizer.cli.main import load_obj, load_params, main
from optimizer.core import process_containment
from optimizer.core import trial_runner
from optimizer.core.parameter import Parameter
from optimizer.core.parameter_space import ParameterSpace
from optimizer.distribution import build_zip, iter_files, manifest
from optimizer.errors import ParameterValidationError
from optimizer.runners import backtest_engine as backtest_runner
from optimizer.version import __version__


def test_release_version_is_4_0_1() -> None:
    assert __version__ == "4.0.1"


def test_stable_hash_falls_back_when_backtest_engine_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fail_backtest_hash(name, *args, **kwargs):
        if name == "backtest_engine.core.deterministic_hash":
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_backtest_hash)

    assert len(backtest_runner._stable_hash({"seed": 7})) == 64


def test_backtest_identity_normalizes_dataclasses_mappings_and_sequences() -> None:
    @dataclass
    class Payload:
        value: int

    assert backtest_runner._identity_value(
        {"payload": Payload(7), "sequence": (1, 2)}
    ) == {"payload": {"value": 7}, "sequence": [1, 2]}


def test_process_timeout_rejects_unpicklable_request_without_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        timeout_backend = "process"
        timeout_per_trial_sec = 1

    assert trial_runner._is_picklable({"value": 1}) is True
    assert trial_runner._is_picklable(lambda: None) is False
    monkeypatch.setattr(trial_runner.mp, "get_all_start_methods", lambda: ["spawn"])

    with pytest.raises(ValueError, match="picklable runner and request"):
        trial_runner._select_timeout_backend(
            Config(), lambda value: value, {"value": 1}, [], 1, "params"
        )


def test_process_timeout_kills_runner_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived"

    def runner(_payload):
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib,time; time.sleep(0.5); "
                    f"pathlib.Path({str(marker)!r}).write_text('survived')"
                ),
            ]
        )
        time.sleep(5)

    with pytest.raises(__import__("concurrent.futures").futures.TimeoutError):
        trial_runner._call_runner_in_process(runner, {}, 0.2)
    time.sleep(0.7)

    assert not marker.exists()


def test_process_timeout_kills_detached_runner_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)
    marker = tmp_path / "detached-descendant-survived"

    def runner(_payload):
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib,time; time.sleep(0.6); "
                    f"pathlib.Path({str(marker)!r}).write_text('escaped')"
                ),
            ],
            start_new_session=True,
        )
        time.sleep(5)

    with pytest.raises(__import__("concurrent.futures").futures.TimeoutError):
        trial_runner._call_runner_in_process(runner, {}, 0.2)
    time.sleep(0.8)

    assert not marker.exists()


def test_process_timeout_backend_drains_large_success_before_join() -> None:
    payload = trial_runner._call_runner_in_process(
        lambda _payload: b"x" * (8 * 1024 * 1024),
        {},
        2,
    )
    assert len(payload) == 8 * 1024 * 1024


def test_process_group_helpers_contain_startup_and_reap_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_calls: list[bool] = []
    monkeypatch.setattr(trial_runner.os, "setsid", lambda: session_calls.append(True))
    output = __import__("queue").Queue()
    trial_runner._runner_process_entry(
        output, lambda payload: payload, {"ok": True}, True
    )
    assert session_calls == [True]
    assert output.get_nowait() == ("ok", {"ok": True})

    class Process:
        pid = 123

        def __init__(self) -> None:
            self.joins: list[object] = []

        def join(self, timeout=None) -> None:
            self.joins.append(timeout)

        def is_alive(self) -> bool:
            return False

    process = Process()
    monkeypatch.setattr(
        trial_runner.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(OSError("already gone")),
    )
    assert process_containment.isolated_process_group(process) is None

    monkeypatch.setattr(trial_runner.os, "getpgid", lambda pid: pid)
    signals: list[int] = []

    def missing_group(_pid: int, sig: int) -> None:
        signals.append(sig)
        raise ProcessLookupError

    monkeypatch.setattr(trial_runner.os, "killpg", missing_group)
    trial_runner._terminate_runner_process(process)

    assert signals == [trial_runner.signal.SIGTERM, trial_runner.signal.SIGKILL]
    assert process.joins == [2, None]


def test_process_group_fallback_tolerates_already_reaped_process() -> None:
    class GoneProcess:
        pid = None

        def __init__(self) -> None:
            self.joins: list[object] = []

        def terminate(self) -> None:
            raise ProcessLookupError("already gone")

        def join(self, timeout=None) -> None:
            self.joins.append(timeout)

        def is_alive(self) -> bool:
            return False

    process = GoneProcess()
    trial_runner._terminate_runner_process(process)
    assert process.joins == [2, None]


def test_iter_files_selects_files_relative_to_arbitrary_root(tmp_path: Path) -> None:
    root = tmp_path / "detached-source-tree"
    root.mkdir()
    sentinel = root / "phase0-root-relative-only.sentinel"
    sentinel.write_text("include me\n")

    assert not (Path.cwd() / sentinel.name).exists()
    assert iter_files(root) == [sentinel]


def test_iter_files_rejects_symlinked_files_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "detached-source-tree"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("do not archive\n")
    linked = root / "linked-secret.txt"
    linked.symlink_to(outside)

    assert linked.is_file()
    assert iter_files(root) == []


def test_wheel_smoke_uses_isolated_pep517_builder() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "wheel_smoke.sh").read_text(encoding="utf-8")

    assert '"$PYTHON" -m build --wheel' in script
    assert "pip wheel" not in script
    assert "--no-build-isolation" not in script
    assert "--no-isolation" not in script


def test_smoke_import_parse_does_not_write_default_results_directory() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "smoke_import_parse.sh").read_text(encoding="utf-8")

    assert "TemporaryDirectory" in script
    assert "output_dir=" in script


def test_distribution_manifest_ignores_git_and_egg_info_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "worktree"
    (root / "optimizer").mkdir(parents=True)
    (root / "optimizer" / "version.py").write_text('__version__ = "4.0.0"\n')
    (root / "optimizer" / "__init__.py").write_text("")
    clean_report = manifest(root)

    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "optimizer"], check=True)
    egg_info = root / "optimizer.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated metadata\n")

    worktree_report = manifest(root)

    assert worktree_report.hygiene_ok is True
    assert worktree_report.forbidden_files == []
    assert worktree_report.file_count == clean_report.file_count
    assert worktree_report.sha256 == clean_report.sha256


def test_distribution_manifest_reports_excluded_artifacts(tmp_path: Path) -> None:
    (tmp_path / "optimizer").mkdir()
    (tmp_path / "optimizer" / "version.py").write_text('__version__ = "4.0.0"\n')
    (tmp_path / "optimizer" / "__init__.py").write_text("")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "state").write_text("cache")
    (tmp_path / "optimizer_results").mkdir()
    (tmp_path / "optimizer_results" / "trial.json").write_text("{}")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "artifact.whl").write_text("wheel")
    (tmp_path / "old.zip").write_text("zip")

    report = manifest(tmp_path)

    assert report.hygiene_ok is False
    assert ".pytest_cache" not in report.forbidden_files
    assert ".pytest_cache/state" not in report.forbidden_files
    assert "optimizer_results" in report.forbidden_files
    assert "optimizer_results/trial.json" in report.forbidden_files
    assert "dist" in report.forbidden_files
    assert "dist/artifact.whl" in report.forbidden_files
    assert "old.zip" in report.forbidden_files


def test_distribution_zip_skips_release_artifacts(tmp_path: Path) -> None:
    (tmp_path / "optimizer").mkdir()
    (tmp_path / "optimizer" / "version.py").write_text('__version__ = "4.0.0"\n')
    (tmp_path / "optimizer" / "__init__.py").write_text("x = 1\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "tmp.py").write_text("bad")
    output = tmp_path / "out.zip"

    built = build_zip(tmp_path, output)

    assert built == output
    assert manifest(tmp_path).hygiene_ok is False


def test_cli_load_params_validation_errors(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_params(bad_json)

    not_list = tmp_path / "not_list.json"
    not_list.write_text(json.dumps({"parameters": {"name": "x"}}))
    with pytest.raises(ValueError, match="parameter file must contain"):
        load_params(not_list)

    bad_row = tmp_path / "bad_row.json"
    bad_row.write_text(json.dumps(["not-object"]))
    with pytest.raises(ValueError, match="parameter row 0"):
        load_params(bad_row)


def test_cli_load_runner_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FILE:OBJECT"):
        load_obj("runner.py")
    with pytest.raises(ValueError, match="FILE:OBJECT"):
        load_obj(str(tmp_path / "runner.py") + ":")

    runner_file = tmp_path / "runner.py"
    runner_file.write_text("value = 1\n")
    with pytest.raises(ValueError, match="cannot load runner module"):
        load_obj("runner_without_suffix:runner")
    with pytest.raises(ValueError, match="not callable"):
        load_obj(f"{runner_file}:value")


def test_cli_errors_are_clean_user_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    params = tmp_path / "params.json"
    params.write_text("{")

    assert (
        main(["dry-run", "--params", str(params), "--output-dir", str(tmp_path)]) == 2
    )
    assert "invalid JSON" in capsys.readouterr().err

    valid = tmp_path / "params_valid.json"
    valid.write_text(
        json.dumps(
            [
                {
                    "name": "x",
                    "param_type": "int",
                    "default": 1,
                    "min_val": 1,
                    "max_val": 1,
                    "step": 1,
                }
            ]
        )
    )
    assert main(["run", "--params", str(valid), "--runner", "missing-format"]) == 2
    assert "FILE:OBJECT" in capsys.readouterr().err


def test_cli_main_module_execution_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["optimizer.cli.main", "--help"])
    with pytest.raises(SystemExit) as exc, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("optimizer.cli.main", run_name="__main__")
    assert exc.value.code == 0


def test_cli_package_main_delegates() -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["--help"])
    assert exc.value.code == 0


def test_parameter_space_strict_parameter_validation() -> None:
    with pytest.raises(ParameterValidationError, match="unsupported parameter type"):
        ParameterSpace([Parameter("x", "integer", 1, 1, 2, 1)])  # type: ignore[arg-type]
    with pytest.raises(ParameterValidationError, match="must be numeric"):
        ParameterSpace([Parameter("x", "float", "bad", 1, 2, 1)])
    with pytest.raises(ParameterValidationError, match="must be numeric"):
        ParameterSpace([Parameter("x", "float", True, 1, 2, 1)])
    with pytest.raises(ParameterValidationError, match="max_val must be >= min_val"):
        ParameterSpace([Parameter("x", "float", 1.0, 2.0, 1.0, 1.0)])
    with pytest.raises(ParameterValidationError, match="integral"):
        ParameterSpace([Parameter("x", "int", 1.5, 1, 2, 1)])
    with pytest.raises(ParameterValidationError, match="bool default"):
        ParameterSpace([Parameter("flag", "bool", 1)])

    diagnostics = ParameterSpace([Parameter("x", "int", 1, 1, 2, 1)]).validate_params(
        {"x": True}
    )
    assert diagnostics[0].code == "PARAM_TYPE"
    float_diagnostics = ParameterSpace(
        [Parameter("x", "float", 1.0, 1.0, 2.0, 1.0)]
    ).validate_params({"x": False})
    assert float_diagnostics[0].code == "PARAM_TYPE"
