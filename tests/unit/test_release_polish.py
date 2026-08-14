from __future__ import annotations

import builtins
import json
import multiprocessing as mp
import os
import runpy
import signal
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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


def _raise_during_result_unpickle() -> None:
    raise RuntimeError("unpickle exploded")


class _ExplodesDuringResultUnpickle:
    def __reduce__(self):  # type: ignore[no-untyped-def]
        return _raise_during_result_unpickle, ()


def _return_unpickle_exploder(_payload):  # type: ignore[no-untyped-def]
    return _ExplodesDuringResultUnpickle()


def test_release_version_is_4_0_1() -> None:
    assert __version__ == "4.0.2"


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


def test_process_timeout_kills_detached_descendant_spawned_by_nonleader_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)
    marker = tmp_path / "threaded-detached-descendant-survived"

    def runner(_payload):
        spawned = threading.Event()

        def spawn_detached() -> None:
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib,time; time.sleep(1.5); "
                        f"pathlib.Path({str(marker)!r}).write_text('escaped')"
                    ),
                ],
                start_new_session=True,
            )
            spawned.set()
            time.sleep(5)

        threading.Thread(target=spawn_detached).start()
        spawned.wait()
        time.sleep(5)

    with pytest.raises(__import__("concurrent.futures").futures.TimeoutError):
        trial_runner._call_runner_in_process(runner, {}, 1)
    time.sleep(0.8)

    assert not marker.exists()


def test_process_timeout_prevents_late_detached_spawn_from_signaled_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)
    marker = tmp_path / "late-detached-grandchild-survived"
    ready = tmp_path / "late-spawn-handler-ready"

    def runner(_payload):
        grandchild = (
            "import pathlib,time; time.sleep(0.5); "
            f"pathlib.Path({str(marker)!r}).write_text('escaped')"
        )
        helper = (
            "import pathlib,signal,subprocess,sys,time\n"
            "def handler(*_args):\n"
            f" subprocess.Popen([sys.executable,'-c',{grandchild!r}],start_new_session=True)\n"
            " raise SystemExit\n"
            "signal.signal(signal.SIGTERM,handler)\n"
            f"pathlib.Path({str(ready)!r}).write_text('ready')\n"
            "time.sleep(10)\n"
        )
        subprocess.Popen(
            [sys.executable, "-c", helper],
            start_new_session=True,
        )
        deadline = time.monotonic() + 0.2
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not ready.exists():
            raise RuntimeError("late-spawn helper did not become ready")
        time.sleep(5)

    with pytest.raises(__import__("concurrent.futures").futures.TimeoutError):
        trial_runner._call_runner_in_process(runner, {}, 0.4)
    time.sleep(0.8)

    assert not marker.exists()


def test_process_success_kills_detached_descendant_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)
    marker = tmp_path / "successful-runner-descendant-survived"

    def runner(_payload):
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib,time; time.sleep(0.5); "
                    f"pathlib.Path({str(marker)!r}).write_text('escaped')"
                ),
            ],
            start_new_session=True,
        )
        return 42

    assert trial_runner._call_runner_in_process(runner, {}, 2) == 42
    time.sleep(0.7)

    assert not marker.exists()


def test_process_result_unpickle_error_reaps_waiting_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)
    before = {process.pid for process in mp.active_children()}
    leaked = mp.active_children()[:0]

    try:
        with pytest.raises(RuntimeError, match="unpickle exploded"):
            trial_runner._call_runner_in_process(_return_unpickle_exploder, {}, 2)
    finally:
        leaked = [
            process
            for process in mp.active_children()
            if process.pid not in before and process.is_alive()
        ]
        for process in leaked:
            process.kill()
            process.join(2)

    assert leaked == []


def test_abrupt_runner_exit_contains_detached_descendant_without_cgroup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "abrupt-escape.marker"
    child_pid_path = tmp_path / "abrupt-child.pid"
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)

    def runner(_payload):  # type: ignore[no-untyped-def]
        code = (
            "import os,pathlib,time; "
            f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
            "time.sleep(0.4); "
            f"pathlib.Path({str(marker)!r}).write_text('escaped')"
        )
        subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
        os._exit(7)

    child_pid: int | None = None
    try:
        with pytest.raises(RuntimeError, match="exited with code 7"):
            trial_runner._call_runner_in_process(runner, {}, 1)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not child_pid_path.exists():
            time.sleep(0.005)
        if child_pid_path.exists():
            child_pid = int(child_pid_path.read_text())
        time.sleep(0.6)
        assert not marker.exists()
        if child_pid is not None:
            assert not Path(f"/proc/{child_pid}").exists()
    finally:
        if child_pid is not None and Path(f"/proc/{child_pid}").exists():
            os.kill(child_pid, signal.SIGKILL)


def test_process_success_contains_descendant_reparented_during_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trigger = tmp_path / "trigger"
    helper_pid_path = tmp_path / "helper.pid"
    escaped_pid_path = tmp_path / "escaped.pid"
    marker = tmp_path / "escaped.marker"
    original_children = process_containment._process_children
    fired = False

    def process_state(pid: int) -> str | None:
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
        except (IndexError, OSError):
            return None

    def wrapped_children(pid: int) -> list[int]:
        nonlocal fired
        children = original_children(pid)
        if not fired:
            fired = True
            trigger.write_text("spawn-and-exit")
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if helper_pid_path.exists() and escaped_pid_path.exists():
                    helper_pid = int(helper_pid_path.read_text())
                    if process_state(helper_pid) == "Z":
                        break
                time.sleep(0.002)
        return children

    def runner(_payload):  # type: ignore[no-untyped-def]
        escaped_code = (
            "import os,pathlib,time; "
            f"pathlib.Path({str(escaped_pid_path)!r}).write_text(str(os.getpid())); "
            "time.sleep(0.45); "
            f"pathlib.Path({str(marker)!r}).write_text('escaped')"
        )
        helper_code = (
            "import os,pathlib,subprocess,sys,time; "
            f"pathlib.Path({str(helper_pid_path)!r}).write_text(str(os.getpid())); "
            f"trigger=pathlib.Path({str(trigger)!r}); "
            "deadline=time.monotonic()+3; "
            'exec("while not trigger.exists() and time.monotonic() < deadline:\\n time.sleep(0.001)"); '
            f"subprocess.Popen([sys.executable,'-c',{escaped_code!r}],start_new_session=True); "
            "os._exit(0)"
        )
        subprocess.Popen([sys.executable, "-c", helper_code])
        return 42

    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)
    monkeypatch.setattr(process_containment, "_process_children", wrapped_children)
    escaped_pid = None
    try:
        assert trial_runner._call_runner_in_process(runner, {}, 2) == 42
        if escaped_pid_path.exists():
            escaped_pid = int(escaped_pid_path.read_text())
        time.sleep(0.7)
        assert not marker.exists()
    finally:
        if escaped_pid is not None and process_state(escaped_pid) is not None:
            os.kill(escaped_pid, signal.SIGKILL)


def test_process_success_freezes_runner_before_descendant_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)
    trigger = tmp_path / "snapshot-complete"
    child_pid_file = tmp_path / "child.pid"
    marker = tmp_path / "post-snapshot-descendant-survived"
    original_process_children = process_containment._process_children
    triggered = False

    def wrapped_process_children(pid: int) -> list[int]:
        nonlocal triggered
        children = original_process_children(pid)
        if not triggered:
            triggered = True
            trigger.write_text("go")
            deadline = time.monotonic() + 0.3
            while not child_pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.002)
            time.sleep(0.03)
        return children

    def runner(_payload):
        def spawn_after_snapshot() -> None:
            deadline = time.monotonic() + 2
            while not trigger.exists() and time.monotonic() < deadline:
                time.sleep(0.001)
            if not trigger.exists():
                return
            child_code = (
                "import os,pathlib,time; "
                f"pathlib.Path({str(child_pid_file)!r}).write_text(str(os.getpid())); "
                "time.sleep(0.4); "
                f"pathlib.Path({str(marker)!r}).write_text('escaped'); "
                "time.sleep(10)"
            )
            subprocess.Popen([sys.executable, "-c", child_code], start_new_session=True)

        threading.Thread(target=spawn_after_snapshot, daemon=True).start()
        return 42

    monkeypatch.setattr(
        process_containment, "_process_children", wrapped_process_children
    )
    try:
        assert trial_runner._call_runner_in_process(runner, {}, 2) == 42
        time.sleep(0.7)
        assert triggered is True
        assert not marker.exists()
    finally:
        if child_pid_file.exists():
            child_pid = int(child_pid_file.read_text())
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


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
    assert process_containment.isolated_process_group(SimpleNamespace(pid=None)) is None
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
    monkeypatch.setattr(process_containment.sys, "platform", "darwin")
    trial_runner._terminate_runner_process(process)

    assert signals == []
    assert process.joins == [2, 0]


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
    assert process.joins == [2, 0]


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
