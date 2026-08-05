from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from optimizer.core import process_containment as containment
from optimizer.core import trial_runner


class _Queue:
    def __init__(self, record: Any = ...):
        self.record = record
        self.closed = False
        self.cancelled = False

    def get(self, timeout=None):  # type: ignore[no-untyped-def]
        if self.record is ...:
            raise queue.Empty
        return self.record

    def close(self) -> None:
        self.closed = True

    def cancel_join_thread(self) -> None:
        self.cancelled = True


class _Event:
    def __init__(self) -> None:
        self.set_called = False
        self.wait_called = False

    def set(self) -> None:
        self.set_called = True

    def wait(self) -> None:
        self.wait_called = True


class _Context:
    def __init__(self, process, output: _Queue) -> None:  # type: ignore[no-untyped-def]
        self.process = process
        self.output = output
        self.event = _Event()

    def Queue(self, maxsize=1):  # type: ignore[no-untyped-def]
        return self.output

    def Event(self) -> _Event:
        return self.event

    def Process(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.process


def test_current_cgroup_base_rejects_unsupported_missing_and_escape_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(containment.os, "name", "nt")
        assert containment.current_cgroup_base() is None

    root = tmp_path / "root"
    root.mkdir()
    proc = tmp_path / "self.cgroup"
    monkeypatch.setattr(containment, "CGROUP_ROOT", root)
    monkeypatch.setattr(containment, "PROC_SELF_CGROUP", proc)

    proc.write_text("1:name:/ignored\n")
    assert containment.current_cgroup_base() is None

    proc.write_text("0::/missing\n")
    assert containment.current_cgroup_base() is None

    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    proc.write_text("0::/escape\n")
    assert containment.current_cgroup_base() is None

    valid = root / "valid"
    valid.mkdir()
    (valid / "cgroup.procs").touch()
    proc.write_text("0::/valid\n")
    assert containment.current_cgroup_base() == valid


def test_cgroup_creation_io_and_cleanup_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(containment, "current_cgroup_base", lambda: None)
    assert containment.create_trial_cgroup() is None

    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setattr(containment, "current_cgroup_base", lambda: base)
    assert containment.create_trial_cgroup() is None

    base_file = tmp_path / "base-file"
    base_file.write_text("not a directory")
    monkeypatch.setattr(containment, "current_cgroup_base", lambda: base_file)
    assert containment.create_trial_cgroup() is None

    monkeypatch.setattr(containment, "current_cgroup_base", lambda: base)
    monkeypatch.setattr(containment.os, "getpid", lambda: 7)
    monkeypatch.setattr(containment.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    original_mkdir = Path.mkdir

    def kernel_mkdir(path: Path, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        original_mkdir(path, *args, **kwargs)
        (path / "cgroup.kill").touch()
        (path / "cgroup.procs").touch()

    monkeypatch.setattr(Path, "mkdir", kernel_mkdir)
    group = containment.create_trial_cgroup()
    assert group == base / "optimizer-trial-7-fixed"
    assert group is not None
    assert containment.attach_trial_cgroup(group, SimpleNamespace(pid=123)) is True
    assert (group / "cgroup.procs").read_text() == "123\n"
    assert containment.kill_trial_cgroup(group) is True
    assert (group / "cgroup.kill").read_text() == "1\n"
    assert containment.attach_trial_cgroup(tmp_path / "absent", object()) is False
    assert containment.kill_trial_cgroup(tmp_path / "absent") is False

    for child in group.iterdir():
        child.unlink()
    containment.remove_trial_cgroup(group)
    assert not group.exists()

    retry = tmp_path / "retry"
    original_mkdir(retry)
    blocker = retry / "blocker"
    blocker.touch()
    sleeps: list[float] = []

    def clear_after_retry(value: float) -> None:
        sleeps.append(value)
        blocker.unlink(missing_ok=True)

    monkeypatch.setattr(containment.time, "sleep", clear_after_retry)
    containment.remove_trial_cgroup(retry)
    assert sleeps == [0.01]
    assert not retry.exists()

    persistent = tmp_path / "persistent"
    original_mkdir(persistent)
    (persistent / "blocker").touch()
    monkeypatch.setattr(containment.time, "sleep", lambda _value: None)
    containment.remove_trial_cgroup(persistent)
    assert persistent.exists()


def test_procfs_descendant_snapshot_and_signaling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proc_root = tmp_path / "proc"
    monkeypatch.setattr(containment, "PROC_ROOT", proc_root)

    def children(pid: int, value: str) -> None:
        path = proc_root / str(pid) / "task" / str(pid)
        path.mkdir(parents=True)
        (path / "children").write_text(value)

    children(10, "11 12")
    children(11, "13")
    children(12, "10")
    children(13, "")
    children(20, "not-a-pid")

    assert containment.descendant_processes(10) == [11, 12, 13]
    assert containment.descendant_processes(20) == []
    assert containment.descendant_processes(99) == []

    attempts: list[tuple[int, int]] = []

    def signal(pid: int, sig: int) -> None:
        attempts.append((pid, sig))
        if pid == 12:
            raise ProcessLookupError

    monkeypatch.setattr(containment.os, "kill", signal)
    containment.signal_processes([11, 12], containment.signal.SIGTERM)
    assert attempts == [
        (12, containment.signal.SIGTERM),
        (11, containment.signal.SIGTERM),
    ]


def test_terminate_runner_process_cgroup_and_direct_error_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    actions: list[str] = []
    monkeypatch.setattr(
        containment, "kill_trial_cgroup", lambda _path: actions.append("cgkill")
    )
    monkeypatch.setattr(
        containment, "remove_trial_cgroup", lambda _path: actions.append("remove")
    )

    class CgroupProcess:
        def __init__(self, *, kill_raises: bool = False) -> None:
            self.kill_raises = kill_raises
            self.alive = True
            self.joins: list[object] = []

        def join(self, timeout=None) -> None:  # type: ignore[no-untyped-def]
            self.joins.append(timeout)

        def is_alive(self) -> bool:
            return self.alive

        def kill(self) -> None:
            self.alive = False
            if self.kill_raises:
                raise ProcessLookupError

    group = tmp_path / "group"
    process = CgroupProcess()
    containment.terminate_runner_process(process, group)
    assert process.joins == [2, None]
    assert actions == ["cgkill", "remove"]

    actions.clear()
    process = CgroupProcess(kill_raises=True)
    containment.terminate_runner_process(process, group)
    assert process.joins == [2, None]
    assert actions == ["cgkill", "remove"]

    class DirectProcess:
        pid = None

        def __init__(self) -> None:
            self.joins: list[object] = []

        def terminate(self) -> None:
            return None

        def join(self, timeout=None) -> None:  # type: ignore[no-untyped-def]
            self.joins.append(timeout)

        def is_alive(self) -> bool:
            return True

        def kill(self) -> None:
            raise OSError("already gone")

    direct = DirectProcess()
    containment.terminate_runner_process(direct)
    assert direct.joins == [2, None]


def test_process_entry_waits_for_start_gate() -> None:
    output: queue.Queue[object] = queue.Queue()
    gate = _Event()
    trial_runner._runner_process_entry(
        output,
        lambda payload: payload,
        {"ok": True},
        False,
        gate,
    )
    assert gate.wait_called is True
    assert output.get_nowait() == ("ok", {"ok": True})


def test_process_call_start_failure_empty_result_and_lingering_leader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trial_runner.mp, "get_all_start_methods", lambda: ["fork"])
    removed: list[object] = []
    group = tmp_path / "group"
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: group)
    monkeypatch.setattr(
        trial_runner, "_remove_trial_cgroup", lambda path: removed.append(path)
    )

    class StartFailure:
        def start(self) -> None:
            raise OSError("start failed")

    context = _Context(StartFailure(), _Queue(("ok", 1)))
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    with pytest.raises(OSError, match="start failed"):
        trial_runner._call_runner_in_process(lambda value: value, {}, 1)
    assert removed == [group]

    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)
    with pytest.raises(OSError, match="start failed"):
        trial_runner._call_runner_in_process(lambda value: value, {}, 1)

    class Exited:
        exitcode = 0

        def start(self) -> None:
            return None

        def join(self, _timeout=None) -> None:  # type: ignore[no-untyped-def]
            return None

        def is_alive(self) -> bool:
            return False

    context = _Context(Exited(), _Queue())
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    with pytest.raises(RuntimeError, match="without a result"):
        trial_runner._call_runner_in_process(lambda value: value, {}, 0.01)

    class ExitOne(Exited):
        exitcode = 1

    context = _Context(ExitOne(), _Queue())
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    with pytest.raises(RuntimeError, match="exited with code 1"):
        trial_runner._call_runner_in_process(lambda value: value, {}, 0.01)

    class Lingering(Exited):
        def is_alive(self) -> bool:
            return True

    terminated: list[object] = []
    context = _Context(Lingering(), _Queue(("ok", 42)))
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    monkeypatch.setattr(
        trial_runner,
        "_terminate_runner_process",
        lambda proc, cgroup: terminated.append((proc, cgroup)),
    )
    assert trial_runner._call_runner_in_process(lambda value: value, {}, 1) == 42
    assert len(terminated) == 1

    context = _Context(Exited(), _Queue(None))
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    with pytest.raises(RuntimeError, match="without a result record"):
        trial_runner._call_runner_in_process(lambda value: value, {}, 1)
