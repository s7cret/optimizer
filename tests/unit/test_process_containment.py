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

    containment.remove_trial_cgroup(tmp_path / "already-removed")

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
    with pytest.raises(RuntimeError, match="cgroup remained populated"):
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

    def send_signal(pidfd: int, sig: int) -> None:
        attempts.append((pidfd, sig))
        if pidfd == 112:
            raise ProcessLookupError

    monkeypatch.setattr(containment, "PIDFD_SEND_SIGNAL", send_signal)
    containment.signal_processes([(11, 111), (12, 112)], containment.signal.SIGTERM)
    assert attempts == [
        (112, containment.signal.SIGTERM),
        (111, containment.signal.SIGTERM),
    ]


def test_procfs_descendant_snapshot_scans_children_of_every_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proc_root = tmp_path / "proc"
    monkeypatch.setattr(containment, "PROC_ROOT", proc_root)

    def children(process: int, task: int, value: str) -> None:
        path = proc_root / str(process) / "task" / str(task)
        path.mkdir(parents=True)
        (path / "children").write_text(value)

    children(10, 10, "11")
    children(10, 101, "12")
    children(11, 11, "")
    children(12, 12, "13")
    children(13, 13, "")

    assert containment.descendant_processes(10) == [11, 12, 13]


def test_descendant_handle_snapshot_rejects_reused_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proc_root = tmp_path / "proc"
    monkeypatch.setattr(containment, "PROC_ROOT", proc_root)

    def process(pid: int, parent: int, started: int, children: str = "") -> None:
        task = proc_root / str(pid) / "task" / str(pid)
        task.mkdir(parents=True, exist_ok=True)
        (task / "children").write_text(children)
        fields = ["S", str(parent), *("0" for _ in range(17)), str(started)]
        (proc_root / str(pid) / "stat").write_text(
            f"{pid} (process {pid}) {' '.join(fields)}"
        )

    process(10, 1, 100, "11")
    process(11, 10, 101)
    closed: list[int] = []

    def reuse_pid(_pid: int) -> int:
        process(11, 99, 102)
        return 111

    monkeypatch.setattr(containment, "_open_pidfd", reuse_pid, raising=False)
    monkeypatch.setattr(containment.os, "close", closed.append)

    assert containment.descendant_process_handles(10) == []
    assert closed == [111]


def test_pidfd_fallback_uses_portable_libc_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    class Function:
        restype: object | None = None
        argtypes: object | None = None
        result = 55

        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, *args: int) -> int:
            calls.append((self.name, args))
            return self.result

    pidfd_open = Function("pidfd_open")
    pidfd_send_signal = Function("pidfd_send_signal")
    libc = SimpleNamespace(
        pidfd_open=pidfd_open,
        pidfd_send_signal=pidfd_send_signal,
    )
    monkeypatch.setattr(containment.ctypes, "CDLL", lambda *_args, **_kwargs: libc)

    assert containment._libc_pidfd_open(7) == 55
    containment._libc_pidfd_send_signal(55, 9)
    assert calls == [
        ("pidfd_open", (7, 0)),
        ("pidfd_send_signal", (55, 9, 0, 0)),
    ]

    pidfd_send_signal.result = -1
    monkeypatch.setattr(containment.ctypes, "get_errno", lambda: 3)
    with pytest.raises(OSError, match="No such process"):
        containment._libc_pidfd_send_signal(55, 9)

    pidfd_send_signal.result = 55
    pidfd_open.result = -1
    with pytest.raises(OSError, match="No such process"):
        containment._libc_pidfd_open(7)

    monkeypatch.setattr(
        containment.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    with pytest.raises(OSError, match="Function not implemented"):
        containment._libc_pidfd_open(7)
    with pytest.raises(OSError, match="Function not implemented"):
        containment._libc_pidfd_send_signal(55, 9)


def test_child_subreaper_setup_success_and_failure_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, ...]] = []

    class Prctl:
        def __init__(self, result: int) -> None:
            self.result = result

        def __call__(self, *args: int) -> int:
            calls.append(args)
            return self.result

    monkeypatch.setattr(containment.sys, "platform", "darwin")
    assert not containment.enable_child_subreaper()

    monkeypatch.setattr(containment.sys, "platform", "linux")
    monkeypatch.setattr(
        containment.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no libc")),
    )
    assert not containment.enable_child_subreaper()

    failed = Prctl(-1)
    monkeypatch.setattr(
        containment.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(prctl=failed),
    )
    assert not containment.enable_child_subreaper()

    succeeded = Prctl(0)
    monkeypatch.setattr(
        containment.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(prctl=succeeded),
    )
    assert containment.enable_child_subreaper()
    assert calls[-1] == (containment.PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)


def test_frozen_descendant_snapshot_stops_each_parent_before_scanning_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int]] = []
    children = {10: [11, 11], 11: [12], 12: []}
    parents = {11: 10, 12: 11}

    def process_children(pid: int) -> list[int]:
        events.append(("children", pid))
        return children[pid]

    def freeze_handle(
        _pid: int,
        pidfd: int,
        _identity: tuple[int, int],
        *,
        timeout: float,
        deadline: float | None = None,
    ) -> bool:
        assert timeout > 0
        assert deadline is not None
        events.append(("stop", pidfd))
        return True

    monkeypatch.setattr(containment, "_process_children", process_children)
    monkeypatch.setattr(
        containment,
        "_process_identity",
        lambda pid: (parents[pid], pid * 100),
    )
    monkeypatch.setattr(containment, "_open_pidfd", lambda pid: pid + 100)
    monkeypatch.setattr(containment, "_freeze_process_handle", freeze_handle)

    assert containment.descendant_process_handles(10, freeze=True) == [
        (11, 111),
        (12, 112),
    ]
    assert events.index(("stop", 111)) < events.index(("children", 11))
    assert events.index(("stop", 112)) < events.index(("children", 12))


def test_frozen_descendant_snapshot_retries_transient_identity_and_pidfd_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = iter((None, (10, 1100), (10, 1100), (10, 1100)))
    pidfds = iter((None, 111))
    monkeypatch.setattr(
        containment, "_process_children", lambda pid: [11] if pid == 10 else []
    )
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: next(identities))
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: next(pidfds))
    monkeypatch.setattr(containment, "_is_descendant", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        containment, "_freeze_process_handle", lambda *_args, **_kwargs: True
    )

    assert containment.descendant_process_handles(10, freeze=True) == [(11, 111)]


def test_frozen_descendant_snapshot_fails_closed_on_persistent_scan_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(containment, "_process_children", lambda _pid: None)
    with pytest.raises(RuntimeError, match="stable descendant snapshot"):
        containment.descendant_process_handles(10, freeze=True, retry_timeout=0)

    monkeypatch.setattr(containment, "_process_children", lambda _pid: [11])
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: (10, 1100))
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: 111)
    monkeypatch.setattr(containment, "_is_descendant", lambda *_args, **_kwargs: False)
    closed: list[int] = []
    monkeypatch.setattr(containment.os, "close", closed.append)
    with pytest.raises(RuntimeError, match="stable descendant snapshot"):
        containment.descendant_process_handles(10, freeze=True, retry_timeout=0.01)
    assert closed and set(closed) == {111}

    monkeypatch.setattr(containment, "_is_descendant", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        containment, "_freeze_process_handle", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        containment, "_signal_process_handle", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        containment, "_wait_process_handles", lambda *_args, **_kwargs: False
    )
    owned: list[tuple[int, int]] = []
    with pytest.raises(RuntimeError, match="did not exit after failed freeze"):
        containment.descendant_process_handles(10, freeze=True, handles=owned)
    assert owned == [(11, 111)]

    ticks = iter((0.0, 0.1, 0.2, 0.3, 0.4, 0.5))
    monkeypatch.setattr(containment.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        containment, "_process_children", lambda pid: [pid + 1] if pid < 13 else []
    )
    monkeypatch.setattr(containment, "_process_identity", lambda pid: (pid - 1, pid))
    monkeypatch.setattr(containment, "_open_pidfd", lambda pid: pid + 100)
    monkeypatch.setattr(
        containment, "_freeze_process_handle", lambda *_args, **_kwargs: True
    )
    with pytest.raises(RuntimeError, match="global containment deadline"):
        containment.descendant_process_handles(10, freeze=True, retry_timeout=0.15)


def test_frozen_descendant_freeze_uses_remaining_absolute_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    identities = 0
    observed: list[tuple[float, float | None]] = []

    def process_identity(_pid: int) -> tuple[int, int]:
        nonlocal identities
        identities += 1
        if identities == 2:
            clock["now"] = 0.99
        return (10, 1100)

    def freeze_handle(
        _pid: int,
        _pidfd: int,
        _identity: tuple[int, int],
        *,
        timeout: float,
        deadline: float | None = None,
    ) -> bool:
        observed.append((timeout, deadline))
        return True

    monkeypatch.setattr(containment.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        containment, "_process_children", lambda pid: [11] if pid == 10 else []
    )
    monkeypatch.setattr(containment, "_process_identity", process_identity)
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: 111)
    monkeypatch.setattr(containment, "_is_descendant", lambda *_args: True)
    monkeypatch.setattr(containment, "_freeze_process_handle", freeze_handle)

    assert containment.descendant_process_handles(10, freeze=True, deadline=1.0) == [
        (11, 111)
    ]
    assert observed == [(pytest.approx(0.01), 1.0)]


def test_frozen_descendant_registers_pidfd_before_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned: list[tuple[int, int]] = []
    monkeypatch.setattr(containment, "_process_children", lambda _pid: [11])
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: (10, 1100))
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: 111)
    monkeypatch.setattr(
        containment,
        "_is_descendant",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        containment.descendant_process_handles(10, freeze=True, handles=owned)
    assert owned == [(11, 111)]


def test_frozen_descendant_snapshot_waits_for_stop_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    states = iter(({11: "R"}, {11: "T"}))

    def process_children(pid: int) -> list[int]:
        events.append(("children", pid))
        return [11] if pid == 10 else []

    def process_task_states(_pid: int) -> dict[int, str]:
        task_states = next(states)
        events.append(("state", next(iter(task_states.values()))))
        return task_states

    monkeypatch.setattr(containment, "_process_children", process_children)
    monkeypatch.setattr(containment, "_process_identity", lambda pid: (10, pid * 100))
    monkeypatch.setattr(containment, "_process_task_states", process_task_states)
    monkeypatch.setattr(containment, "_open_pidfd", lambda pid: pid + 100)
    monkeypatch.setattr(
        containment,
        "_signal_process_handle",
        lambda pidfd, sig: events.append(("signal", (pidfd, sig))) or True,
    )

    assert containment.descendant_process_handles(10, freeze=True) == [(11, 111)]
    assert events.index(("state", "T")) < events.index(("children", 11))


def test_wait_process_stopped_rejection_and_timeout_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = (10, 1100)
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: None)
    assert not containment._wait_process_stopped(11, identity)

    monkeypatch.setattr(containment, "_process_identity", lambda _pid: identity)
    monkeypatch.setattr(containment, "_process_task_states", lambda _pid: {11: "Z"})
    assert not containment._wait_process_stopped(11, identity)

    monkeypatch.setattr(containment, "_process_task_states", lambda _pid: {11: "R"})
    assert not containment._wait_process_stopped(11, identity, timeout=0)

    identities = iter((identity, (10, 999)))
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: next(identities))
    monkeypatch.setattr(containment, "_process_task_states", lambda _pid: {11: "T"})
    assert not containment._wait_process_stopped(11, identity)


def test_wait_process_stopped_requires_every_current_task_to_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = (10, 1100)
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: identity)
    monkeypatch.setattr(containment, "_process_state", lambda _pid: "T")
    monkeypatch.setattr(
        containment,
        "_process_task_states",
        lambda _pid: {11: "T", 12: "R"},
        raising=False,
    )

    assert not containment._wait_process_stopped(11, identity, timeout=0.01)


def test_descendant_handle_snapshot_covers_frozen_rejection_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children = {10: [11, 12, 13, 14, 15]}
    identity_calls: dict[int, int] = {}

    def identity(pid: int) -> tuple[int, int] | None:
        identity_calls[pid] = identity_calls.get(pid, 0) + 1
        if pid == 11 and identity_calls[pid] == 1:
            return None
        return (10, pid * 100)

    monkeypatch.setattr(
        containment, "_process_children", lambda pid: children.get(pid, [])
    )
    monkeypatch.setattr(containment, "_process_identity", identity)
    monkeypatch.setattr(
        containment,
        "_open_pidfd",
        lambda pid: None if pid == 12 and identity_calls[pid] == 1 else pid + 100,
    )
    monkeypatch.setattr(containment, "_is_descendant", lambda _pid, _root: True)
    monkeypatch.setattr(
        containment,
        "_freeze_process_handle",
        lambda _pid, pidfd, _identity, **_kwargs: pidfd != 114,
    )
    closed: list[int] = []
    monkeypatch.setattr(containment.os, "close", closed.append)

    assert containment.descendant_process_handles(10, freeze=True) == [
        (13, 113),
        (14, 114),
        (15, 115),
        (11, 111),
        (12, 112),
    ]
    assert closed == []

    monkeypatch.setattr(containment, "descendant_processes", lambda _root: [16])
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: (10, 1600))
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: 116)
    assert containment.descendant_process_handles(10) == [(16, 116)]


def test_process_children_and_handle_error_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proc_root = tmp_path / "proc"
    task_root = proc_root / "10" / "task"
    (task_root / "10").mkdir(parents=True)
    (task_root / "11").mkdir()
    (task_root / "10" / "children").write_text("invalid")
    (task_root / "11" / "children").write_text("12 13")
    monkeypatch.setattr(containment, "PROC_ROOT", proc_root)

    assert containment._process_children(10) is None
    (task_root / "10" / "children").write_text("")
    assert containment._process_children(10) == [12, 13]
    assert containment._process_children(99) is None
    assert containment._process_state(99) is None
    assert containment._process_task_states(10) is None
    (proc_root / "20" / "task").mkdir(parents=True)
    assert containment._process_task_states(20) is None

    monkeypatch.setattr(containment, "PIDFD_SEND_SIGNAL", None)
    assert not containment._signal_process_handle(10, containment.signal.SIGSTOP)
    monkeypatch.setattr(
        containment,
        "PIDFD_SEND_SIGNAL",
        lambda _pidfd, _sig: (_ for _ in ()).throw(OSError("gone")),
    )
    assert not containment._signal_process_handle(10, containment.signal.SIGSTOP)

    monkeypatch.setattr(
        containment.os,
        "close",
        lambda _fd: (_ for _ in ()).throw(OSError("already closed")),
    )
    containment.close_process_handles([(1, 10)])

    registrations: list[tuple[int, int]] = []
    poll_results = iter(([], [(10, containment.select.POLLIN)]))

    class Poll:
        def register(self, pidfd: int, events: int) -> None:
            registrations.append((pidfd, events))

        def poll(self, _timeout=None):  # type: ignore[no-untyped-def]
            return next(poll_results)

    monkeypatch.setattr(containment.select, "poll", Poll)
    containment._wait_process_handles([(1, 10)])
    assert registrations == [
        (
            10,
            containment.select.POLLIN
            | containment.select.POLLHUP
            | containment.select.POLLERR,
        )
    ]

    monkeypatch.setattr(
        containment.select,
        "poll",
        lambda: SimpleNamespace(register=lambda *_args: None, poll=lambda _timeout: []),
    )
    assert not containment._wait_process_handles([(1, 10)], timeout=0.001)

    monkeypatch.setattr(
        containment,
        "PIDFD_SEND_SIGNAL",
        lambda *_args: (_ for _ in ()).throw(OSError("signal failed")),
    )
    assert not containment.signal_processes([(1, 10)], containment.signal.SIGKILL)


def test_supervise_runner_worker_child_flushes_records_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitCalled(BaseException):
        def __init__(self, code: int) -> None:
            self.code = code

    class Output:
        def __init__(self) -> None:
            self.records: list[object] = []
            self.closed = False
            self.joined = False

        def put(self, record: object) -> None:
            self.records.append(record)

        def close(self) -> None:
            self.closed = True

        def join_thread(self) -> None:
            self.joined = True

    output = Output()
    closed: list[int] = []
    writes: list[tuple[int, bytes]] = []
    monkeypatch.setattr(containment.os, "pipe", lambda: (3, 4))
    monkeypatch.setattr(containment.os, "fork", lambda: 0)
    monkeypatch.setattr(containment.os, "close", closed.append)
    monkeypatch.setattr(
        containment.os, "write", lambda fd, value: writes.append((fd, value)) or 1
    )
    monkeypatch.setattr(
        containment.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(ExitCalled(code)),
    )

    ran: list[bool] = []
    with pytest.raises(ExitCalled) as exited:
        containment.supervise_runner_worker(output, lambda: ran.append(True))
    assert exited.value.code == 0
    assert ran == [True]
    assert output.closed and output.joined
    assert writes == [(4, b"1")]
    assert closed == [3, 4]

    class MinimalOutput:
        records: list[object] = []

        def put(self, record: object) -> None:
            self.records.append(record)

    minimal = MinimalOutput()
    monkeypatch.setattr(
        containment.os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("ack failed")),
    )
    monkeypatch.setattr(
        containment.os,
        "close",
        lambda fd: (_ for _ in ()).throw(OSError("close failed")) if fd == 4 else None,
    )
    with pytest.raises(ExitCalled) as failed:
        containment.supervise_runner_worker(
            minimal, lambda: (_ for _ in ()).throw(ValueError("worker failed"))
        )
    assert failed.value.code == 1
    assert isinstance(minimal.records[0], tuple)
    assert minimal.records[0][0:3] == ("err", "ValueError", "worker failed")


def test_supervise_runner_worker_parent_observes_ack_or_reports_abrupt_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Output:
        def __init__(self) -> None:
            self.records: list[object] = []

        def put(self, record: object) -> None:
            self.records.append(record)

    class Poll:
        def __init__(self, events: list[tuple[int, int]]) -> None:
            self.events = events

        def register(self, _fd: int, _events: int) -> None:
            return None

        def poll(self, _timeout: int) -> list[tuple[int, int]]:
            return self.events

    output = Output()
    closed: list[int] = []
    waits = iter((InterruptedError(), (123, 0)))

    def waitpid(_pid: int, _options: int):  # type: ignore[no-untyped-def]
        result = next(waits)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(containment.os, "pipe", lambda: (3, 4))
    monkeypatch.setattr(containment.os, "fork", lambda: 123)
    monkeypatch.setattr(containment.os, "close", closed.append)
    monkeypatch.setattr(containment.os, "waitpid", waitpid)
    monkeypatch.setattr(containment.select, "poll", lambda: Poll([(3, 1)]))
    monkeypatch.setattr(containment.os, "read", lambda _fd, _size: b"1")
    containment.supervise_runner_worker(output, lambda: None)
    assert output.records == []
    assert closed == [4, 3]

    closed.clear()
    monkeypatch.setattr(containment.os, "waitpid", lambda _pid, _options: (123, 7 << 8))
    monkeypatch.setattr(containment.select, "poll", lambda: Poll([]))
    containment.supervise_runner_worker(output, lambda: None)
    last = output.records[-1]
    assert isinstance(last, tuple)
    assert last[0:3] == (
        "err",
        "RuntimeError",
        "runner worker exited with code 7",
    )
    assert closed == [4, 3]


def test_terminate_runner_closes_root_handle_when_freeze_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 10

        def __init__(self) -> None:
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True

        def join(self, _timeout=None) -> None:  # type: ignore[no-untyped-def]
            return None

        def is_alive(self) -> bool:
            return False

    process = Process()
    closed: list[int] = []
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: (1, 100))
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: 110)
    monkeypatch.setattr(containment, "_signal_process_handle", lambda _fd, _sig: False)
    monkeypatch.setattr(containment, "isolated_process_group", lambda _proc: None)
    monkeypatch.setattr(containment.os, "close", closed.append)

    with pytest.raises(RuntimeError, match="could not freeze runner root"):
        containment.terminate_runner_process(process)

    assert not process.terminated
    assert closed == [110]


def test_terminate_runner_revalidates_root_identity_and_fails_closed_on_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 10

        def join(self, _timeout=None) -> None:  # type: ignore[no-untyped-def]
            return None

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(containment, "_process_identity", lambda _pid: None)
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: None)
    with pytest.raises(RuntimeError, match="pin a stable runner root identity"):
        containment.terminate_runner_process(Process())

    identities = iter(((1, 100), (1, 999)))
    closed: list[int] = []
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: next(identities))
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: 110)
    monkeypatch.setattr(
        containment,
        "_freeze_process_handle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not signal an unvalidated root")
        ),
    )
    monkeypatch.setattr(containment.os, "close", closed.append)

    with pytest.raises(RuntimeError, match="stable root identity"):
        containment.terminate_runner_process(Process())
    assert closed == [110]

    monkeypatch.setattr(containment, "_process_identity", lambda _pid: (1, 100))
    monkeypatch.setattr(
        containment, "_freeze_process_handle", lambda *_args, **_kwargs: False
    )
    group_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(containment, "isolated_process_group", lambda _proc: 10)
    monkeypatch.setattr(
        containment.os,
        "killpg",
        lambda group, sig: group_signals.append((group, sig)),
    )
    with pytest.raises(RuntimeError, match="could not freeze runner root"):
        containment.terminate_runner_process(Process())
    assert group_signals == []


def test_terminate_runner_process_falls_back_when_cgroup_kill_fails_and_closes_handles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Process:
        pid = 10

        def __init__(self) -> None:
            self.alive = True

        def join(self, _timeout=None) -> None:  # type: ignore[no-untyped-def]
            return None

        def is_alive(self) -> bool:
            return self.alive

        def kill(self) -> None:
            self.alive = False

    signals: list[tuple[int, int]] = []
    waited: list[list[tuple[int, int]]] = []
    closed: list[int] = []
    monkeypatch.setattr(containment, "kill_trial_cgroup", lambda _group: False)
    monkeypatch.setattr(containment, "remove_trial_cgroup", lambda _group: None)
    monkeypatch.setattr(containment, "isolated_process_group", lambda _proc: 10)
    monkeypatch.setattr(containment, "_process_identity", lambda pid: (1, pid * 100))
    monkeypatch.setattr(containment, "_open_pidfd", lambda pid: pid + 100)
    monkeypatch.setattr(
        containment,
        "_freeze_process_handle",
        lambda _pid, pidfd, _identity, **_kwargs: signals.append(
            (pidfd, containment.signal.SIGSTOP)
        )
        or True,
    )

    def snapshot(
        _root: int,
        *,
        freeze: bool = False,
        handles: list[tuple[int, int]] | None = None,
        deadline: float | None = None,
    ) -> list[tuple[int, int]]:
        assert freeze and handles is not None
        handles.extend(((11, 111), (12, 112)))
        return handles

    monkeypatch.setattr(containment, "descendant_process_handles", snapshot)
    monkeypatch.setattr(
        containment,
        "PIDFD_SEND_SIGNAL",
        lambda pidfd, sig: signals.append((pidfd, sig)),
    )
    monkeypatch.setattr(containment.os, "close", closed.append)
    monkeypatch.setattr(
        containment,
        "_wait_process_handles",
        lambda handles, **_kwargs: waited.append(list(handles)) or True,
        raising=False,
    )
    monkeypatch.setattr(
        containment.os,
        "killpg",
        lambda _group, _sig: (_ for _ in ()).throw(OSError("group disappeared")),
    )

    containment.terminate_runner_process(Process(), tmp_path / "group")

    assert (110, containment.signal.SIGSTOP) in signals
    assert (111, containment.signal.SIGKILL) in signals
    assert (112, containment.signal.SIGKILL) in signals
    assert (110, containment.signal.SIGKILL) in signals
    assert waited == [
        [(11, 111), (12, 112)],
        [(10, 110)],
        [(11, 111), (12, 112)],
    ]
    assert sorted(closed) == [110, 111, 112]


def test_terminate_runner_exception_still_reaps_stopped_root_and_owned_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions: list[object] = []

    class Process:
        pid = 10
        alive = True

        def join(self, timeout=None) -> None:  # type: ignore[no-untyped-def]
            actions.append(("join", timeout))

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            actions.append("terminate")

        def kill(self) -> None:
            actions.append("kill")
            self.alive = False

    def fail_snapshot(
        _root_pid: int,
        *,
        freeze: bool = False,
        handles: list[tuple[int, int]] | None = None,
        deadline: float | None = None,
    ) -> list[tuple[int, int]]:
        assert freeze
        assert handles is not None
        handles.append((11, 111))
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(containment, "_process_identity", lambda _pid: (1, 100))
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: 110)
    monkeypatch.setattr(
        containment, "_freeze_process_handle", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(containment, "descendant_process_handles", fail_snapshot)
    monkeypatch.setattr(
        containment,
        "_signal_process_handle",
        lambda pidfd, sig: actions.append(("signal", pidfd, sig)) or True,
    )
    monkeypatch.setattr(
        containment.os, "close", lambda pidfd: actions.append(("close", pidfd))
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        containment.terminate_runner_process(Process())

    assert ("signal", 110, containment.signal.SIGKILL) in actions
    assert "kill" in actions
    assert ("close", 111) in actions
    assert ("close", 110) in actions


@pytest.mark.parametrize(
    ("wait_results", "message"),
    [
        ([False, True], "descendants did not exit"),
        ([True, False, True], "runner did not exit"),
    ],
)
def test_terminate_runner_surfaces_bounded_pidfd_wait_failures(
    monkeypatch: pytest.MonkeyPatch,
    wait_results: list[bool],
    message: str,
) -> None:
    class Process:
        pid = 10

        def join(self, _timeout=None) -> None:  # type: ignore[no-untyped-def]
            return None

        def is_alive(self) -> bool:
            return False

    def snapshot(
        _root: int,
        *,
        freeze: bool = False,
        handles: list[tuple[int, int]] | None = None,
        deadline: float | None = None,
    ) -> list[tuple[int, int]]:
        assert freeze and handles is not None
        handles.append((11, 111))
        return handles

    waits = iter(wait_results)
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: (1, 100))
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: 110)
    monkeypatch.setattr(
        containment, "_freeze_process_handle", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(containment, "descendant_process_handles", snapshot)
    monkeypatch.setattr(
        containment, "_wait_process_handles", lambda *_args, **_kwargs: next(waits)
    )
    monkeypatch.setattr(
        containment, "_signal_process_handle", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(containment.os, "close", lambda _fd: None)

    with pytest.raises(RuntimeError, match=message):
        containment.terminate_runner_process(Process())


def test_terminate_runner_cleanup_errors_do_not_skip_reaping_or_handle_closure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class JoinFailure:
        pid = 10

        def join(self, _timeout=None) -> None:  # type: ignore[no-untyped-def]
            raise RuntimeError("join cleanup failed")

        def is_alive(self) -> bool:
            return False

    def fail_snapshot(
        _root: int,
        *,
        freeze: bool = False,
        handles: list[tuple[int, int]] | None = None,
        deadline: float | None = None,
    ) -> list[tuple[int, int]]:
        assert freeze and handles is not None
        handles.append((11, 111))
        raise RuntimeError("snapshot failed")

    closed: list[int] = []
    monkeypatch.setattr(containment, "_process_identity", lambda _pid: (1, 100))
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: 110)
    monkeypatch.setattr(
        containment, "_freeze_process_handle", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(containment, "descendant_process_handles", fail_snapshot)
    monkeypatch.setattr(
        containment, "_wait_process_handles", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        containment, "_signal_process_handle", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(containment.os, "close", closed.append)

    with pytest.raises(RuntimeError, match="snapshot failed"):
        containment.terminate_runner_process(JoinFailure())
    assert sorted(closed) == [110, 111]

    class Gone:
        pid = None

        def __init__(self) -> None:
            self.alive = True

        def terminate(self) -> None:
            raise ProcessLookupError("gone")

        def join(self, timeout=None) -> None:  # type: ignore[no-untyped-def]
            if timeout == 2:
                self.alive = False

        def is_alive(self) -> bool:
            return self.alive

    containment.terminate_runner_process(Gone())

    group = tmp_path / "group"
    monkeypatch.setattr(containment, "kill_trial_cgroup", lambda _path: True)
    monkeypatch.setattr(
        containment,
        "remove_trial_cgroup",
        lambda _path: (_ for _ in ()).throw(ValueError("remove failed")),
    )
    with pytest.raises(ValueError, match="remove failed"):
        containment.terminate_runner_process(Gone(), group)

    with pytest.raises(RuntimeError, match="join cleanup failed"):
        containment.terminate_runner_process(JoinFailure(), group)


def test_pidfd_unavailable_and_process_identity_error_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(containment, "PROC_ROOT", tmp_path / "missing-proc")
    assert containment._process_identity(7) is None

    def send(_pidfd: int, _sig: int) -> None:
        return None

    monkeypatch.setattr(containment, "PIDFD_OPEN", None)
    monkeypatch.setattr(containment, "PIDFD_SEND_SIGNAL", send)
    assert containment._open_pidfd(7) is None

    monkeypatch.setattr(containment, "PIDFD_OPEN", lambda _pid: 70)
    monkeypatch.setattr(containment, "PIDFD_SEND_SIGNAL", None)
    assert containment._open_pidfd(7) is None

    def fail_open(_pid: int) -> int:
        raise ProcessLookupError

    monkeypatch.setattr(containment, "PIDFD_OPEN", fail_open)
    monkeypatch.setattr(containment, "PIDFD_SEND_SIGNAL", send)
    assert containment._open_pidfd(7) is None

    monkeypatch.setattr(containment, "_process_identity", lambda _pid: None)
    assert containment._is_descendant(2, 1) is False

    parents = {2: (3, 20), 3: (2, 30)}
    monkeypatch.setattr(containment, "_process_identity", parents.get)
    assert containment._is_descendant(2, 1) is False

    monkeypatch.setattr(containment, "descendant_processes", lambda _root: [4, 5])
    identities = {4: None, 5: (1, 50)}
    monkeypatch.setattr(containment, "_process_identity", identities.get)
    monkeypatch.setattr(containment, "_open_pidfd", lambda _pid: None)
    assert containment.descendant_process_handles(1) == []

    monkeypatch.setattr(containment, "PIDFD_SEND_SIGNAL", None)
    containment.signal_processes([(5, 50)], containment.signal.SIGTERM)


def test_terminate_runner_process_cgroup_and_direct_error_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    actions: list[str] = []

    def kill_cgroup(_path: Path) -> bool:
        actions.append("cgkill")
        return True

    monkeypatch.setattr(containment, "kill_trial_cgroup", kill_cgroup)
    monkeypatch.setattr(
        containment, "remove_trial_cgroup", lambda _path: actions.append("remove")
    )

    class CgroupProcess:
        def __init__(self, *, alive: bool = True, kill_raises: bool = False) -> None:
            self.kill_raises = kill_raises
            self.alive = alive
            self.kill_calls = 0
            self.joins: list[object] = []

        def join(self, timeout=None) -> None:  # type: ignore[no-untyped-def]
            self.joins.append(timeout)

        def is_alive(self) -> bool:
            return self.alive

        def kill(self) -> None:
            self.kill_calls += 1
            self.alive = False
            if self.kill_raises:
                raise ProcessLookupError

    group = tmp_path / "group"
    process = CgroupProcess()
    containment.terminate_runner_process(process, group)
    assert process.joins == [2, 0, 2]
    assert actions == ["cgkill", "remove"]

    actions.clear()
    process = CgroupProcess(alive=False)
    containment.terminate_runner_process(process, group)
    assert process.kill_calls == 0
    assert process.joins == [2, 0]
    assert actions == ["cgkill", "remove"]

    actions.clear()
    process = CgroupProcess(kill_raises=True)
    containment.terminate_runner_process(process, group)
    assert process.joins == [2, 0, 2]
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
    with pytest.raises(RuntimeError, match="remained alive"):
        containment.terminate_runner_process(direct)
    assert direct.joins == [2, 0, 2]


def test_runner_process_entry_waits_for_start_and_finish_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output: queue.Queue[object] = queue.Queue()
    start_gate = _Event()
    finish_gate = _Event()
    trial_runner._runner_process_entry(
        output,
        lambda payload: payload,
        {"ok": True},
        False,
        start_gate,
        finish_gate,
    )
    assert start_gate.wait_called is True
    assert finish_gate.wait_called is True
    assert output.get_nowait() == ("ok", {"ok": True})

    monkeypatch.setattr(trial_runner, "_enable_child_subreaper", lambda: False)
    trial_runner._runner_process_entry(output, lambda payload: payload, {}, True)
    failed = output.get_nowait()
    assert isinstance(failed, tuple)
    assert failed[0:3] == (
        "err",
        "RuntimeError",
        "failed to enable child-subreaper containment",
    )

    monkeypatch.setattr(
        trial_runner, "sys", SimpleNamespace(platform="darwin"), raising=False
    )
    trial_runner._runner_process_entry(output, lambda payload: payload, {"ok": 2}, True)
    assert output.get_nowait() == ("ok", {"ok": 2})

    monkeypatch.setattr(
        trial_runner, "sys", SimpleNamespace(platform="linux"), raising=False
    )
    monkeypatch.setattr(trial_runner, "_enable_child_subreaper", lambda: True)
    monkeypatch.setattr(trial_runner.os, "setsid", lambda: None)
    supervised: list[bool] = []

    def supervise(_output, worker):  # type: ignore[no-untyped-def]
        supervised.append(True)
        worker()

    monkeypatch.setattr(trial_runner, "_supervise_runner_worker", supervise)
    trial_runner._runner_process_entry(
        output, lambda payload: payload, {"ok": 3}, True, None, None, True
    )
    assert supervised == [True]
    assert output.get_nowait() == ("ok", {"ok": 3})


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

    class PartialStartFailure(StartFailure):
        pid = 10

    partially_terminated: list[tuple[object, object]] = []
    partial = PartialStartFailure()
    context = _Context(partial, _Queue(("ok", 1)))
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: group)
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    monkeypatch.setattr(
        trial_runner,
        "_terminate_runner_process",
        lambda proc, cgroup: partially_terminated.append((proc, cgroup)),
    )
    with pytest.raises(OSError, match="start failed"):
        trial_runner._call_runner_in_process(lambda value: value, {}, 1)
    assert partially_terminated == [(partial, group)]
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)

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


def test_process_call_cgroup_attach_failure_and_success_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    group = tmp_path / "group"
    actions: list[str] = []

    class Completed:
        exitcode = 0

        def start(self) -> None:
            return None

        def join(self, _timeout=None) -> None:  # type: ignore[no-untyped-def]
            return None

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    monkeypatch.setattr(trial_runner.mp, "get_all_start_methods", lambda: ["fork"])
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: group)
    monkeypatch.setattr(
        trial_runner,
        "_remove_trial_cgroup",
        lambda _path: actions.append("remove"),
    )
    monkeypatch.setattr(
        trial_runner,
        "_kill_trial_cgroup",
        lambda _path: actions.append("kill"),
    )

    context = _Context(Completed(), _Queue(("ok", 41)))
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    monkeypatch.setattr(
        trial_runner, "_attach_trial_cgroup", lambda _path, _proc: False
    )
    assert trial_runner._call_runner_in_process(lambda value: value, {}, 1) == 41
    assert actions == ["remove"]

    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)
    context = _Context(Completed(), _Queue(("ok", 43)))
    monkeypatch.setattr(context, "Event", None)
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    assert trial_runner._call_runner_in_process(lambda value: value, {}, 1) == 43

    actions.clear()
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: group)
    context = _Context(Completed(), _Queue(("ok", 42)))
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    monkeypatch.setattr(trial_runner, "_attach_trial_cgroup", lambda _path, _proc: True)
    assert trial_runner._call_runner_in_process(lambda value: value, {}, 1) == 42
    assert actions == ["kill", "remove"]

    class TimedOut(Completed):
        exitcode = None

        def is_alive(self) -> bool:
            return True

    actions.clear()
    context = _Context(TimedOut(), _Queue())
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    monkeypatch.setattr(
        trial_runner,
        "_terminate_runner_process",
        lambda _proc, _group: actions.append("terminate"),
    )
    with pytest.raises(__import__("concurrent.futures").futures.TimeoutError):
        trial_runner._call_runner_in_process(lambda value: value, {}, 1)
    assert actions == ["terminate", "remove"]


def test_process_call_start_gate_failure_reaps_started_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Started:
        pid = 10

        def start(self) -> None:
            return None

    class SetFailure(_Event):
        def set(self) -> None:
            raise RuntimeError("start gate failed")

    output = _Queue(("ok", 1))
    process = Started()
    context = _Context(process, output)
    context.event = SetFailure()
    terminated: list[tuple[object, object]] = []
    monkeypatch.setattr(trial_runner.mp, "get_all_start_methods", lambda: ["fork"])
    monkeypatch.setattr(trial_runner.mp, "get_context", lambda _name: context)
    monkeypatch.setattr(trial_runner, "_create_trial_cgroup", lambda: None)
    monkeypatch.setattr(
        trial_runner,
        "_terminate_runner_process",
        lambda proc, cgroup: terminated.append((proc, cgroup)),
    )

    with pytest.raises(RuntimeError, match="start gate failed"):
        trial_runner._call_runner_in_process(lambda value: value, {}, 1)

    assert terminated == [(process, None)]
    assert output.closed and output.cancelled
