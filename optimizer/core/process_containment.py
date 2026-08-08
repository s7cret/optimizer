"""Best-effort process lifecycle containment for optimizer trials."""

from __future__ import annotations

import ctypes
import errno
import os
import select
import signal
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_SELF_CGROUP = Path("/proc/self/cgroup")
PROC_ROOT = Path("/proc")
PR_SET_CHILD_SUBREAPER = 36


def _libc_pidfd_open(pid: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.pidfd_open
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS)) from exc
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = int(function(pid, 0))
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return result


def _libc_pidfd_send_signal(pidfd: int, sig: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.pidfd_send_signal
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS)) from exc
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(pidfd, sig, 0, 0) < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


PIDFD_OPEN = cast(Callable[[int], int] | None, getattr(os, "pidfd_open", None))
PIDFD_SEND_SIGNAL = cast(
    Callable[[int, int], None] | None, getattr(signal, "pidfd_send_signal", None)
)
if sys.platform.startswith("linux"):  # pragma: no branch - procfs backend is Linux-only
    PIDFD_OPEN = PIDFD_OPEN or _libc_pidfd_open
    PIDFD_SEND_SIGNAL = PIDFD_SEND_SIGNAL or _libc_pidfd_send_signal


def enable_child_subreaper() -> bool:
    """Keep orphaned callable descendants parented below the runner process."""

    if not sys.platform.startswith("linux"):
        return False
    try:
        function = ctypes.CDLL(None, use_errno=True).prctl
    except (AttributeError, OSError):
        return False
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    function.restype = ctypes.c_int
    return function(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0


def isolated_process_group(proc: Any) -> int | None:
    pid = getattr(proc, "pid", None)
    if pid is None or not hasattr(os, "killpg"):
        return None
    try:
        return pid if os.getpgid(pid) == pid else None
    except OSError:
        return None


def current_cgroup_base() -> Path | None:
    if os.name != "posix":
        return None
    try:
        line = next(
            item
            for item in PROC_SELF_CGROUP.read_text().splitlines()
            if item.startswith("0::")
        )
        root = CGROUP_ROOT.resolve()
        base = (root / line[3:].lstrip("/")).resolve()
        if root != base and root not in base.parents:
            return None
        if not (base / "cgroup.procs").exists():
            return None
        return base
    except (OSError, StopIteration):
        return None


def create_trial_cgroup() -> Path | None:
    base = current_cgroup_base()
    if base is None:
        return None
    path = base / f"optimizer-trial-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        path.mkdir()
        if not (path / "cgroup.kill").exists():
            path.rmdir()
            return None
        return path
    except OSError:
        return None


def attach_trial_cgroup(path: Path, proc: Any) -> bool:
    try:
        (path / "cgroup.procs").write_text(f"{proc.pid}\n")
        return True
    except (AttributeError, OSError, TypeError):
        return False


def kill_trial_cgroup(path: Path) -> bool:
    try:
        (path / "cgroup.kill").write_text("1\n")
    except OSError:
        return False
    return True


def remove_trial_cgroup(path: Path) -> None:
    for _ in range(50):
        try:
            path.rmdir()
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.01)
    raise RuntimeError(f"cgroup remained populated after cleanup: {path}")


def _process_children(pid: int) -> list[int] | None:
    task_root = PROC_ROOT / str(pid) / "task"
    try:
        children_paths = sorted(task / "children" for task in task_root.iterdir())
    except OSError:
        return None
    children: list[int] = []
    for children_path in children_paths:
        try:
            children.extend(int(value) for value in children_path.read_text().split())
        except (OSError, ValueError):
            return None
    return children


def descendant_processes(root_pid: int) -> list[int]:
    """Snapshot descendants through procfs, including detached sessions."""

    pending = [root_pid]
    seen = {root_pid}
    descendants: list[int] = []
    while pending:
        parent = pending.pop()
        children = _process_children(parent)
        if children is None:
            continue
        for child in children:
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            pending.append(child)
    return descendants


def _process_identity(pid: int) -> tuple[int, int] | None:
    try:
        fields = (PROC_ROOT / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        return int(fields[1]), int(fields[19])
    except (IndexError, OSError, ValueError):
        return None


def _process_state(pid: int) -> str | None:
    try:
        return (PROC_ROOT / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()[0]
    except (IndexError, OSError):
        return None


def _process_task_states(pid: int) -> dict[int, str] | None:
    task_root = PROC_ROOT / str(pid) / "task"
    try:
        task_paths = sorted(task_root.iterdir(), key=lambda path: int(path.name))
        states = {
            int(task.name): task.joinpath("stat")
            .read_text()
            .rsplit(") ", 1)[1]
            .split()[0]
            for task in task_paths
        }
        stable_tids = {int(task.name) for task in task_root.iterdir()}
    except (IndexError, OSError, ValueError):
        return None
    if not states or set(states) != stable_tids:
        return None
    return states


def _open_pidfd(pid: int) -> int | None:
    if not callable(PIDFD_OPEN) or not callable(PIDFD_SEND_SIGNAL):
        return None
    try:
        return PIDFD_OPEN(pid)
    except OSError:
        return None


def _is_descendant(pid: int, root_pid: int) -> bool:
    seen: set[int] = set()
    while pid not in seen:
        if pid == root_pid:
            return True
        seen.add(pid)
        identity = _process_identity(pid)
        if identity is None:
            return False
        pid = identity[0]
    return False


def _signal_process_handle(pidfd: int, sig: int) -> bool:
    if not callable(PIDFD_SEND_SIGNAL):
        return False
    try:
        PIDFD_SEND_SIGNAL(pidfd, sig)
    except OSError:
        return False
    return True


def _wait_process_stopped(
    pid: int,
    identity: tuple[int, int],
    *,
    timeout: float = 0.25,
    deadline: float | None = None,
) -> bool:
    """Wait until SIGSTOP is delivered before scanning a process's children."""

    timeout_deadline = time.monotonic() + timeout
    wait_deadline = (
        min(timeout_deadline, deadline) if deadline is not None else timeout_deadline
    )
    while time.monotonic() < wait_deadline:
        if _process_identity(pid) != identity:
            return False
        states = _process_task_states(pid)
        if states and all(state in {"T", "t"} for state in states.values()):
            return _process_identity(pid) == identity
        if states is not None and any(
            state in {"Z", "X", "x"} for state in states.values()
        ):
            return False
        time.sleep(0.001)
    return False


def _freeze_process_handle(
    pid: int,
    pidfd: int,
    identity: tuple[int, int],
    *,
    timeout: float = 0.25,
    deadline: float | None = None,
) -> bool:
    return _signal_process_handle(pidfd, signal.SIGSTOP) and _wait_process_stopped(
        pid, identity, timeout=timeout, deadline=deadline
    )


def descendant_process_handles(
    root_pid: int,
    *,
    freeze: bool = False,
    handles: list[tuple[int, int]] | None = None,
    retry_timeout: float = 2.0,
    deadline: float | None = None,
) -> list[tuple[int, int]]:
    """Pin current descendants so a recycled PID can never be signaled."""

    owned = handles if handles is not None else []
    if freeze:
        frozen_parents = [root_pid]
        seen = {root_pid}
        retry_deadline = (
            deadline if deadline is not None else time.monotonic() + retry_timeout
        )
        while True:
            discovered = False
            unresolved = False
            for parent in tuple(frozen_parents):
                children = _process_children(parent)
                if children is None:
                    unresolved = True
                    continue
                for child in children:
                    if child in seen:
                        continue
                    remaining = retry_deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(
                            "global containment deadline reached before stable descendant snapshot"
                        )
                    identity = _process_identity(child)
                    if identity is None:
                        unresolved = True
                        continue
                    pidfd = _open_pidfd(child)
                    if pidfd is None:
                        unresolved = True
                        continue
                    handle = (child, pidfd)
                    owned.append(handle)
                    if _process_identity(child) != identity or not _is_descendant(
                        child, root_pid
                    ):
                        close_process_handles([handle])
                        owned.remove(handle)
                        unresolved = True
                        continue
                    remaining = retry_deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(
                            "global containment deadline reached before stable descendant snapshot"
                        )
                    if not _freeze_process_handle(
                        child,
                        pidfd,
                        identity,
                        timeout=min(0.25, remaining),
                        deadline=retry_deadline,
                    ):
                        _signal_process_handle(pidfd, signal.SIGKILL)
                        remaining = max(0.0, retry_deadline - time.monotonic())
                        if not _wait_process_handles(
                            [(child, pidfd)], timeout=remaining
                        ):
                            raise RuntimeError(
                                "descendant did not exit after failed freeze"
                            )
                        seen.add(child)
                        discovered = True
                        continue
                    seen.add(child)
                    frozen_parents.append(child)
                    discovered = True
            if time.monotonic() >= retry_deadline:
                raise RuntimeError(
                    "global containment deadline reached before stable descendant snapshot"
                )
            if unresolved:
                time.sleep(0.001)
                continue
            if not discovered:
                break
        return owned
    for pid in descendant_processes(root_pid):
        identity = _process_identity(pid)
        if identity is None:
            continue
        pidfd = _open_pidfd(pid)
        if pidfd is None:
            continue
        if _process_identity(pid) != identity or not _is_descendant(pid, root_pid):
            os.close(pidfd)
            continue
        owned.append((pid, pidfd))
    return owned


def signal_processes(processes: list[tuple[int, int]], sig: int) -> bool:
    """Signal a descendant snapshot deepest-first, ignoring exited processes."""

    if not callable(PIDFD_SEND_SIGNAL):
        return False
    delivered = True
    for _pid, pidfd in reversed(processes):
        try:
            PIDFD_SEND_SIGNAL(pidfd, sig)
        except OSError:
            delivered = False
    return delivered


def close_process_handles(processes: list[tuple[int, int]]) -> None:
    for _pid, pidfd in processes:
        try:
            os.close(pidfd)
        except OSError:
            pass


def _wait_process_handles(
    processes: list[tuple[int, int]], *, timeout: float = 2.0
) -> bool:
    deadline = time.monotonic() + timeout
    for _pid, pidfd in reversed(processes):
        poller = select.poll()
        poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if poller.poll(max(1, int(remaining * 1000))):
                break
    return True


def supervise_runner_worker(output: Any, worker: Callable[[], None]) -> None:
    """Keep a Linux subreaper alive while an expendable worker runs user code."""

    ack_read, ack_write = os.pipe()
    worker_pid = os.fork()
    if worker_pid == 0:
        exitcode = 0
        try:
            os.close(ack_read)
            try:
                worker()
            except BaseException as exc:
                output.put(
                    ("err", exc.__class__.__name__, str(exc), traceback.format_exc())
                )
            close = getattr(output, "close", None)
            join_thread = getattr(output, "join_thread", None)
            if callable(close):
                close()
            if callable(join_thread):
                join_thread()
            os.write(ack_write, b"1")
        except BaseException:
            exitcode = 1
        finally:
            try:
                os.close(ack_write)
            except OSError:
                pass
            os._exit(exitcode)
    os.close(ack_write)
    try:
        while True:
            try:
                _, status = os.waitpid(worker_pid, 0)
                break
            except InterruptedError:
                continue
        poller = select.poll()
        poller.register(ack_read, select.POLLIN | select.POLLHUP | select.POLLERR)
        acknowledged = bool(poller.poll(0)) and os.read(ack_read, 1) == b"1"
        if not acknowledged:
            exitcode = os.waitstatus_to_exitcode(status)
            output.put(
                (
                    "err",
                    "RuntimeError",
                    f"runner worker exited with code {exitcode}",
                    "",
                )
            )
    finally:
        os.close(ack_read)


def terminate_runner_process(proc: Any, cgroup: Path | None = None) -> None:
    deadline = time.monotonic() + 4.0
    pid = getattr(proc, "pid", None)
    root_pid = pid if isinstance(pid, int) else None
    root_handle: int | None = None
    root_validated = False
    descendants: list[tuple[int, int]] = []
    error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        if cgroup is not None and kill_trial_cgroup(cgroup):
            proc.join(min(2.0, max(0.0, deadline - time.monotonic())))
        else:
            linux = sys.platform.startswith("linux")
            root_identity = (
                _process_identity(root_pid) if root_pid is not None else None
            )
            root_handle = _open_pidfd(root_pid) if root_pid is not None else None
            if linux and root_pid is not None:
                if root_identity is None or root_handle is None:
                    raise RuntimeError("could not pin a stable runner root identity")
                if _process_identity(root_pid) != root_identity:
                    raise RuntimeError("could not revalidate stable root identity")
                root_validated = True
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not _freeze_process_handle(
                    root_pid,
                    root_handle,
                    root_identity,
                    timeout=min(0.25, remaining),
                    deadline=deadline,
                ):
                    raise RuntimeError("could not freeze runner root")
                descendant_process_handles(
                    root_pid, freeze=True, handles=descendants, deadline=deadline
                )
                signal_processes(descendants, signal.SIGKILL)
                remaining = max(0.0, deadline - time.monotonic())
                if not _wait_process_handles(descendants, timeout=remaining):
                    raise RuntimeError("descendants did not exit after SIGKILL")
                _signal_process_handle(root_handle, signal.SIGKILL)
                remaining = max(0.0, deadline - time.monotonic())
                if not _wait_process_handles(
                    [(root_pid, root_handle)], timeout=remaining
                ):
                    raise RuntimeError("runner did not exit after pidfd SIGKILL")
            else:
                if proc.is_alive():
                    try:
                        proc.terminate()
                    except (OSError, ProcessLookupError):
                        pass
            proc.join(min(2.0, max(0.0, deadline - time.monotonic())))
    except BaseException as exc:
        error = exc
    finally:
        try:
            if descendants:
                signal_processes(descendants, signal.SIGKILL)
                remaining = max(0.0, deadline - time.monotonic())
                if not _wait_process_handles(descendants, timeout=remaining):
                    raise RuntimeError("descendant cleanup did not quiesce")
            if root_validated and root_handle is not None:
                _signal_process_handle(root_handle, signal.SIGKILL)
        except BaseException as exc:
            cleanup_error = exc
        try:
            proc.join(0)
            if proc.is_alive():
                try:
                    proc.kill()
                except (OSError, ProcessLookupError):
                    pass
                proc.join(min(2.0, max(0.0, deadline - time.monotonic())))
            if proc.is_alive():
                raise RuntimeError("runner process remained alive after SIGKILL")
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        close_process_handles(descendants)
        if root_handle is not None and root_pid is not None:
            close_process_handles([(root_pid, root_handle)])
        if cgroup is not None:
            try:
                remove_trial_cgroup(cgroup)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
    if error is not None:
        raise error.with_traceback(error.__traceback__)
    if cleanup_error is not None:
        raise cleanup_error.with_traceback(cleanup_error.__traceback__)
