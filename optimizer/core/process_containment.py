"""Best-effort process lifecycle containment for optimizer trials."""

from __future__ import annotations

import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any

CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_SELF_CGROUP = Path("/proc/self/cgroup")


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
        except OSError:
            time.sleep(0.01)


def terminate_runner_process(proc: Any, cgroup: Path | None = None) -> None:
    if cgroup is not None:
        kill_trial_cgroup(cgroup)
        proc.join(2)
        if proc.is_alive():
            try:
                proc.kill()
            except (OSError, ProcessLookupError):
                pass
        proc.join()
        remove_trial_cgroup(cgroup)
        return
    group = isolated_process_group(proc)
    if group is None:
        try:
            proc.terminate()
        except (OSError, ProcessLookupError):
            pass
    else:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Do not join/reap the leader before the final group signal: keeping its
        # PID allocated closes the PGID reuse race against unrelated processes.
        time.sleep(0.05)
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.join(2)
    if group is None and proc.is_alive():
        try:
            proc.kill()
        except (OSError, ProcessLookupError):
            pass
    proc.join()
