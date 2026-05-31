import concurrent.futures
import multiprocessing as mp
import pickle
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any

from optimizer.core.metric_extractor import MetricExtractor
from optimizer.core.metric_registry import MetricRegistry
from optimizer.core.constraints import evaluate_constraints, merge_constraints
from optimizer.core.objective import compute_objective, objective_direction
from optimizer.core.normalization import balanced_score
from optimizer.core.expression import stable_hash
from optimizer.results.trial import Trial
from optimizer.core.diagnostic import Diagnostic
from optimizer.protocols import RunnerRequest
from optimizer.version import __version__

RUNNER_CONTRACT = "pain.optimizer_runner.v1"


@dataclass(frozen=True)
class NormalizedRunnerResponse:
    metrics_source: Any
    hashes: dict[str, str]
    diagnostics: list[Diagnostic]
    is_contract_response: bool = False
    trades_available: bool = False
    equity_available: bool = False

    @property
    def summary_metrics_available(self) -> bool:
        return bool(self.metrics_source)

    def hash(self, name: str, raw: Any) -> str | None:
        if name in self.hashes:
            return self.hashes[name]
        value = getattr(raw, name, None) if raw is not None else None
        return None if value is None else str(value)


def now_ms():
    return int(time.time() * 1000)


def _invoke_runner(runner, payload):
    return runner(payload)


def _runner_process_entry(out, runner, payload):
    try:
        out.put(("ok", _invoke_runner(runner, payload)))
    except BaseException as exc:  # pragma: no cover - exercised through parent process result
        out.put(("err", exc.__class__.__name__, str(exc), traceback.format_exc()))


def _is_picklable(value):
    try:
        pickle.dumps(value)
    except Exception:
        return False
    return True


def _call_runner_in_process(runner, payload, timeout):
    if timeout is None or timeout <= 0:
        return _invoke_runner(runner, payload)
    ctx = mp.get_context("spawn")
    out = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_runner_process_entry, args=(out, runner, payload))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise concurrent.futures.TimeoutError("trial timeout")
    try:
        status, *parts = out.get_nowait()
    except queue.Empty:
        if proc.exitcode == 0:
            raise RuntimeError("runner process exited without a result")
        raise RuntimeError(f"runner process exited with code {proc.exitcode}")
    if status == "ok":
        return parts[0]
    exc_name, message, tb = parts
    raise RuntimeError(f"{exc_name}: {message}\n{tb}")


def _select_timeout_backend(config, runner, payload, diagnostics, trial_id, params_hash):
    backend = getattr(config, "timeout_backend", "thread")
    timeout = getattr(config, "timeout_per_trial_sec", None)
    if backend == "thread" or timeout is None or timeout <= 0:
        return "thread"
    if _is_picklable((runner, payload)):
        return "process"
    if backend == "process":
        raise ValueError("process timeout backend requires a picklable runner and request")
    diagnostics.append(
        Diagnostic(
            "RUNNER_TIMEOUT_THREAD_FALLBACK",
            "runner/request is not picklable; timeout isolation fell back to thread backend",
            "warning",
            trial_id,
            params_hash,
        )
    )
    return "thread"


def _call_with_timeout(fn, timeout):
    if timeout is None or timeout <= 0:
        return fn()
    # Thread backend cannot forcibly kill arbitrary Python/user code, but this
    # returns control on timeout instead of blocking in executor shutdown.
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _call_runner_with_timeout(runner, payload, timeout, backend):
    if backend == "process":
        return _call_runner_in_process(runner, payload, timeout)
    return _call_with_timeout(lambda: _invoke_runner(runner, payload), timeout)


def _required_metrics(config):
    reg = MetricRegistry()
    required = set(merge_constraints(config).keys())
    if config.objective_expression:
        required |= reg.extract_expression_metrics(config.objective_expression)
    else:
        required.add(config.objective)
    if config.objective_secondary:
        required.add(config.objective_secondary)
    if config.report_profiles and not config.objective_expression:
        required |= reg.profile_required_metrics(
            ["best_profit", "best_drawdown", "best_profit_factor", "best_sharpe", "best_balanced"]
        )
    return reg.get_required_metrics(required)


def _response_field(raw, name, default=None):
    if isinstance(raw, dict):
        return raw.get(name, default)
    return getattr(raw, name, default)


def _response_diagnostics(raw, trial_id, params_hash):
    out = []
    for item in _response_field(raw, "diagnostics", []) or []:
        if isinstance(item, Diagnostic):
            out.append(item)
            continue
        if isinstance(item, dict):
            out.append(
                Diagnostic(
                    str(item.get("code", "RUNNER_DIAGNOSTIC")),
                    str(item.get("message", "runner diagnostic")),
                    str(item.get("severity", "warning")),
                    trial_id,
                    params_hash,
                    context=item.get("context", {}),
                )
            )
    return out


def _normalize_runner_response(raw, trial_id, params_hash):
    contract = _response_field(raw, "contract")
    has_response_shape = contract is not None or _response_field(raw, "metrics") is not None
    if contract is not None and contract != RUNNER_CONTRACT:
        raise ValueError(f"runner response contract mismatch: {contract!r} != {RUNNER_CONTRACT!r}")
    if not has_response_shape:
        return NormalizedRunnerResponse(metrics_source=raw, hashes={}, diagnostics=[])
    metrics = _response_field(raw, "metrics", {}) or {}
    if not isinstance(metrics, dict):
        raise ValueError("runner response metrics must be a dict")
    hashes = _response_field(raw, "hashes", {}) or {}
    if hashes is not None and not isinstance(hashes, dict):
        raise ValueError("runner response hashes must be a dict")
    return NormalizedRunnerResponse(
        metrics_source=metrics,
        hashes={str(key): str(value) for key, value in dict(hashes or {}).items()},
        diagnostics=_response_diagnostics(raw, trial_id, params_hash),
        is_contract_response=contract == RUNNER_CONTRACT,
        trades_available=bool(_response_field(raw, "trades_available", False)),
        equity_available=bool(_response_field(raw, "equity_available", False)),
    )


def _failed_trial(
    trial_id,
    params,
    metrics,
    direction,
    status,
    err,
    tr,
    diags,
    started,
    t0,
    space_hash,
    config_hash,
    is_baseline=False,
    baseline_name=None,
):
    return Trial(
        trial_id,
        dict(params),
        metrics,
        None,
        direction,
        None,
        False,
        {},
        0,
        None,
        None,
        None,
        None,
        None,
        time.perf_counter() - t0,
        status,
        is_baseline=is_baseline,
        baseline_name=baseline_name,
        parameter_space_hash=space_hash,
        optimizer_config_hash=config_hash,
        code_version=__version__,
        error_message=err,
        traceback=tr,
        diagnostics=diags,
        missing_metrics=set(),
        started_at=started,
        finished_at=now_ms(),
        params_hash=stable_hash(params),
    )


def run_one(
    trial_id, params, runner, config, space_hash, config_hash, is_baseline=False, baseline_name=None
):
    started = now_ms()
    t0 = time.perf_counter()
    diags = []
    metrics = {}
    raw = None
    params_hash = stable_hash(params)
    direction = (
        config.objective_direction
        if config.objective_expression and config.objective_direction != "auto"
        else (
            "maximize"
            if config.objective_expression
            else objective_direction(config.objective, config.objective_direction)
        )
    )
    constraints = merge_constraints(config)
    try:
        caps = getattr(runner, "capabilities", None)
        required = _required_metrics(config)
        outputs = MetricRegistry().get_required_outputs(required)
        if caps and getattr(caps, "supports_required_outputs", False):
            supported_outputs = set(getattr(caps, "supported_outputs", set()) or set())
            missing_outputs = outputs - supported_outputs
            if missing_outputs:
                err = "runner does not support required outputs: " + ", ".join(
                    sorted(missing_outputs)
                )
                diags.append(
                    Diagnostic(
                        "RUNNER_REQUIRED_OUTPUT_UNSUPPORTED",
                        err,
                        "error",
                        trial_id,
                        params_hash,
                        context={
                            "required_outputs": sorted(outputs),
                            "supported_outputs": sorted(supported_outputs),
                            "missing_outputs": sorted(missing_outputs),
                        },
                    )
                )
                return _failed_trial(
                    trial_id,
                    params,
                    metrics,
                    direction,
                    "failed",
                    err,
                    None,
                    diags,
                    started,
                    t0,
                    space_hash,
                    config_hash,
                    is_baseline,
                    baseline_name,
                )
        if caps and getattr(caps, "supports_runner_request", False):
            req = RunnerRequest(
                params=params,
                trial_id=trial_id,
                required_metrics=required,
                required_outputs=outputs,
                early_stop_conditions=config.early_stop_conditions
                if config.early_stop_enabled
                else [],
                seed=config.seed if getattr(caps, "supports_seed", False) else None,
                fingerprints={
                    "parameter_space_hash": space_hash,
                    "optimizer_config_hash": config_hash,
                },
            )
            timeout_backend = _select_timeout_backend(
                config, runner, req, diags, trial_id, params_hash
            )
            raw = _call_runner_with_timeout(
                runner,
                req,
                config.timeout_per_trial_sec,
                timeout_backend,
            )
        else:
            if outputs or config.early_stop_enabled:
                diags.append(
                    Diagnostic(
                        "BASIC_RUNNER_CONTRACT_USED",
                        "runner called with params only; advanced hints unavailable",
                        "warning",
                        trial_id,
                        params_hash,
                    )
                )
            timeout_backend = _select_timeout_backend(
                config, runner, params, diags, trial_id, params_hash
            )
            raw = _call_runner_with_timeout(
                runner,
                params,
                config.timeout_per_trial_sec,
                timeout_backend,
            )
        response = _normalize_runner_response(raw, trial_id, params_hash)
        diags.extend(response.diagnostics)
        runner_errors = [d for d in diags if getattr(d, "severity", None) == "error"]
        if runner_errors:
            err = "runner returned error diagnostics: " + "; ".join(
                f"{d.code}: {d.message}" for d in runner_errors
            )
            return _failed_trial(
                trial_id,
                params,
                metrics,
                direction,
                "failed",
                err,
                None,
                diags,
                started,
                t0,
                space_hash,
                config_hash,
                is_baseline,
                baseline_name,
            )
        unavailable_outputs = set()
        if outputs and response.is_contract_response:
            if "closed_trades" in outputs and not response.trades_available:
                unavailable_outputs.add("closed_trades")
            if "equity_curve" in outputs and not response.equity_available:
                unavailable_outputs.add("equity_curve")
            if "summary_metrics" in outputs and not response.summary_metrics_available:
                unavailable_outputs.add("summary_metrics")
        if caps and getattr(caps, "supports_required_outputs", False) and unavailable_outputs:
            err = "runner response missing required outputs: " + ", ".join(
                sorted(unavailable_outputs)
            )
            diags.append(
                Diagnostic(
                    "RUNNER_REQUIRED_OUTPUT_MISSING",
                    err,
                    "error",
                    trial_id,
                    params_hash,
                    context={
                        "required_outputs": sorted(outputs),
                        "missing_outputs": sorted(unavailable_outputs),
                    },
                )
            )
            return _failed_trial(
                trial_id,
                params,
                metrics,
                direction,
                "failed",
                err,
                None,
                diags,
                started,
                t0,
                space_hash,
                config_hash,
                is_baseline,
                baseline_name,
            )
        metrics = MetricExtractor().extract(response.metrics_source)
        if (
            "return_drawdown_ratio" not in metrics
            and metrics.get("net_profit") is not None
            and metrics.get("max_drawdown")
        ):
            metrics["return_drawdown_ratio"] = metrics["net_profit"] / abs(metrics["max_drawdown"])
        missing_required = set(required) - set(metrics)
        if caps and getattr(caps, "supports_required_outputs", False) and missing_required:
            err = "runner response missing required metrics: " + ", ".join(sorted(missing_required))
            diags.append(
                Diagnostic(
                    "RUNNER_REQUIRED_METRICS_MISSING",
                    err,
                    "error",
                    trial_id,
                    params_hash,
                    context={
                        "required_metrics": sorted(required),
                        "missing_metrics": sorted(missing_required),
                    },
                )
            )
            return _failed_trial(
                trial_id,
                params,
                metrics,
                direction,
                "failed",
                err,
                None,
                diags,
                started,
                t0,
                space_hash,
                config_hash,
                is_baseline,
                baseline_name,
            )
        obj = compute_objective(metrics, config.objective, direction, config.objective_expression)
        c = evaluate_constraints(metrics, constraints, trial_id=trial_id, params_hash=params_hash)
        diags.extend(c.diagnostics)
        effective_obj = obj
        if (
            config.constraint_mode in {"penalty", "both"}
            and c.penalty
            and effective_obj is not None
        ):
            mult = float(config.constraint_penalty_multiplier)
            effective_obj = (
                effective_obj - c.penalty * mult
                if direction == "maximize"
                else effective_obj + c.penalty * mult
            )
        bs = balanced_score(metrics)
        r = getattr(raw, "to_dict", lambda: raw if isinstance(raw, dict) else None)()
        # `passed_constraints` means hard-threshold eligibility, not whether the
        # trial remains visible in the leaderboard. In `penalty` mode a hard
        # violator is kept and penalized, but it must still be ineligible for
        # non-best_objective recommendations.
        trial = Trial(
            trial_id,
            dict(params),
            metrics,
            effective_obj,
            direction,
            None,
            c.hard_passed,
            {**c.violations, **c.soft_violations},
            len(c.violations) + len(c.soft_violations),
            bs,
            metrics.get("robustness_score"),
            metrics.get("overfitting_score"),
            metrics.get("profit_concentration_score"),
            r if config.save_backtest_result else None,
            time.perf_counter() - t0,
            "completed",
            False,
            is_baseline,
            baseline_name,
            response.hash("content_hash", raw),
            response.hash("data_fingerprint", raw),
            response.hash("runner_fingerprint", raw),
            response.hash("engine_config_hash", raw),
            space_hash,
            config_hash,
            __version__,
            None,
            None,
            diags,
            set(required) - set(metrics),
            started,
            now_ms(),
            params_hash,
            obj,
        )
        return trial
    except concurrent.futures.TimeoutError:
        status = "failed"
        err = "trial timeout"
        tr = traceback.format_exc()
        diags.append(Diagnostic("TRIAL_TIMEOUT", err, "error", trial_id, params_hash))
    except Exception as e:
        status = "failed"
        err = str(e)
        tr = traceback.format_exc()
        diags.append(Diagnostic("TRIAL_FAILED", err, "error", trial_id, params_hash))
    return _failed_trial(
        trial_id,
        params,
        metrics,
        direction,
        status,
        err,
        tr,
        diags,
        started,
        t0,
        space_hash,
        config_hash,
        is_baseline,
        baseline_name,
    )
