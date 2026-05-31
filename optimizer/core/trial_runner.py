import concurrent.futures
import time
import traceback

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


def now_ms():
    return int(time.time() * 1000)


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


def _response_hash(raw, name):
    hashes = _response_field(raw, "hashes", {}) or {}
    if isinstance(hashes, dict) and name in hashes:
        return hashes[name]
    return getattr(raw, name, None) if raw is not None else None


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
        return raw, {}, {}
    metrics = _response_field(raw, "metrics", {}) or {}
    if not isinstance(metrics, dict):
        raise ValueError("runner response metrics must be a dict")
    hashes = _response_field(raw, "hashes", {}) or {}
    if hashes is not None and not isinstance(hashes, dict):
        raise ValueError("runner response hashes must be a dict")
    return metrics, dict(hashes or {}), {"diagnostics": _response_diagnostics(raw, trial_id, params_hash)}


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
            raw = _call_with_timeout(lambda: runner(req), config.timeout_per_trial_sec)
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
            raw = _call_with_timeout(lambda: runner(params), config.timeout_per_trial_sec)
        metrics_source, response_hashes, response_meta = _normalize_runner_response(
            raw, trial_id, params_hash
        )
        diags.extend(response_meta.get("diagnostics", []))
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
        if outputs and _response_field(raw, "contract") == RUNNER_CONTRACT:
            if "closed_trades" in outputs and not bool(_response_field(raw, "trades_available", False)):
                unavailable_outputs.add("closed_trades")
            if "equity_curve" in outputs and not bool(_response_field(raw, "equity_available", False)):
                unavailable_outputs.add("equity_curve")
            if "summary_metrics" in outputs and not bool(_response_field(raw, "metrics", {})):
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
        metrics = MetricExtractor().extract(metrics_source)
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
            _response_hash(raw, "content_hash"),
            _response_hash(raw, "data_fingerprint"),
            _response_hash(raw, "runner_fingerprint"),
            _response_hash(raw, "engine_config_hash"),
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
