from pathlib import Path
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
import json
import os
from itertools import islice
from optimizer.config import OptimizerConfig
from optimizer.core.parameter_space import ParameterSpace
from optimizer.core.trial_runner import run_one
from optimizer.core.expression import stable_hash
from optimizer.results.leaderboard import rank_trials
from optimizer.results.result import OptimizerRunResult
from optimizer.selection.selector import build_profiles, choose_recommended
from optimizer.selection.baseline import baseline_comparison
from optimizer.storage.sqlite_backend import SQLiteStorage
from optimizer.storage.json_backend import JsonStorage
from optimizer.storage.checkpoint import check_resume
from optimizer.algorithms import grid_search, random_search, adaptive_grid, bayesian, genetic
from optimizer.core.diagnostic import Diagnostic
from optimizer.errors import UnsupportedFeatureError
from optimizer.results.trial import Trial
from optimizer.requests import OptimizerRunRequest


SUPPORTED_OPTIMIZE_ALGORITHMS = {
    "grid",
    "random",
    "adaptive_grid",
    "genetic",
    "bayesian",
    "walk_forward",
}


def _storage(config):
    return (
        JsonStorage(config.output_dir)
        if config.storage_backend == "json"
        else SQLiteStorage(config.output_dir)
    )


def _key(params):
    return json.dumps(params, sort_keys=True, default=str)


def _runner_fingerprint(runner):
    fp = getattr(runner, "fingerprint", None)
    if callable(fp):
        return fp()
    return fp


def _params_for(space, config):
    if config.algorithm == "grid":
        return islice(grid_search.generate(space, config), config.max_trials)
    if config.algorithm == "random":
        return islice(random_search.generate(space, config), config.max_trials)
    if config.algorithm == "adaptive_grid":
        return islice(adaptive_grid.initial(space, config), config.max_trials)
    raise UnsupportedFeatureError(
        f"Unknown optimizer algorithm {config.algorithm!r}; supported values are "
        f"{', '.join(sorted(SUPPORTED_OPTIMIZE_ALGORITHMS))}"
    )


def _trial_from_raw(d):
    if not d:
        return None
    fields = getattr(Trial, "__dataclass_fields__", {})
    payload = {k: v for k, v in d.items() if k in fields}
    diag_fields = getattr(Diagnostic, "__dataclass_fields__", {})
    payload["diagnostics"] = [
        x
        if isinstance(x, Diagnostic)
        else Diagnostic(**{k: v for k, v in x.items() if k in diag_fields})
        for x in payload.get("diagnostics", [])
    ]
    payload["missing_metrics"] = set(payload.get("missing_metrics", []))
    return Trial(**payload)


def _check_parallel_policy(config, diagnostics):
    cpu = os.cpu_count() or 1
    if config.max_parallel > cpu:
        msg = f"max_parallel={config.max_parallel} exceeds cpu_count={cpu}"
        if config.max_parallel_over_cpu_policy == "error":
            raise ValueError(msg)
        if config.max_parallel_over_cpu_policy == "warn":
            diagnostics.append(
                Diagnostic(
                    "MAX_PARALLEL_OVER_CPU",
                    msg,
                    "warning",
                    context={"max_parallel": config.max_parallel, "cpu_count": cpu},
                )
            )


def _is_proven(value):
    return value is True or (isinstance(value, str) and value.lower() == "proven")


def _lookup_nested(payload, *names):
    if not isinstance(payload, dict):
        return None
    for name in names:
        if name in payload:
            return payload[name]
    gates = payload.get("oracle_gates") or payload.get("oracleGates") or payload.get("gates")
    if isinstance(gates, dict):
        for name in names:
            if name in gates:
                return gates[name]
    return None


def _data_query_risk_reasons(payload):
    reasons = set()

    def walk(value, key=""):
        key_l = str(key).lower()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item, key)
            return
        text = str(value).lower() if isinstance(value, str) else ""
        truthy = value is True or (isinstance(value, str) and text in {"true", "yes", "1"})
        if key_l in {"realtime", "live", "use_realtime", "allow_realtime"} and truthy:
            reasons.add("realtime")
        if key_l in {"intrabar", "use_intrabar", "bar_magnifier", "use_bar_magnifier"} and truthy:
            reasons.add("intrabar")
        if key_l in {"tick", "ticks", "tick_data", "du_tick_data"} and truthy:
            reasons.add("tick")
        if key_l in {"mode", "kind", "type", "data_type", "source", "feed"} and text in {
            "realtime",
            "live",
            "tick",
            "ticks",
            "intrabar",
            "du_tick",
        }:
            reasons.add(text)
        if key_l in {"lower_timeframe", "lower_tf", "intrabar_timeframe"} and value not in {
            None,
            "",
            False,
        }:
            reasons.add("intrabar")

    walk(payload)
    return reasons


def _validate_data_query(data_query):
    reasons = _data_query_risk_reasons(data_query)
    if not reasons:
        return None
    required = {
        "tvRealtimeBoundary": _lookup_nested(
            data_query, "tvRealtimeBoundary", "tv_realtime_boundary", "final_tick_commit"
        ),
        "duTickCompleteness": _lookup_nested(
            data_query, "duTickCompleteness", "du_tick_completeness", "tick_completeness"
        ),
        "intrabarOrderFill": _lookup_nested(
            data_query, "intrabarOrderFill", "intrabar_order_fill", "intrabar_fill_oracle"
        ),
    }
    missing = [name for name, value in required.items() if not _is_proven(value)]
    if not missing:
        return None
    return Diagnostic(
        "UNPROVEN_REALTIME_INTRABAR_DATA_QUERY",
        "realtime/tick/intrabar optimizer inputs require proven oracle gates",
        "error",
        context={
            "risk_reasons": sorted(reasons),
            "missing_or_unproven_gates": missing,
        },
    )


def _failed_request_result(request, diagnostic, output_dir):
    return OptimizerRunResult(
        None,
        None,
        None,
        [],
        [],
        [],
        [],
        str(output_dir),
        {"completed": 0, "failed": 0},
        diagnostics=[diagnostic],
        run_id=request.run_id,
        status="failed",
        trials=(),
        artifact_path=Path(output_dir),
        data_query=request.data_query,
    )


def _stop_requested(trials, config):
    if config.fail_fast and any(t.status != "completed" for t in trials):
        return True
    return (
        config.max_failed_trials is not None
        and sum(1 for x in trials if x.status != "completed") >= config.max_failed_trials
    )


def _run_jobs(jobs, runner, config, space_hash, config_hash, store):
    trials = []
    if config.max_parallel > 1 and len(jobs) > 1:
        executor_cls = (
            ProcessPoolExecutor if config.parallel_backend == "process" else ThreadPoolExecutor
        )
        with executor_cls(max_workers=config.max_parallel) as ex:
            job_iter = iter(jobs)
            pending = {}

            def submit_next():
                try:
                    tid, params = next(job_iter)
                except StopIteration:
                    return False
                fut = ex.submit(run_one, tid, params, runner, config, space_hash, config_hash)
                pending[fut] = tid
                return True

            for _ in range(min(config.max_parallel, len(jobs))):
                submit_next()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    pending.pop(f, None)
                    t = f.result()
                    store.save_trial(t)
                    trials.append(t)
                if _stop_requested(trials, config):
                    for pending_future in pending:
                        pending_future.cancel()
                    pending.clear()
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
                while len(pending) < config.max_parallel and submit_next():
                    pass
        if config.ordered_results:
            order = {tid: idx for idx, (tid, _params) in enumerate(jobs)}
            trials.sort(key=lambda t: order.get(t.id, t.id))
    else:
        for tid, params in jobs:
            t = run_one(tid, params, runner, config, space_hash, config_hash)
            store.save_trial(t)
            trials.append(t)
            if _stop_requested(trials, config):
                break
    return trials


def _sequential_advanced(space, runner, config, space_hash, config_hash, store, next_id):
    trials = []
    seen = set()
    if config.algorithm == "genetic":
        population = genetic.initial_population(space, config)
        evaluated = []
        for generation in range(max(1, config.genetic_generations)):
            jobs = []
            for params in population:
                if len(trials) + len(jobs) >= config.max_trials:
                    break
                k = _key(params)
                if k in seen:
                    continue
                seen.add(k)
                jobs.append((next_id, params))
                next_id += 1
            batch = _run_jobs(jobs, runner, config, space_hash, config_hash, store)
            trials.extend(batch)
            evaluated.extend(
                (t.params, float(t.objective_value))
                for t in batch
                if t.status == "completed" and t.objective_value is not None
            )
            if _stop_requested(trials, config):
                break
            if len(trials) >= config.max_trials or not evaluated:
                break
            population = genetic.next_generation(space, config, evaluated, seen, generation)
            if not population:
                break
        return trials, next_id
    if config.algorithm == "bayesian":
        evaluated = []
        for params in bayesian.warmup(space, config):
            if len(trials) >= config.max_trials:
                break
            k = _key(params)
            if k in seen:
                continue
            seen.add(k)
            batch = _run_jobs([(next_id, params)], runner, config, space_hash, config_hash, store)
            next_id += 1
            trials.extend(batch)
            evaluated.extend(
                (t.params, float(t.objective_value))
                for t in batch
                if t.status == "completed" and t.objective_value is not None
            )
            if _stop_requested(trials, config):
                break
        iteration = 0
        target = min(config.bayesian_trials, config.max_trials)
        while len(trials) < target and evaluated:
            params = bayesian.propose(space, config, evaluated, seen, iteration)
            iteration += 1
            if params is None:
                break
            seen.add(_key(params))
            batch = _run_jobs([(next_id, params)], runner, config, space_hash, config_hash, store)
            next_id += 1
            trials.extend(batch)
            evaluated.extend(
                (t.params, float(t.objective_value))
                for t in batch
                if t.status == "completed" and t.objective_value is not None
            )
            if _stop_requested(trials, config):
                break
        return trials, next_id
    raise UnsupportedFeatureError(
        f"Unknown optimizer algorithm {config.algorithm!r}; supported values are "
        f"{', '.join(sorted(SUPPORTED_OPTIMIZE_ALGORITHMS))}"
    )


def _analysis(config, trials, space):
    out = {}
    if config.compute_robustness or config.robustness_enabled or config.analysis_profile == "full":
        from optimizer.analysis.robustness import analyze

        out["robustness"] = analyze(
            trials, space, config.robustness_neighbor_radius_steps, config.robustness_min_neighbors
        )
    if (
        config.compute_sensitivity
        or config.compute_parameter_importance
        or config.analysis_profile == "full"
    ):
        from optimizer.analysis.sensitivity import analyze

        out["sensitivity"] = analyze(trials)
    if config.compute_overfitting or config.analysis_profile == "full":
        from optimizer.analysis.overfitting import analyze

        out["overfitting"] = analyze(trials, config.objective)
    if config.compute_profit_concentration or config.analysis_profile == "full":
        from optimizer.analysis.profit_concentration import analyze

        out["profit_concentration"] = analyze(trials)
    if config.compute_monte_carlo or config.analysis_profile == "full":
        from optimizer.analysis.monte_carlo import analyze

        out["monte_carlo"] = analyze(trials, config.monte_carlo_simulations, config.seed)
    return out


def optimize(
    parameters,
    runner,
    config: OptimizerConfig | None = None,
    *,
    cross_constraints=None,
    start: int | None = None,
    end: int | None = None,
):
    config = config or OptimizerConfig()
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(parameters, ParameterSpace):
        space = parameters
    else:
        space = ParameterSpace(
            parameters,
            cross_constraints if cross_constraints is not None else config.cross_constraints,
        )
    if config.algorithm == "walk_forward":
        if start is None or end is None:
            raise ValueError(
                "walk_forward optimize() requires explicit start=... and end=... range bounds"
            )
        if end <= start:
            raise ValueError("walk_forward optimize() requires end > start")
        from optimizer.algorithms.walk_forward import run as walk_forward_run

        return walk_forward_run(space, runner, config, start=start, end=end)
    space_hash = space.fingerprint()
    config_hash = stable_hash(config.to_dict())
    fingerprints = {
        "parameter_space_hash": space_hash,
        "optimizer_config_hash": config_hash,
        "data_fingerprint": None,
        "runner_fingerprint": _runner_fingerprint(runner),
        "engine_config_hash": None,
    }
    store = _storage(config)
    previous_run_metadata = check_resume(
        store, fingerprints, config.force_resume_on_fingerprint_mismatch
    )
    diagnostics: list[Diagnostic] = []
    _check_parallel_policy(config, diagnostics)
    raw_existing = store.load_trials_raw() if config.resume else []
    if config.resume and previous_run_metadata and not raw_existing:
        from optimizer.errors import StorageError

        raise StorageError("resume metadata exists but no persisted optimizer trials were found")
    existing_trials = [t for t in (_trial_from_raw(x) for x in raw_existing) if t is not None]
    done_hashes = {
        x.get("params_hash") or stable_hash(x.get("params", {}))
        for x in raw_existing
        if x.get("status") == "completed"
    }
    trials = list(existing_trials)
    next_id = max([t.id for t in trials] or [0]) + 1
    if (
        config.baseline_params
        and config.run_baseline_first
        and stable_hash(config.baseline_params) not in done_hashes
    ):
        t = run_one(
            0,
            config.baseline_params,
            runner,
            config,
            space_hash,
            config_hash,
            True,
            config.baseline_name,
        )
        store.save_trial(t)
        trials.append(t)
    if config.algorithm in {"genetic", "bayesian"}:
        advanced, next_id = _sequential_advanced(
            space, runner, config, space_hash, config_hash, store, next_id
        )
        trials.extend(advanced)
    else:
        jobs = []
        for params in _params_for(space, config):
            tid = next_id
            next_id += 1
            if stable_hash(params) not in done_hashes:
                jobs.append((tid, params))
        trials.extend(_run_jobs(jobs, runner, config, space_hash, config_hash, store))
    if config.algorithm == "adaptive_grid" and trials:
        jobs = []
        for p in adaptive_grid.refine(space, trials, config):
            if len(trials) + len(jobs) >= config.max_trials:
                break
            jobs.append((next_id, p))
            next_id += 1
        trials.extend(_run_jobs(jobs, runner, config, space_hash, config_hash, store))
    ranked = rank_trials(trials, config)
    profiles, pf = build_profiles(trials, config)
    if hasattr(store, "save_profile"):
        for p in profiles.values():
            store.save_profile(p)
    rec, rec_name = choose_recommended(profiles, config.selection_mode)
    counts = {
        s: sum(1 for t in trials if t.status == s)
        for s in ["completed", "failed"]
    }
    baseline_trial = next((t for t in trials if t.is_baseline), None)
    base_cmp = baseline_comparison(baseline_trial, rec)
    if base_cmp.get("recommended_worse_than_baseline"):
        diagnostics.append(
            Diagnostic(
                "RECOMMENDED_WORSE_THAN_BASELINE",
                "Recommended trial is worse than configured baseline",
                "warning",
                context=base_cmp,
            )
        )
    diagnostics.append(
        Diagnostic(
            "ADVANCED_FEATURES_ACTIVE",
            "Advanced algorithms/analysis run natively when enabled; optional reports degrade with diagnostics if dependencies/data are missing",
            "info",
        )
    )
    if not [t for t in trials if t.status == "completed" and t.passed_constraints]:
        causes: dict[str, int] = {}
        for t in trials:
            for k, v in (t.constraint_violations or {}).items():
                causes[f"{k}: {v}"] = causes.get(f"{k}: {v}", 0) + 1
        diagnostics.append(
            Diagnostic(
                "NO_TRIALS_PASSED_CONSTRAINTS",
                "No completed trials passed hard constraints",
                "warning",
                context={
                    "top_rejection_causes": sorted(
                        causes.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                },
            )
        )
    completed_count = sum(1 for t in trials if t.status == "completed")
    min_completed_satisfied = completed_count >= config.min_completed_trials
    if not min_completed_satisfied:
        diagnostics.append(
            Diagnostic(
                "MIN_COMPLETED_TRIALS_NOT_MET",
                (
                    f"completed trials {completed_count} is below "
                    f"min_completed_trials={config.min_completed_trials}"
                ),
                "error",
                context={
                    "completed_trials": completed_count,
                    "min_completed_trials": config.min_completed_trials,
                },
            )
        )
    res = OptimizerRunResult(
        rec,
        rec_name,
        profiles["best_objective"].trial,
        ranked[: config.top_n],
        trials if config.save_all_trials else None,
        [t for t in trials if t.passed_constraints],
        [t for t in trials if t.status == "completed" and not t.passed_constraints],
        str(getattr(store, "path", config.output_dir)),
        counts,
        profiles,
        profiles["best_objective"].trial,
        profiles["best_passed_constraints"].trial,
        profiles["best_balanced"].trial,
        profiles["most_robust"].trial,
        profiles["best_profit"].trial,
        profiles["best_drawdown"].trial,
        profiles["best_profit_factor"].trial,
        profiles["best_sharpe"].trial,
        pf,
        diagnostics=diagnostics,
        analysis=_analysis(config, trials, space),
        baseline_trial=baseline_trial,
        baseline_comparison=base_cmp,
        run_id=config.run_id or stable_hash(
            {
                "parameter_space_hash": space_hash,
                "optimizer_config_hash": config_hash,
                "runner_fingerprint": fingerprints["runner_fingerprint"],
            }
        )[:16],
        status=(
            "completed"
            if min_completed_satisfied
            and any(t.status == "completed" and t.objective_value is not None for t in trials)
            else "failed"
        ),
        best_params=(
            rec.params
            if min_completed_satisfied and rec is not None and rec.status == "completed"
            else None
        ),
        best_score=(
            rec.objective_value
            if min_completed_satisfied and rec is not None and rec.status == "completed"
            else None
        ),
        trials=tuple(trials),
        artifact_path=Path(getattr(store, "path", config.output_dir)),
    )
    return res


def optimize_request(
    request: OptimizerRunRequest,
    runner,
    config: OptimizerConfig | None = None,
) -> OptimizerRunResult:
    cfg = config or OptimizerConfig()
    cfg.output_dir = Path(cfg.output_dir)
    data_query_diagnostic = _validate_data_query(request.data_query)
    if data_query_diagnostic is not None:
        return _failed_request_result(request, data_query_diagnostic, cfg.output_dir)
    cfg.run_id = request.run_id
    cfg.objective = request.objective.metric
    cfg.objective_direction = request.objective.direction
    cfg.objective_expression = request.objective.expression
    cfg.constraints = dict(request.constraints.metrics)
    cfg.cross_constraints = list(request.constraints.cross_constraints)
    result = optimize(request.parameter_space, runner, cfg)
    result.data_query = request.data_query
    return result
