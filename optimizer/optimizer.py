from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
import os
from itertools import islice
from optimizer.config import OptimizerConfig
from optimizer.core.parameter_space import ParameterSpace
from optimizer.core.trial_runner import run_one
from optimizer.core.expression import stable_hash
from optimizer.results.leaderboard import rank_trials
from optimizer.results.result import OptimizerResult
from optimizer.selection.selector import build_profiles, choose_recommended
from optimizer.selection.baseline import baseline_comparison
from optimizer.storage.sqlite_backend import SQLiteStorage
from optimizer.storage.json_backend import JsonStorage
from optimizer.storage.checkpoint import check_resume
from optimizer.algorithms import grid_search, random_search, adaptive_grid, bayesian, genetic
from optimizer.core.diagnostic import Diagnostic
from optimizer.errors import UnsupportedFeatureError
from optimizer.results.trial import Trial


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


def _run_jobs(jobs, runner, config, space_hash, config_hash, store):
    trials = []
    if config.max_parallel > 1 and len(jobs) > 1:
        executor_cls = (
            ProcessPoolExecutor if config.parallel_backend == "process" else ThreadPoolExecutor
        )
        with executor_cls(max_workers=config.max_parallel) as ex:
            future_map = {
                ex.submit(run_one, tid, params, runner, config, space_hash, config_hash): tid
                for tid, params in jobs
            }
            iterator = (
                [f for f in future_map] if config.ordered_results else as_completed(future_map)
            )
            for f in iterator:
                t = f.result()
                store.save_trial(t)
                trials.append(t)
                if config.fail_fast and t.status != "completed":
                    break
                if (
                    config.max_failed_trials is not None
                    and sum(1 for x in trials if x.status != "completed")
                    >= config.max_failed_trials
                ):
                    break
    else:
        for tid, params in jobs:
            t = run_one(tid, params, runner, config, space_hash, config_hash)
            store.save_trial(t)
            trials.append(t)
            if config.fail_fast and t.status != "completed":
                break
            if (
                config.max_failed_trials is not None
                and sum(1 for x in trials if x.status != "completed") >= config.max_failed_trials
            ):
                break
    return trials


def _skipped_trial(tid, params, config, space_hash, config_hash, message):
    ph = stable_hash(params)
    direction = "maximize" if config.objective_direction == "auto" else config.objective_direction
    return Trial(
        tid,
        dict(params),
        {},
        None,
        direction,
        None,
        False,
        {"parameters": message},
        1,
        None,
        None,
        None,
        None,
        None,
        0.0,
        "skipped",
        parameter_space_hash=space_hash,
        optimizer_config_hash=config_hash,
        diagnostics=[
            Diagnostic(
                "INVALID_PARAM_COMBINATION",
                message,
                "warning",
                tid,
                ph,
                context={"params": dict(params)},
            )
        ],
        params_hash=ph,
    )


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
    check_resume(store, fingerprints, config.force_resume_on_fingerprint_mismatch)
    diagnostics: list[Diagnostic] = []
    _check_parallel_policy(config, diagnostics)
    raw_existing = store.load_trials_raw() if config.resume else []
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
        if config.algorithm == "grid" and space.cross_constraints:
            generated = 0
            for params, valid in space.iter_grid_with_validity():
                if generated >= config.max_trials:
                    break
                tid = next_id
                next_id += 1
                generated += 1
                ph = stable_hash(params)
                if ph in done_hashes:
                    continue
                if valid:
                    jobs.append((tid, params))
                else:
                    t = _skipped_trial(
                        tid,
                        params,
                        config,
                        space_hash,
                        config_hash,
                        "cross-constraint expression rejected parameter combination",
                    )
                    store.save_trial(t)
                    trials.append(t)
        elif config.algorithm == "random" and space.cross_constraints:
            generated = 0
            for params, valid in random_search.generate_with_validity(space, config):
                if generated >= config.max_trials:
                    break
                tid = next_id
                next_id += 1
                generated += 1
                ph = stable_hash(params)
                if ph in done_hashes:
                    continue
                if valid:
                    jobs.append((tid, params))
                else:
                    t = _skipped_trial(
                        tid,
                        params,
                        config,
                        space_hash,
                        config_hash,
                        "cross-constraint expression rejected parameter combination",
                    )
                    store.save_trial(t)
                    trials.append(t)
        else:
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
    ranked = rank_trials(trials)
    profiles, pf = build_profiles(trials, config)
    if hasattr(store, "save_profile"):
        for p in profiles.values():
            store.save_profile(p)
    rec, rec_name = choose_recommended(profiles, config.selection_mode)
    counts = {
        s: sum(1 for t in trials if t.status == s)
        for s in ["completed", "failed", "timeout", "skipped"]
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
    res = OptimizerResult(
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
    )
    return res
