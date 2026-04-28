from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from optimizer.config import OptimizerConfig
from optimizer.core.parameter_space import ParameterSpace
from optimizer.core.trial_runner import run_one
from optimizer.core.expression import stable_hash
from optimizer.results.leaderboard import rank_trials
from optimizer.results.result import OptimizerResult
from optimizer.selection.selector import build_profiles, choose_recommended
from optimizer.storage.sqlite_backend import SQLiteStorage
from optimizer.storage.json_backend import JsonStorage
from optimizer.storage.checkpoint import check_resume
from optimizer.algorithms import grid_search, random_search, adaptive_grid, bayesian, genetic
from optimizer.core.diagnostic import Diagnostic
from optimizer.errors import UnsupportedFeatureError


def _storage(config):
    return JsonStorage(config.output_dir) if config.storage_backend == 'json' else SQLiteStorage(config.output_dir)


def _key(params):
    return json.dumps(params, sort_keys=True, default=str)


def _params_for(space, config):
    if config.algorithm == 'grid':
        return list(grid_search.generate(space, config))[: config.max_trials]
    if config.algorithm == 'random':
        return list(random_search.generate(space, config))[: config.max_trials]
    if config.algorithm == 'adaptive_grid':
        return list(adaptive_grid.initial(space, config))[: config.max_trials]
    raise UnsupportedFeatureError(f'{config.algorithm} requires sequential optimizer integration')


def _run_jobs(jobs, runner, config, space_hash, config_hash, store):
    trials = []
    if config.max_parallel > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=config.max_parallel) as ex:
            future_map = {ex.submit(run_one, tid, params, runner, config, space_hash, config_hash): tid for tid, params in jobs}
            iterator = [f for f in future_map] if config.ordered_results else as_completed(future_map)
            for f in iterator:
                t = f.result()
                store.save_trial(t)
                trials.append(t)
                if config.fail_fast and t.status != 'completed':
                    break
                if config.max_failed_trials is not None and sum(1 for x in trials if x.status != 'completed') >= config.max_failed_trials:
                    break
    else:
        for tid, params in jobs:
            t = run_one(tid, params, runner, config, space_hash, config_hash)
            store.save_trial(t)
            trials.append(t)
            if config.fail_fast and t.status != 'completed':
                break
            if config.max_failed_trials is not None and sum(1 for x in trials if x.status != 'completed') >= config.max_failed_trials:
                break
    return trials


def _sequential_advanced(space, runner, config, space_hash, config_hash, store, next_id):
    trials = []
    seen = set()
    if config.algorithm == 'genetic':
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
            evaluated.extend((t.params, float(t.objective_value)) for t in batch if t.status == 'completed' and t.objective_value is not None)
            if len(trials) >= config.max_trials or not evaluated:
                break
            population = genetic.next_generation(space, config, evaluated, seen, generation)
            if not population:
                break
        return trials, next_id
    if config.algorithm == 'bayesian':
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
            evaluated.extend((t.params, float(t.objective_value)) for t in batch if t.status == 'completed' and t.objective_value is not None)
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
            evaluated.extend((t.params, float(t.objective_value)) for t in batch if t.status == 'completed' and t.objective_value is not None)
        return trials, next_id
    raise UnsupportedFeatureError(f'{config.algorithm} is unsupported by optimize(); use algorithms.walk_forward.run for walk-forward')


def _analysis(config, trials, space):
    out = {}
    if config.compute_robustness or config.robustness_enabled or config.analysis_profile == 'full':
        from optimizer.analysis.robustness import analyze
        out['robustness'] = analyze(trials, space, config.robustness_neighbor_radius_steps, config.robustness_min_neighbors)
    if config.compute_sensitivity or config.compute_parameter_importance or config.analysis_profile == 'full':
        from optimizer.analysis.sensitivity import analyze
        out['sensitivity'] = analyze(trials)
    if config.compute_overfitting or config.analysis_profile == 'full':
        from optimizer.analysis.overfitting import analyze
        out['overfitting'] = analyze(trials, config.objective)
    if config.compute_profit_concentration or config.analysis_profile == 'full':
        from optimizer.analysis.profit_concentration import analyze
        out['profit_concentration'] = analyze(trials)
    if config.compute_monte_carlo or config.analysis_profile == 'full':
        from optimizer.analysis.monte_carlo import analyze
        out['monte_carlo'] = analyze(trials, config.monte_carlo_simulations, config.seed)
    return out


def optimize(parameters, runner, config: OptimizerConfig | None = None, *, cross_constraints=None):
    config = config or OptimizerConfig()
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(parameters, ParameterSpace):
        space = parameters
    else:
        space = ParameterSpace(parameters, cross_constraints if cross_constraints is not None else config.cross_constraints)
    space_hash = space.fingerprint()
    config_hash = stable_hash(config.to_dict())
    fingerprints = {'parameter_space_hash': space_hash, 'optimizer_config_hash': config_hash, 'data_fingerprint': None, 'runner_fingerprint': getattr(runner, 'fingerprint', None), 'engine_config_hash': None}
    store = _storage(config)
    check_resume(store, fingerprints, config.force_resume_on_fingerprint_mismatch)
    raw_existing = store.load_trials_raw() if config.resume else []
    done_ids = {x['id'] for x in raw_existing if x.get('status') == 'completed'}
    trials = []
    next_id = 1
    if config.baseline_params and config.run_baseline_first and 0 not in done_ids:
        t = run_one(0, config.baseline_params, runner, config, space_hash, config_hash, True, config.baseline_name)
        store.save_trial(t)
        trials.append(t)
    if config.algorithm in {'genetic', 'bayesian'}:
        advanced, next_id = _sequential_advanced(space, runner, config, space_hash, config_hash, store, next_id)
        trials.extend(advanced)
    elif config.algorithm == 'walk_forward':
        raise UnsupportedFeatureError('walk_forward uses optimizer.algorithms.walk_forward.run(parameters, runner, config, start=..., end=...)')
    else:
        params_list = _params_for(space, config)
        jobs = []
        for params in params_list:
            tid = next_id
            next_id += 1
            if tid not in done_ids:
                jobs.append((tid, params))
        trials.extend(_run_jobs(jobs, runner, config, space_hash, config_hash, store))
    if config.algorithm == 'adaptive_grid' and trials:
        jobs = []
        for p in adaptive_grid.refine(space, trials, config):
            if len(trials) + len(jobs) >= config.max_trials:
                break
            jobs.append((next_id, p))
            next_id += 1
        trials.extend(_run_jobs(jobs, runner, config, space_hash, config_hash, store))
    ranked = rank_trials(trials)
    profiles, pf = build_profiles(trials, config)
    if hasattr(store, 'save_profile'):
        for p in profiles.values():
            store.save_profile(p)
    rec, rec_name = choose_recommended(profiles, config.selection_mode)
    counts = {s: sum(1 for t in trials if t.status == s) for s in ['completed', 'failed', 'timeout', 'skipped']}
    diagnostics = [Diagnostic('ADVANCED_FEATURES_ACTIVE', 'info', 'Advanced algorithms/analysis run natively when enabled; optional reports degrade with diagnostics if dependencies/data are missing')]
    res = OptimizerResult(rec, rec_name, profiles['best_objective'].trial, ranked[: config.top_n], trials if config.save_all_trials else None, [t for t in trials if t.passed_constraints], [t for t in trials if t.status == 'completed' and not t.passed_constraints], str(getattr(store, 'path', config.output_dir)), counts, profiles, profiles['best_objective'].trial, profiles['best_passed_constraints'].trial, profiles['best_balanced'].trial, profiles['most_robust'].trial, profiles['best_profit'].trial, profiles['best_drawdown'].trial, profiles['best_profit_factor'].trial, profiles['best_sharpe'].trial, pf, diagnostics=diagnostics, analysis=_analysis(config, trials, space))
    return res
