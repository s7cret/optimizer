from pathlib import Path
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
from optimizer.algorithms import grid_search, random_search, adaptive_grid
from optimizer.core.diagnostic import Diagnostic
from optimizer.errors import UnsupportedFeatureError

def _storage(config): return JsonStorage(config.output_dir) if config.storage_backend=='json' else SQLiteStorage(config.output_dir)

def _params_for(space, config):
    if config.algorithm=='grid': return list(grid_search.generate(space, config))[:config.max_trials]
    if config.algorithm=='random': return list(random_search.generate(space, config))[:config.max_trials]
    if config.algorithm=='adaptive_grid': return list(adaptive_grid.initial(space, config))[:config.max_trials]
    raise UnsupportedFeatureError(f'{config.algorithm} is full-scope placeholder in this release')

def optimize(parameters, runner, config:OptimizerConfig|None=None, *, cross_constraints=None):
    config=config or OptimizerConfig(); config.output_dir=Path(config.output_dir); config.output_dir.mkdir(parents=True,exist_ok=True)
    if isinstance(parameters, ParameterSpace): space=parameters
    else: space=ParameterSpace(parameters, cross_constraints if cross_constraints is not None else config.cross_constraints)
    space_hash=space.fingerprint(); config_hash=stable_hash(config.to_dict())
    fingerprints={'parameter_space_hash':space_hash,'optimizer_config_hash':config_hash,'data_fingerprint':None,'runner_fingerprint':getattr(runner,'fingerprint',None),'engine_config_hash':None}
    store=_storage(config); check_resume(store, fingerprints, config.force_resume_on_fingerprint_mismatch)
    raw_existing=store.load_trials_raw() if config.resume else []
    done_ids={x['id'] for x in raw_existing if x.get('status')=='completed'}
    trials=[]; next_id=1
    if config.baseline_params and config.run_baseline_first and 0 not in done_ids:
        t=run_one(0, config.baseline_params, runner, config, space_hash, config_hash, True, config.baseline_name); store.save_trial(t); trials.append(t)
    params_list=_params_for(space,config)
    for params in params_list:
        tid=next_id; next_id+=1
        if tid in done_ids: continue
        t=run_one(tid, params, runner, config, space_hash, config_hash); store.save_trial(t); trials.append(t)
        if config.fail_fast and t.status!='completed': break
        if config.max_failed_trials is not None and sum(1 for x in trials if x.status!='completed')>=config.max_failed_trials: break
    if config.algorithm=='adaptive_grid' and trials:
        for p in adaptive_grid.refine(space,trials,config):
            if len(trials)>=config.max_trials: break
            t=run_one(next_id, p, runner, config, space_hash, config_hash); next_id+=1; store.save_trial(t); trials.append(t)
    ranked=rank_trials(trials)
    profiles, pf=build_profiles(trials, config)
    if hasattr(store,'save_profile'):
        for p in profiles.values(): store.save_profile(p)
    rec, rec_name=choose_recommended(profiles, config.selection_mode)
    counts={s:sum(1 for t in trials if t.status==s) for s in ['completed','failed','timeout','skipped']}
    res=OptimizerResult(rec, rec_name, profiles['best_objective'].trial, ranked[:config.top_n], trials if config.save_all_trials else None, [t for t in trials if t.passed_constraints], [t for t in trials if t.status=='completed' and not t.passed_constraints], str(getattr(store,'path',config.output_dir)), counts, profiles, profiles['best_objective'].trial, profiles['best_passed_constraints'].trial, profiles['best_balanced'].trial, profiles['most_robust'].trial, profiles['best_profit'].trial, profiles['best_drawdown'].trial, profiles['best_profit_factor'].trial, profiles['best_sharpe'].trial, pf, diagnostics=[Diagnostic('ADVANCED_PLACEHOLDERS','info','bayesian/genetic/walk-forward/robustness/profit concentration are diagnostic placeholders where not MVP-complete')])
    return res
