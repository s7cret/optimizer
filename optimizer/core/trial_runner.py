import time, traceback, concurrent.futures
from optimizer.protocols import RunnerRequest
from optimizer.core.metric_extractor import MetricExtractor
from optimizer.core.metric_registry import MetricRegistry
from optimizer.core.constraints import evaluate_constraints
from optimizer.core.objective import compute_objective, objective_direction
from optimizer.core.normalization import balanced_score
from optimizer.core.expression import stable_hash
from optimizer.results.trial import Trial
from optimizer.core.diagnostic import Diagnostic
from optimizer.version import __version__

def now_ms(): return int(time.time()*1000)

def run_one(trial_id, params, runner, config, space_hash, config_hash, is_baseline=False, baseline_name=None):
    started=now_ms(); t0=time.perf_counter(); diags=[]; metrics={}; raw=None
    direction=objective_direction(config.objective, config.objective_direction)
    try:
        caps=getattr(runner,'capabilities',None)
        required={config.objective, *(config.constraints or {}).keys()}
        if config.objective_secondary: required.add(config.objective_secondary)
        required = MetricRegistry().required_metrics(required)
        outputs=MetricRegistry().required_outputs(required)
        if caps and getattr(caps,'supports_runner_request',False):
            req=RunnerRequest(params=params, trial_id=trial_id, required_metrics=required, required_outputs=outputs, early_stop_conditions=config.early_stop_conditions if config.early_stop_enabled else [], seed=config.seed if getattr(caps,'supports_seed',False) else None)
            raw=runner(req)
        else:
            if outputs or config.early_stop_enabled: diags.append(Diagnostic('BASIC_RUNNER_CONTRACT_USED','warning','runner called with params only; advanced hints unavailable', trial_id=trial_id))
            raw=runner(params)
        metrics=MetricExtractor().extract(raw)
        if 'return_drawdown_ratio' not in metrics and metrics.get('net_profit') is not None and metrics.get('max_drawdown'):
            metrics['return_drawdown_ratio']=metrics['net_profit']/abs(metrics['max_drawdown'])
        obj=compute_objective(metrics, config.objective, direction, config.objective_expression)
        violations=evaluate_constraints(metrics, config.constraints)
        bs=balanced_score(metrics)
        r=getattr(raw,'to_dict',lambda: raw if isinstance(raw,dict) else None)()
        trial=Trial(trial_id, dict(params), metrics, obj, direction, None, not violations, violations, len(violations), bs, metrics.get('robustness_score'), metrics.get('overfitting_score'), metrics.get('profit_concentration_score'), r if config.save_backtest_result else None, time.perf_counter()-t0, 'completed', False, is_baseline, baseline_name, getattr(raw,'content_hash',None) if raw is not None else None, getattr(raw,'data_fingerprint',None) if raw is not None else None, getattr(raw,'runner_fingerprint',None) if raw is not None else None, getattr(raw,'engine_config_hash',None) if raw is not None else None, space_hash, config_hash, __version__, diagnostics=diags, started_at=started, finished_at=now_ms())
        return trial
    except concurrent.futures.TimeoutError:
        status='timeout'; err='trial timeout'; tr=traceback.format_exc()
    except Exception as e:
        status='failed'; err=str(e); tr=traceback.format_exc()
    return Trial(trial_id, dict(params), metrics, None, direction, None, False, {}, 0, None, None, None, None, None, time.perf_counter()-t0, status, is_baseline=is_baseline, baseline_name=baseline_name, parameter_space_hash=space_hash, optimizer_config_hash=config_hash, code_version=__version__, error_message=err, traceback=tr, diagnostics=diags, started_at=started, finished_at=now_ms())
