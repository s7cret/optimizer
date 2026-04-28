import concurrent.futures
import time
import traceback

from optimizer.protocols import RunnerRequest
from optimizer.core.metric_extractor import MetricExtractor
from optimizer.core.metric_registry import MetricRegistry
from optimizer.core.constraints import evaluate_constraints, merge_constraints
from optimizer.core.objective import compute_objective, objective_direction
from optimizer.core.normalization import balanced_score
from optimizer.core.expression import stable_hash
from optimizer.results.trial import Trial
from optimizer.core.diagnostic import Diagnostic
from optimizer.version import __version__


def now_ms(): return int(time.time() * 1000)


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
    required = {config.objective, *merge_constraints(config).keys()}
    required |= reg.extract_expression_metrics(config.objective_expression)
    if config.objective_secondary: required.add(config.objective_secondary)
    if config.report_profiles:
        required |= reg.profile_required_metrics(['best_profit', 'best_drawdown', 'best_profit_factor', 'best_sharpe', 'best_balanced'])
    return reg.get_required_metrics(required)


def run_one(trial_id, params, runner, config, space_hash, config_hash, is_baseline=False, baseline_name=None):
    started = now_ms(); t0 = time.perf_counter(); diags = []; metrics = {}; raw = None
    params_hash = stable_hash(params)
    direction = objective_direction(config.objective, config.objective_direction)
    constraints = merge_constraints(config)
    try:
        caps = getattr(runner, 'capabilities', None)
        required = _required_metrics(config)
        outputs = MetricRegistry().get_required_outputs(required)
        if caps and getattr(caps, 'supports_runner_request', False):
            req = RunnerRequest(params=params, trial_id=trial_id, required_metrics=required, required_outputs=outputs, early_stop_conditions=config.early_stop_conditions if config.early_stop_enabled else [], seed=config.seed if getattr(caps, 'supports_seed', False) else None)
            raw = _call_with_timeout(lambda: runner(req), config.timeout_per_trial_sec)
        else:
            if outputs or config.early_stop_enabled:
                diags.append(Diagnostic('BASIC_RUNNER_CONTRACT_USED', 'runner called with params only; advanced hints unavailable', 'warning', trial_id, params_hash))
            raw = _call_with_timeout(lambda: runner(params), config.timeout_per_trial_sec)
        metrics = MetricExtractor().extract(raw)
        if 'return_drawdown_ratio' not in metrics and metrics.get('net_profit') is not None and metrics.get('max_drawdown'):
            metrics['return_drawdown_ratio'] = metrics['net_profit'] / abs(metrics['max_drawdown'])
        obj = compute_objective(metrics, config.objective, direction, config.objective_expression)
        c = evaluate_constraints(metrics, constraints, trial_id=trial_id, params_hash=params_hash)
        diags.extend(c.diagnostics)
        effective_obj = obj
        if config.constraint_mode in {'penalty', 'both'} and c.penalty and effective_obj is not None:
            mult = float(config.constraint_penalty_multiplier)
            effective_obj = effective_obj - c.penalty * mult if direction == 'maximize' else effective_obj + c.penalty * mult
        bs = balanced_score(metrics)
        r = getattr(raw, 'to_dict', lambda: raw if isinstance(raw, dict) else None)()
        trial = Trial(trial_id, dict(params), metrics, effective_obj, direction, None, c.hard_passed if config.constraint_mode in {'filter', 'both'} else True, {**c.violations, **c.soft_violations}, len(c.violations) + len(c.soft_violations), bs, metrics.get('robustness_score'), metrics.get('overfitting_score'), metrics.get('profit_concentration_score'), r if config.save_backtest_result else None, time.perf_counter() - t0, 'completed', False, is_baseline, baseline_name, getattr(raw, 'content_hash', None) if raw is not None else None, getattr(raw, 'data_fingerprint', None) if raw is not None else None, getattr(raw, 'runner_fingerprint', None) if raw is not None else None, getattr(raw, 'engine_config_hash', None) if raw is not None else None, space_hash, config_hash, __version__, None, None, diags, set(required) - set(metrics), started, now_ms(), params_hash, obj)
        return trial
    except concurrent.futures.TimeoutError:
        status = 'timeout'; err = 'trial timeout'; tr = traceback.format_exc()
        diags.append(Diagnostic('TRIAL_TIMEOUT', err, 'error', trial_id, params_hash))
    except Exception as e:
        status = 'failed'; err = str(e); tr = traceback.format_exc()
        diags.append(Diagnostic('TRIAL_FAILED', err, 'error', trial_id, params_hash))
    return Trial(trial_id, dict(params), metrics, None, direction, None, False, {}, 0, None, None, None, None, None, time.perf_counter() - t0, status, is_baseline=is_baseline, baseline_name=baseline_name, parameter_space_hash=space_hash, optimizer_config_hash=config_hash, code_version=__version__, error_message=err, traceback=tr, diagnostics=diags, missing_metrics=set(), started_at=started, finished_at=now_ms(), params_hash=params_hash)
