from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal

from optimizer.core.diagnostic import Diagnostic
from optimizer.core.metric_registry import MetricRegistry

ConstraintStage = Literal['pre', 'post', 'selection']


@dataclass
class ConstraintEvaluation:
    passed: bool
    hard_passed: bool
    violations: dict[str, str]
    soft_violations: dict[str, str] = field(default_factory=dict)
    penalty: float = 0.0
    diagnostics: list[Diagnostic] = field(default_factory=list)


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if isfinite(f) else None


def auto_constraints_for_profile(profile: str) -> dict[str, dict[str, float | str | bool]]:
    if profile == 'conservative':
        return {'max_drawdown_percent': {'max': 25.0, 'hard': False}, 'profit_factor': {'min': 1.05, 'hard': False}}
    if profile == 'aggressive':
        return {'net_profit': {'min': 0.0, 'hard': False}}
    if profile in {'balanced', 'robust', 'best_after_constraints'}:
        return {'net_profit': {'min': 0.0, 'hard': False}, 'profit_factor': {'min': 1.0, 'hard': False}}
    return {}


def merge_constraints(config) -> dict[str, dict[str, Any]]:
    custom = {k: dict(v) for k, v in (config.constraints or {}).items()}
    if not getattr(config, 'use_profile_auto_constraints', True) or config.constraints_merge_mode == 'custom_only':
        return custom
    auto = auto_constraints_for_profile(getattr(config, 'selection_mode', 'best_after_constraints'))
    if config.constraints_merge_mode == 'merge_auto_and_custom':
        out = {k: dict(v) for k, v in auto.items()}
        for metric, rules in custom.items():
            out.setdefault(metric, {}).update(rules)
        return out
    # custom_overrides_auto: auto constraints not mentioned by custom are kept.
    out = {k: dict(v) for k, v in auto.items()}
    out.update(custom)
    return out


def evaluate_constraints(
    metrics: dict[str, float | None],
    constraints: dict[str, dict[str, Any]] | None,
    *,
    trial_id: int | None = None,
    params_hash: str | None = None,
    stage: ConstraintStage = 'post',
) -> ConstraintEvaluation:
    violations: dict[str, str] = {}
    soft: dict[str, str] = {}
    diagnostics: list[Diagnostic] = []
    penalty = 0.0
    for name, rules in (constraints or {}).items():
        rule_stage = rules.get('stage')
        if rule_stage and rule_stage != stage:
            continue
        v = _num(metrics.get(name))
        hard = bool(rules.get('hard', True))
        failed: list[str] = []
        if v is None:
            failed.append('missing')
        else:
            if 'min' in rules and v < float(rules['min']):
                failed.append(f'{v} < min {rules["min"]}')
                penalty += (float(rules['min']) - v) * float(rules.get('penalty', 1.0))
            if 'max' in rules and v > float(rules['max']):
                failed.append(f'{v} > max {rules["max"]}')
                penalty += (v - float(rules['max'])) * float(rules.get('penalty', 1.0))
            if 'eq' in rules and v != float(rules['eq']):
                failed.append(f'{v} != {rules["eq"]}')
                penalty += abs(v - float(rules['eq'])) * float(rules.get('penalty', 1.0))
            if 'neq' in rules and v == float(rules['neq']):
                failed.append(f'{v} == forbidden {rules["neq"]}')
                penalty += float(rules.get('penalty', 1.0))
        if failed:
            msg = '; '.join(failed)
            target = violations if hard else soft
            target[name] = msg
            diagnostics.append(Diagnostic('CONSTRAINT_VIOLATION', msg, 'warning' if hard else 'info', trial_id, params_hash, name, {'hard': hard, 'stage': stage, 'rules': dict(rules)}))
    mode = 'both'
    hard_passed = not violations
    return ConstraintEvaluation(hard_passed and not soft, hard_passed, violations, soft, penalty, diagnostics)


def required_constraint_metrics(config) -> set[str]:
    return MetricRegistry().required_metrics(set(merge_constraints(config)))
