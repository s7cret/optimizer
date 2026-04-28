from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Literal

MetricSource = Literal['runner', 'optimizer_derived', 'optimizer_analysis', 'external']


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: MetricSource
    direction: Literal['maximize', 'minimize', 'neutral'] = 'neutral'
    required_metrics: set[str] = field(default_factory=set)
    required_outputs: set[str] = field(default_factory=set)
    required_statistics_profile: Literal['minimal', 'standard', 'full'] = 'minimal'
    requires_completed_trials: bool = False
    requires_trade_level_data: bool = False
    can_be_constraint: bool = True
    can_be_objective: bool = True


class _NameVisitor(ast.NodeVisitor):
    def __init__(self): self.names: set[str] = set()
    def visit_Name(self, node):
        if node.id not in {'true', 'false', 'True', 'False'}:
            self.names.add(node.id)


class MetricRegistry:
    METRICS = {
        'net_profit': MetricSpec('net_profit', 'runner', 'maximize', required_outputs={'summary_metrics'}),
        'net_profit_percent': MetricSpec('net_profit_percent', 'runner', 'maximize', required_outputs={'summary_metrics'}),
        'profit_factor': MetricSpec('profit_factor', 'runner', 'maximize', required_outputs={'summary_metrics'}),
        'sharpe_ratio': MetricSpec('sharpe_ratio', 'runner', 'maximize', required_outputs={'summary_metrics'}),
        'sortino_ratio': MetricSpec('sortino_ratio', 'runner', 'maximize', required_outputs={'summary_metrics'}),
        'max_drawdown': MetricSpec('max_drawdown', 'runner', 'minimize', required_outputs={'summary_metrics'}),
        'max_drawdown_percent': MetricSpec('max_drawdown_percent', 'runner', 'minimize', required_outputs={'summary_metrics'}),
        'return_drawdown_ratio': MetricSpec('return_drawdown_ratio', 'optimizer_derived', 'maximize', required_metrics={'net_profit', 'max_drawdown'}, required_outputs={'summary_metrics'}),
        'expectancy': MetricSpec('expectancy', 'runner', 'maximize', required_outputs={'summary_metrics'}),
        'robustness_score': MetricSpec('robustness_score', 'optimizer_analysis', 'maximize', requires_completed_trials=True, can_be_objective=False),
        'overfitting_score': MetricSpec('overfitting_score', 'optimizer_analysis', 'minimize', requires_completed_trials=True, can_be_objective=False),
        'profit_concentration_score': MetricSpec('profit_concentration_score', 'optimizer_analysis', 'minimize', required_outputs={'closed_trades'}, requires_trade_level_data=True, requires_completed_trials=True, can_be_objective=False),
    }
    PROFILE_METRICS = {
        'best_profit': {'net_profit'},
        'best_drawdown': {'max_drawdown_percent'},
        'best_profit_factor': {'profit_factor'},
        'best_sharpe': {'sharpe_ratio'},
        'best_balanced': {'net_profit', 'max_drawdown_percent', 'profit_factor', 'sharpe_ratio'},
        'most_robust': {'robustness_score'},
        'pareto_front': {'net_profit', 'max_drawdown_percent'},
        'pareto_knee': {'net_profit', 'max_drawdown_percent'},
    }

    def get(self, name): return self.METRICS.get(name, MetricSpec(name, 'runner', 'neutral'))
    def direction(self, name): return self.get(name).direction

    def extract_expression_metrics(self, expression: str | None) -> set[str]:
        if not expression: return set()
        visitor = _NameVisitor(); visitor.visit(ast.parse(expression, mode='eval')); return visitor.names

    def get_required_metrics(self, names) -> set[str]:
        req = set(names or [])
        changed = True
        while changed:
            changed = False
            for n in list(req):
                before = len(req); req |= self.get(n).required_metrics; changed |= len(req) != before
        return req

    def required_metrics(self, names): return self.get_required_metrics(names)

    def get_required_runner_metrics(self, names) -> set[str]:
        return {n for n in self.get_required_metrics(names) if self.get(n).source in {'runner', 'external'}}

    def get_required_outputs(self, names) -> set[str]:
        outs = set()
        for n in self.get_required_metrics(names): outs |= self.get(n).required_outputs
        return outs

    def required_outputs(self, names): return self.get_required_outputs(names)

    def get_required_statistics_profile(self, names) -> str:
        order = {'minimal': 0, 'standard': 1, 'full': 2}
        best = 'minimal'
        for n in self.get_required_metrics(names):
            prof = self.get(n).required_statistics_profile
            if order[prof] > order[best]: best = prof
        return best

    def profile_required_metrics(self, profiles) -> set[str]:
        req = set()
        for p in profiles or []: req |= self.PROFILE_METRICS.get(p, set())
        return self.get_required_metrics(req)
