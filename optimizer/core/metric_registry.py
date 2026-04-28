from dataclasses import dataclass, field
from typing import Literal
MetricSource = Literal['runner','optimizer_derived','optimizer_analysis','external']
@dataclass(frozen=True)
class MetricSpec:
    name: str; source: MetricSource; direction: Literal['maximize','minimize','neutral']='neutral'
    required_metrics: set[str]=field(default_factory=set); required_outputs: set[str]=field(default_factory=set)
    required_statistics_profile: Literal['minimal','standard','full']='minimal'
    requires_completed_trials: bool=False; requires_trade_level_data: bool=False; can_be_constraint: bool=True; can_be_objective: bool=True
class MetricRegistry:
    METRICS={
      'net_profit':MetricSpec('net_profit','runner','maximize',required_outputs={'summary_metrics'}),
      'net_profit_percent':MetricSpec('net_profit_percent','runner','maximize',required_outputs={'summary_metrics'}),
      'profit_factor':MetricSpec('profit_factor','runner','maximize',required_outputs={'summary_metrics'}),
      'sharpe_ratio':MetricSpec('sharpe_ratio','runner','maximize',required_outputs={'summary_metrics'}),
      'sortino_ratio':MetricSpec('sortino_ratio','runner','maximize',required_outputs={'summary_metrics'}),
      'max_drawdown':MetricSpec('max_drawdown','runner','minimize',required_outputs={'summary_metrics'}),
      'max_drawdown_percent':MetricSpec('max_drawdown_percent','runner','minimize',required_outputs={'summary_metrics'}),
      'return_drawdown_ratio':MetricSpec('return_drawdown_ratio','optimizer_derived','maximize',required_metrics={'net_profit','max_drawdown'},required_outputs={'summary_metrics'}),
      'expectancy':MetricSpec('expectancy','runner','maximize',required_outputs={'summary_metrics'}),
      'robustness_score':MetricSpec('robustness_score','optimizer_analysis','maximize',requires_completed_trials=True,can_be_objective=False),
      'overfitting_score':MetricSpec('overfitting_score','optimizer_analysis','minimize',requires_completed_trials=True,can_be_objective=False),
      'profit_concentration_score':MetricSpec('profit_concentration_score','optimizer_analysis','minimize',required_outputs={'closed_trades'},requires_trade_level_data=True,requires_completed_trials=True,can_be_objective=False),
    }
    def get(self,name): return self.METRICS.get(name, MetricSpec(name,'runner','neutral'))
    def direction(self,name): return self.get(name).direction
    def required_metrics(self, names):
        req=set(names)
        for n in list(req): req |= self.get(n).required_metrics
        return req
    def required_outputs(self, names):
        outs=set()
        for n in self.required_metrics(names): outs |= self.get(n).required_outputs
        return outs
