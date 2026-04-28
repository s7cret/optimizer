from dataclasses import dataclass
from optimizer.core.metric_extractor import MetricExtractor

@dataclass
class R:
    net_profit: int
    max_drawdown: int

def test_metric_extractor_dataclass():
    assert MetricExtractor().extract(R(10,-2))['max_drawdown']==2.0
