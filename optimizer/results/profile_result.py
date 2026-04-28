from dataclasses import dataclass, field, asdict
from optimizer.core.diagnostic import Diagnostic
from optimizer.results.trial import Trial
@dataclass
class ResultProfile:
    name:str; trial:Trial|None; reason:str; score_name:str; score_value:float|None; warnings:list[Diagnostic]=field(default_factory=list); diagnostics:list[Diagnostic]=field(default_factory=list)
    def to_dict(self): return asdict(self)
