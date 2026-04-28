from dataclasses import asdict, dataclass, field
from typing import Literal

@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Literal['info','warning','error'] = 'warning'
    message: str = ''
    context: dict[str, object] = field(default_factory=dict)
    trial_id: int | None = None
    source: str | None = None
    def to_dict(self): return asdict(self)
