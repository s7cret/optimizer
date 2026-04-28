from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Diagnostic:
    """Structured optimizer diagnostic.

    Positional order is part of the public contract: code, message, severity,
    trial_id, params_hash, metric, context.
    """

    code: str
    message: str
    severity: Literal['info', 'warning', 'error'] = 'warning'
    trial_id: int | None = None
    params_hash: str | None = None
    metric: str | None = None
    context: dict[str, object] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
