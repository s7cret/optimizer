from dataclasses import dataclass
from typing import Any, Literal
@dataclass(frozen=True)
class Parameter:
    name: str
    param_type: Literal['int','float','bool','string','enum']
    default: Any
    min_val: Any | None = None
    max_val: Any | None = None
    step: Any | None = None
    options: list[Any] | None = None
    enabled: bool = True
    group: str | None = None
    description: str | None = None
