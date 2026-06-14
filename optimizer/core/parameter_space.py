from decimal import Decimal, InvalidOperation
from itertools import product
import random
from dataclasses import asdict
from typing import Any, cast
from optimizer.core.parameter import Parameter
from optimizer.core.diagnostic import Diagnostic
from optimizer.core.expression import safe_eval_bool, stable_hash
from optimizer.errors import ParameterValidationError

SUPPORTED_PARAMETER_TYPES = {"int", "float", "bool", "string", "enum"}


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ParameterValidationError(f"{name} must be numeric")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ParameterValidationError(f"{name} must be numeric") from exc


class ParameterSpace:
    def __init__(
        self, parameters: list[Parameter], cross_constraints: list[str] | None = None
    ) -> None:
        self.parameters = parameters
        self.cross_constraints = cross_constraints or []
        self._by_name = {p.name: p for p in parameters}
        self.validate()

    def validate(self) -> None:
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ParameterValidationError("duplicate parameter names")
        for p in self.parameters:
            if p.param_type not in SUPPORTED_PARAMETER_TYPES:
                raise ParameterValidationError(
                    f"unsupported parameter type {p.param_type!r}"
                )
            if not p.name or not p.name.replace("_", "").isalnum():
                raise ParameterValidationError(f"invalid parameter name {p.name}")
            if (
                p.param_type in {"int", "float"}
                and p.enabled
                and (p.min_val is None or p.max_val is None or p.step is None)
            ):
                raise ParameterValidationError(
                    f"{p.name} requires min_val/max_val/step"
                )
            if p.param_type in {"int", "float"} and p.enabled:
                lo = _decimal(p.min_val, f"{p.name}.min_val")
                hi = _decimal(p.max_val, f"{p.name}.max_val")
                step = _decimal(p.step, f"{p.name}.step")
                if step <= 0:
                    raise ParameterValidationError(f"{p.name} step must be positive")
                if hi < lo:
                    raise ParameterValidationError(
                        f"{p.name} max_val must be >= min_val"
                    )
                default = _decimal(p.default, f"{p.name}.default")
                if p.param_type == "int" and not all(
                    value == value.to_integral_value()
                    for value in (default, lo, hi, step)
                ):
                    raise ParameterValidationError(
                        f"{p.name} int parameter bounds/default must be integral"
                    )
            if p.param_type in {"enum", "string"} and p.enabled and not p.options:
                raise ParameterValidationError(f"{p.name} requires options")
            if p.param_type == "bool" and not isinstance(p.default, bool):
                raise ParameterValidationError(f"{p.name} bool default invalid")

    def values_for(self, p: Parameter):
        if not p.enabled:
            return [p.default]
        if p.param_type == "bool":
            return [False, True]
        if p.param_type in {"enum", "string"}:
            return list(p.options or [p.default])
        vals = []
        x = _decimal(p.min_val, f"{p.name}.min_val")
        end = _decimal(p.max_val, f"{p.name}.max_val")
        step = _decimal(p.step, f"{p.name}.step")
        while x <= end:
            vals.append(int(x) if p.param_type == "int" else float(x))
            x += step
        return vals

    def grid_size(self, *, respect_constraints: bool = False) -> int:
        if not respect_constraints:
            n = 1
            for p in self.parameters:
                n *= len(self.values_for(p))
            return n
        return sum(1 for _ in self.generate_grid())

    def iter_grid_with_validity(self):
        for vals in product(*[self.values_for(p) for p in self.parameters]):
            params = {p.name: v for p, v in zip(self.parameters, vals, strict=True)}
            yield params, self.is_valid_combination(params)

    def generate_grid(self, max_combinations: int | None = None):
        count = 0
        for params, valid in self.iter_grid_with_validity():
            if valid:
                yield params
                count += 1
                if max_combinations and count >= max_combinations:
                    return

    def random_sample(self, rng: random.Random) -> dict[str, object]:
        return {p.name: rng.choice(self.values_for(p)) for p in self.parameters}

    def clamp(self, params: dict[str, object]) -> dict[str, object]:
        out = {}
        for p in self.parameters:
            v = params.get(p.name, p.default)
            if not p.enabled:
                v = p.default
            elif p.param_type in {"int", "float"}:
                # validated non-None for enabled numeric params
                lo = float(cast(Any, p.min_val))
                hi = float(cast(Any, p.max_val))
                v = max(lo, min(hi, float(cast(Any, v))))
                if p.param_type == "int":
                    step = max(1, int(round(float(cast(Any, p.step or 1)))))
                    base = int(round(lo))
                    v = base + round((int(round(v)) - base) / step) * step
                    v = int(max(lo, min(hi, v)))
            elif p.param_type == "bool":
                v = bool(v)
            elif p.options and v not in p.options:
                v = p.default
            out[p.name] = v
        return out

    def validate_params(self, params):
        ds = []
        for p in self.parameters:
            if p.name not in params:
                ds.append(Diagnostic("PARAM_MISSING", f"{p.name} missing", "error"))
                continue
            v = params[p.name]
            if p.param_type == "int" and (
                isinstance(v, bool) or not isinstance(v, int)
            ):
                ds.append(Diagnostic("PARAM_TYPE", f"{p.name} not int", "error"))
            if p.param_type == "float" and (
                isinstance(v, bool) or not isinstance(v, (int, float))
            ):
                ds.append(Diagnostic("PARAM_TYPE", f"{p.name} not float", "error"))
            if p.param_type == "bool" and not isinstance(v, bool):
                ds.append(Diagnostic("PARAM_TYPE", f"{p.name} not bool", "error"))
            if p.options and v not in p.options:
                ds.append(Diagnostic("PARAM_OPTION", f"{p.name} not allowed", "error"))
        return ds

    def is_valid_combination(self, params):
        try:
            return all(safe_eval_bool(expr, params) for expr in self.cross_constraints)
        except Exception:
            return False

    def neighbors(self, params, radius_steps: int = 1):
        out = []
        value_lists = {p.name: self.values_for(p) for p in self.parameters}
        for p in self.parameters:
            vals = value_lists[p.name]
            if params.get(p.name) not in vals:
                continue
            idx = vals.index(params[p.name])
            for j in range(
                max(0, idx - radius_steps), min(len(vals), idx + radius_steps + 1)
            ):
                if j == idx:
                    continue
                q = dict(params)
                q[p.name] = vals[j]
                if self.is_valid_combination(q):
                    out.append(q)
        return out

    def refine_around(
        self,
        center_params,
        refinement_factor: float,
        min_step: dict[str, float] | None = None,
    ):
        candidates = [dict(center_params)]
        min_step = min_step or {}
        for p in self.parameters:
            if p.param_type not in {"int", "float"} or not p.enabled:
                continue
            step = max(
                float(p.step or 0) * refinement_factor, float(min_step.get(p.name, 0))
            )
            for delta in (-step, step):
                q = dict(center_params)
                q[p.name] = q[p.name] + delta
                q = self.clamp(q)
                if self.is_valid_combination(q):
                    candidates.append(q)
        return candidates

    def fingerprint(self) -> str:
        return stable_hash(
            {
                "parameters": [asdict(p) for p in self.parameters],
                "cross_constraints": self.cross_constraints,
            }
        )
