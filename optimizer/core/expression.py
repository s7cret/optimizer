import ast
import hashlib
import json
import math
from typing import Any, Literal

from optimizer.errors import SafeExpressionError

_MAX_POW_ABS_EXPONENT = 12
_MAX_ABS_RESULT = 1e308
_ALLOWED_BIN = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow}
_ALLOWED_CMP = {ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq}


def _finite(v: Any) -> Any:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        if not math.isfinite(f) or abs(f) > _MAX_ABS_RESULT:
            raise SafeExpressionError("expression produced NaN/Inf/out-of-range value")
    return v


def _apply_bin(op: ast.operator, a: Any, b: Any) -> Any:
    if isinstance(op, ast.Div) and b == 0:
        raise SafeExpressionError("division by zero")
    if isinstance(op, ast.Mod) and b == 0:
        raise SafeExpressionError("modulo by zero")
    if isinstance(op, ast.Pow):
        if abs(float(b)) > _MAX_POW_ABS_EXPONENT:
            raise SafeExpressionError("power exponent too large")
        return _finite(a**b)
    if isinstance(op, ast.Add):
        return _finite(a + b)
    if isinstance(op, ast.Sub):
        return _finite(a - b)
    if isinstance(op, ast.Mult):
        return _finite(a * b)
    if isinstance(op, ast.Div):
        return _finite(a / b)
    if isinstance(op, ast.Mod):
        return _finite(a % b)
    raise SafeExpressionError(f"unsafe expression op {type(op).__name__}")


def safe_eval(
    expr: str, names: dict[str, Any], *, mode: Literal["numeric", "boolean", "any"] = "any"
) -> Any:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return _finite(n.value)
        if isinstance(n, ast.Name):
            if n.id in names:
                return _finite(names[n.id])
            if n.id in {"true", "True"}:
                return True
            if n.id in {"false", "False"}:
                return False
            raise SafeExpressionError(f"unknown name {n.id}")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.Not, ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return (
                (not bool(v))
                if isinstance(n.op, ast.Not)
                else _finite(-v if isinstance(n.op, ast.USub) else +v)
            )
        if isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
            vals = [bool(ev(x)) for x in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BIN:
            return _apply_bin(n.op, ev(n.left), ev(n.right))
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            result = True
            for op, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                if isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                elif isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                else:
                    raise SafeExpressionError(f"unsafe comparator {type(op).__name__}")
                result = result and ok
                left = right
            return result
        raise SafeExpressionError(f"unsafe expression node {type(n).__name__}")

    value = ev(ast.parse(expr, mode="eval"))
    if mode == "numeric" and isinstance(value, bool):
        raise SafeExpressionError("numeric expression returned boolean")
    if mode == "boolean" and not isinstance(value, bool):
        raise SafeExpressionError("constraint expression returned non-boolean")
    return _finite(value)


def safe_eval_numeric(expr: str, names: dict[str, Any]) -> float:
    v = safe_eval(expr, names, mode="numeric")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise SafeExpressionError("numeric expression returned non-number")
    if not math.isfinite(f):
        raise SafeExpressionError("numeric expression returned NaN/Inf")
    return f


def safe_eval_bool(expr: str, names: dict[str, Any]) -> bool:
    return bool(safe_eval(expr, names, mode="boolean"))


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
