import ast, hashlib, json, math, random, time, traceback as tb
from dataclasses import asdict, dataclass, field, is_dataclass
from decimal import Decimal
from itertools import product
from typing import Any, Callable, Literal
from optimizer.errors import ParameterValidationError, SafeExpressionError
from optimizer.core.diagnostic import Diagnostic

_ALLOWED_BIN={ast.Add:lambda a,b:a+b, ast.Sub:lambda a,b:a-b, ast.Mult:lambda a,b:a*b, ast.Div:lambda a,b:a/b, ast.Mod:lambda a,b:a%b, ast.Pow:lambda a,b:a**b}
_ALLOWED_CMP={ast.Lt:lambda a,b:a<b, ast.LtE:lambda a,b:a<=b, ast.Gt:lambda a,b:a>b, ast.GtE:lambda a,b:a>=b, ast.Eq:lambda a,b:a==b, ast.NotEq:lambda a,b:a!=b}
_ALLOWED_BOOL={ast.And:all, ast.Or:any}
def safe_eval(expr:str, names:dict[str,Any])->Any:
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.Name):
            if n.id in names: return names[n.id]
            if n.id in {'true','True'}: return True
            if n.id in {'false','False'}: return False
            raise SafeExpressionError(f'unknown name {n.id}')
        if isinstance(n, ast.UnaryOp) and isinstance(n.op,(ast.Not,ast.USub,ast.UAdd)):
            v=ev(n.operand); return (not v) if isinstance(n.op,ast.Not) else (-v if isinstance(n.op,ast.USub) else +v)
        if isinstance(n, ast.BoolOp) and type(n.op) in _ALLOWED_BOOL: return _ALLOWED_BOOL[type(n.op)]([bool(ev(x)) for x in n.values])
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BIN: return _ALLOWED_BIN[type(n.op)](ev(n.left),ev(n.right))
        if isinstance(n, ast.Compare):
            left=ev(n.left)
            for op, comp in zip(n.ops,n.comparators):
                right=ev(comp)
                if type(op) not in _ALLOWED_CMP or not _ALLOWED_CMP[type(op)](left,right): return False
                left=right
            return True
        raise SafeExpressionError(f'unsafe expression node {type(n).__name__}')
    return ev(ast.parse(expr, mode='eval'))

def stable_hash(obj:Any)->str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(',',':')).encode()).hexdigest()
