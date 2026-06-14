from pathlib import Path

from optimizer.core.diagnostic import Diagnostic
from optimizer.results.result import OptimizerRunResult


def _is_proven(value):
    return value is True or (isinstance(value, str) and value.lower() == "proven")


def _lookup_nested(payload, *names):
    if not isinstance(payload, dict):
        return None
    for name in names:
        if name in payload:
            return payload[name]
    gates = (
        payload.get("oracle_gates")
        or payload.get("oracleGates")
        or payload.get("gates")
    )
    if isinstance(gates, dict):
        for name in names:
            if name in gates:
                return gates[name]
    return None


def _data_query_risk_reasons(payload):
    reasons = set()

    def walk(value, key=""):
        key_l = str(key).lower()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item, key)
            return
        text = str(value).lower() if isinstance(value, str) else ""
        truthy = value is True or (
            isinstance(value, str) and text in {"true", "yes", "1"}
        )
        if key_l in {"realtime", "live", "use_realtime", "allow_realtime"} and truthy:
            reasons.add("realtime")
        if (
            key_l in {"intrabar", "use_intrabar", "bar_magnifier", "use_bar_magnifier"}
            and truthy
        ):
            reasons.add("intrabar")
        if key_l in {"tick", "ticks", "tick_data", "du_tick_data"} and truthy:
            reasons.add("tick")
        if key_l in {
            "mode",
            "kind",
            "type",
            "data_type",
            "source",
            "feed",
        } and text in {
            "realtime",
            "live",
            "tick",
            "ticks",
            "intrabar",
            "du_tick",
        }:
            reasons.add(text)
        if key_l in {
            "lower_timeframe",
            "lower_tf",
            "intrabar_timeframe",
        } and value not in {
            None,
            "",
            False,
        }:
            reasons.add("intrabar")

    walk(payload)
    return reasons


def _validate_data_query(data_query):
    reasons = _data_query_risk_reasons(data_query)
    if not reasons:
        return None
    required = {
        "tvRealtimeBoundary": _lookup_nested(
            data_query,
            "tvRealtimeBoundary",
            "tv_realtime_boundary",
            "final_tick_commit",
        ),
        "duTickCompleteness": _lookup_nested(
            data_query,
            "duTickCompleteness",
            "du_tick_completeness",
            "tick_completeness",
        ),
        "intrabarOrderFill": _lookup_nested(
            data_query,
            "intrabarOrderFill",
            "intrabar_order_fill",
            "intrabar_fill_oracle",
        ),
    }
    missing = [name for name, value in required.items() if not _is_proven(value)]
    if not missing:
        return None
    return Diagnostic(
        "UNPROVEN_REALTIME_INTRABAR_DATA_QUERY",
        "realtime/tick/intrabar optimizer inputs require proven oracle gates",
        "error",
        context={
            "risk_reasons": sorted(reasons),
            "missing_or_unproven_gates": missing,
        },
    )


def _failed_request_result(request, diagnostic, output_dir):
    return OptimizerRunResult(
        None,
        None,
        None,
        [],
        [],
        [],
        [],
        str(output_dir),
        {"completed": 0, "failed": 0},
        diagnostics=[diagnostic],
        run_id=request.run_id,
        status="failed",
        trials=(),
        artifact_path=Path(output_dir),
        data_query=request.data_query,
    )


__all__ = ["_validate_data_query", "_failed_request_result"]
