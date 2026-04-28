from typing import Any


def _trades_from(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        trades = raw.get("closed_trades") or raw.get("trades") or raw.get("closedTrades") or []
        return list(trades) if isinstance(trades, list) else []
    trades = getattr(raw, "closed_trades", None) or getattr(raw, "trades", None) or []
    return list(trades) if isinstance(trades, list) else []


def _profit(trade: Any) -> float:
    if isinstance(trade, dict):
        for key in ("profit", "pnl", "net_profit"):
            if trade.get(key) is not None:
                return float(trade[key])
    return float(getattr(trade, "profit", 0.0) or getattr(trade, "pnl", 0.0) or 0.0)


def analyze_trial(raw: Any) -> dict[str, object]:
    profits = [_profit(t) for t in _trades_from(raw)]
    positive = sorted([p for p in profits if p > 0], reverse=True)
    total = sum(positive)
    if not positive or total <= 0:
        return {
            "status": "insufficient_data",
            "trade_count": len(profits),
            "top_10_percent_share": None,
            "score": None,
        }
    n = max(1, int(len(positive) * 0.1 + 0.999999))
    share = sum(positive[:n]) / total
    return {
        "status": "ok",
        "trade_count": len(profits),
        "top_10_percent_share": share,
        "score": max(0.0, min(1.0, 1.0 - share)),
    }


def analyze(trials: list[Any]) -> dict[str, object]:
    per: dict[int, dict[str, object]] = {}
    for t in trials:
        raw = t.backtest_result or t.metrics
        item = analyze_trial(raw)
        if item["status"] != "ok" and t.metrics.get("profit_concentration_score") is not None:
            item = {
                "status": "ok",
                "trade_count": None,
                "top_10_percent_share": None,
                "score": t.metrics.get("profit_concentration_score"),
            }
        per[t.id] = item
    ok = [v["score"] for v in per.values() if v.get("score") is not None]
    return {
        "status": "ok" if ok else "insufficient_data",
        "by_trial_id": per,
        "diagnostics": []
        if ok
        else [
            {
                "code": "PROFIT_CONCENTRATION_REQUIRES_TRADES",
                "severity": "warning",
                "message": "Expected closed_trades/trades with profit or pnl",
            }
        ],
    }
