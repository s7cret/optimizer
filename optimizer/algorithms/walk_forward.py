from dataclasses import replace
from pathlib import Path
from typing import cast
from typing import Any
from optimizer.config import OptimizerConfig
from optimizer.protocols import RunnerRequest


def _request_with_range(
    request: RunnerRequest,
    range_: tuple[int, int],
    tag: str,
    extra_params: dict[str, object] | None = None,
) -> RunnerRequest:
    params = (
        request.params if extra_params is None else {**request.params, **extra_params}
    )
    return replace(
        request,
        params=params,
        range=range_,
        tags={**request.tags, "walk_forward": tag},
    )


class _RunnerRequestRangeWrapper:
    def __init__(
        self,
        runner: Any,
        range_: tuple[int, int],
        tag: str,
        extra_params: dict[str, object] | None = None,
    ) -> None:
        self.runner = runner
        self.range = range_
        self.tag = tag
        self.extra_params = extra_params
        self.capabilities = getattr(runner, "capabilities", None)

    def __call__(self, request_or_params: Any) -> Any:
        if isinstance(request_or_params, RunnerRequest):
            return self.runner(
                _request_with_range(
                    request_or_params, self.range, self.tag, self.extra_params
                )
            )
        params = request_or_params
        if self.extra_params is not None:
            params = {**params, **self.extra_params}
        return self.runner(
            RunnerRequest(
                params=params,
                trial_id=0,
                required_metrics=set(),
                required_outputs=set(),
                early_stop_conditions=[],
                range=self.range,
                tags={"walk_forward": self.tag},
            )
        )


def windows(
    start: int, end: int, count: int, train_ratio: float, anchor_mode: str = "rolling"
) -> list[dict[str, tuple[int, int]]]:
    if count <= 0 or end <= start:
        return []
    width = max(1, (end - start) // count)
    train_width = max(1, int(width * train_ratio))
    out: list[dict[str, tuple[int, int]]] = []
    for i in range(count):
        w_start = start if anchor_mode == "expanding" else start + i * width
        train_end = min(end, start + i * width + train_width)
        test_end = min(end, start + (i + 1) * width)
        if train_end >= test_end:
            continue
        out.append({"train": (w_start, train_end), "test": (train_end, test_end)})
    return out


def ranged_runner(runner: Any, range_: tuple[int, int], tag: str) -> Any:
    caps = getattr(runner, "capabilities", None)
    if (
        caps
        and getattr(caps, "supports_runner_request", False)
        and getattr(caps, "supports_range", False)
    ):

        return _RunnerRequestRangeWrapper(runner, range_, tag)
    if hasattr(runner, "with_range"):
        return runner.with_range(*range_)
    raise ValueError(
        "walk-forward requires runner.capabilities.supports_range or with_range(start,end)"
    )


def _pre_bars_runner(
    runner: Any, range_: tuple[int, int], tag: str, pre_bars: int
) -> Any:
    """Create a range-aware runner that injects _effective_pre_bars."""
    caps = getattr(runner, "capabilities", None)
    if not (
        caps
        and getattr(caps, "supports_runner_request", False)
        and getattr(caps, "supports_range", False)
    ):
        raise ValueError(
            "walk-forward prehistory requires runner.capabilities.supports_runner_request "
            "and supports_range"
        )

    return _RunnerRequestRangeWrapper(
        runner, range_, tag, {"_effective_pre_bars": pre_bars}
    )


def run(
    parameters: Any, runner: Any, base_config: OptimizerConfig, *, start: int, end: int
):
    from optimizer.optimizer import optimize

    results = []
    include_pre = getattr(base_config, "walk_forward_include_prehistory", False)
    pre_bars = getattr(base_config, "walk_forward_pre_bars", None)
    supports_pre = include_pre and pre_bars is not None and pre_bars > 0

    for idx, w in enumerate(
        windows(
            start,
            end,
            base_config.walk_forward_windows,
            base_config.walk_forward_train_ratio,
            base_config.walk_forward_anchor_mode,
        ),
        start=1,
    ):
        cfg = OptimizerConfig(**base_config.to_dict())
        cfg.output_dir = Path(base_config.output_dir) / f"walk_forward_{idx}"
        cfg.algorithm = (
            "grid" if base_config.algorithm == "walk_forward" else base_config.algorithm
        )
        train_res = optimize(
            parameters, ranged_runner(runner, w["train"], "train"), cfg
        )
        best = (
            train_res.recommended_trial.params if train_res.recommended_trial else None
        )
        test_trial = None
        if best is not None:
            test_cfg = OptimizerConfig(**cfg.to_dict())
            test_cfg.output_dir = (
                Path(base_config.output_dir) / f"walk_forward_{idx}_test"
            )
            test_cfg.max_trials = 1
            if supports_pre:
                test_runner = _pre_bars_runner(
                    runner, w["test"], "test", cast(int, pre_bars)
                )
            else:
                test_runner = ranged_runner(runner, w["test"], "test")
            from optimizer.core.trial_runner import run_one

            test_trial = run_one(
                1,
                best,
                test_runner,
                test_cfg,
                "walk_forward",
                "walk_forward",
            )
        results.append(
            {
                "window": idx,
                "ranges": w,
                "train_result": train_res,
                "test_trial": test_trial,
            }
        )
    if not results:
        raise ValueError(
            "walk-forward produced no valid windows; expand the range or adjust window settings"
        )
    return {
        "status": "ok",
        "windows": results,
        "diagnostics": [],
    }
