from pathlib import Path
from typing import Any
from optimizer.config import OptimizerConfig
from optimizer.core.diagnostic import Diagnostic
from optimizer.protocols import RunnerRequest


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

        def call(params: dict[str, object]) -> Any:
            return runner(
                RunnerRequest(
                    params=params,
                    trial_id=0,
                    required_metrics=set(),
                    required_outputs=set(),
                    early_stop_conditions=[],
                    range=range_,
                    tags={"walk_forward": tag},
                )
            )

        return call
    if hasattr(runner, "with_range"):
        return runner.with_range(*range_)
    raise ValueError(
        "walk-forward requires runner.capabilities.supports_range or with_range(start,end)"
    )


def _pre_bars_runner(runner: Any, pre_bars: int) -> Any:
    """Wrap runner so _effective_pre_bars is injected into RunnerRequest params.
    
    D5-F: For walk-forward test folds with pre-bars, the runner receives
    _effective_pre_bars in params so it knows how many bars are pre-bars.
    """
    caps = getattr(runner, "capabilities", None)
    supports_req = caps and getattr(caps, "supports_runner_request", False)
    
    if supports_req:
        # Wrap __call__ to inject _effective_pre_bars into params
        original_call = runner.__call__
        
        class PreBarsRunner:
            capabilities = runner.capabilities
            
            def __call__(self, request: RunnerRequest) -> Any:
                enriched_params = {**request.params, "_effective_pre_bars": pre_bars}
                enriched = RunnerRequest(
                    params=enriched_params,
                    trial_id=request.trial_id,
                    required_metrics=request.required_metrics,
                    required_outputs=request.required_outputs,
                    early_stop_conditions=request.early_stop_conditions,
                    seed=request.seed,
                    fingerprints=request.fingerprints,
                    range=request.range,
                    tags=request.tags,
                )
                return original_call(enriched)
            
            def with_range(self, start: int, end: int) -> Any:
                return self
        
        return PreBarsRunner()
    
    # Fallback: runner doesn't support RunnerRequest, return as-is
    return runner


def run(parameters: Any, runner: Any, base_config: OptimizerConfig, *, start: int, end: int):
    from optimizer.optimizer import optimize

    results = []
    diags = []
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
        cfg.algorithm = "grid" if base_config.algorithm == "walk_forward" else base_config.algorithm
        train_res = optimize(parameters, ranged_runner(runner, w["train"], "train"), cfg)
        best = train_res.recommended_trial.params if train_res.recommended_trial else None
        test_trial = None
        if best is not None:
            test_cfg = OptimizerConfig(**cfg.to_dict())
            test_cfg.output_dir = Path(base_config.output_dir) / f"walk_forward_{idx}_test"
            test_cfg.max_trials = 1
            # D5-F: use pre-bars runner if enabled
            if supports_pre:
                test_runner = _pre_bars_runner(runner, pre_bars)
            else:
                test_runner = ranged_runner(runner, w["test"], "test")
            # Avoid empty ParameterSpace: call runner directly via optimize-compatible one-trial baseline.
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
            {"window": idx, "ranges": w, "train_result": train_res, "test_trial": test_trial}
        )
    if not results:
        diags.append(
            Diagnostic(
                "WALK_FORWARD_NO_WINDOWS", "No valid walk-forward windows were produced", "warning"
            )
        )
    return {
        "status": "ok" if results else "insufficient_data",
        "windows": results,
        "diagnostics": diags,
    }
