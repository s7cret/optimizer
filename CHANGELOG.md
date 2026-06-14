# Changelog

## 4.0.0

- Promoted Optimizer to the OpenPine 4.0 runner-contract line.
- Renamed the preferred runner contract to `pine.optimizer_runner.v1`; legacy `pain.optimizer_runner.v1` remains accepted only for migration compatibility.
- Added dependency-free release, distribution, quality, import-smoke, and wheel-smoke gates.
- Added `python -m optimizer` and a console-script CLI.
- Refactored the optimizer entrypoint into a thin compatibility façade backed by `optimizer.engine`.
- Stabilized process timeout execution and covered process-timeout failure branches.
- Tightened runner-response validation: structured `metrics` and `hashes` must be dictionaries when present.
- Preserved content, data, runner, runtime, and engine-config fingerprints on trial records where supplied by the runner adapter.
- Normalized safe-expression overflow from power operations into `SafeExpressionError`.
- Added hermetic, self-contained BacktestEngine adapter contract tests; full-stack BacktestEngine/PineLib/OpenPine smoke remains an external integration gate.
- Raised the standalone package coverage gate to 100% and refreshed README, release, architecture, and development documentation.
- Removed unsupported CLI commands from public help and added clean user-facing CLI validation for parameter files and `FILE:OBJECT` runners.
- Tightened `ParameterSpace` validation for unsupported kinds, bool-as-number ambiguity, non-integral int bounds/defaults, invalid numeric defaults, and inverted numeric ranges.
- Hardened deterministic distribution manifests to report real release artifacts while keeping source archives cache-free and reproducible.

## 2.17.0

- Previous public baseline before OpenPine 4.0 hardening.
