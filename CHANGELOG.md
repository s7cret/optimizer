# Changelog

## 5.0.0rc6

- Advances optimizer package and trial-identity producer metadata to RC.6.
- Pins the RC.6 Contracts catalog and records generated-artifact V3 in RC.6 trial schema sets without parsing generated artifacts.

## 5.0.0rc5

- Advances optimizer package and trial-identity producer metadata to RC.5.
- Pins the RC.5 Contracts catalog without changing search or storage semantics.

## 5.0.0rc4

- Separates standalone `openpine.trial.identity.v1` from trial lifecycle envelopes.
- Rejects partial durable identities before reservation, storage, or runner side effects.
- Revalidates trial identity schema and root seals on reserve and load in both storage backends.
- Pins the immutable OpenPine Contracts RC.4 candidate.

## 4.0.2

- Refreshed package and release evidence for the coordinated OpenPine 4.0.2 stack.
- Preserved optimizer runner and search contracts without behavioral changes.

## 4.0.1

- Published the hardened OpenPine 4.0.1 stack with unchanged optimizer runner contracts.
- Aligned package, documentation, and immutable consumer metadata.
- Made process timeouts contain detached descendants with delegated cgroup v2, retained a portable process-group fallback, and documented that runner lifecycle containment is not a security sandbox.

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
