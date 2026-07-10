# Architecture

Optimizer is a coordinator, not a backtest runtime. Its core responsibilities are:

1. build parameter candidates;
2. validate parameter and metric constraints;
3. call a supplied runner through `RunnerRequest`/`RunnerResponse`;
4. score and rank completed trials;
5. persist enough metadata for deterministic resume and audit;
6. produce reports and analysis sidecars.

Concrete market-data, Pine runtime, broker-emulation, and strategy-execution concerns stay outside this package. Those systems connect through the runner protocol rather than concrete imports in the optimizer core.

## Main packages

```text
optimizer.algorithms     candidate generation
optimizer.core           expression, metrics, constraints, trial execution
optimizer.results        trial/run/profile models
optimizer.selection      ranking and recommendation policies
optimizer.storage        JSON/SQLite persistence
optimizer.reporting      output formats
optimizer.analysis       optional post-run analysis
optimizer.runners        optional adapter boundaries
optimizer.cli            command-line interface
```

## Public runner contract

Preferred contract id:

```text
pine.optimizer_runner.v1
```

The legacy typo `pain.optimizer_runner.v1` is accepted only for migration compatibility. New integrations should emit the `pine.*` contract.

Runner response hashes preserve content, data, runner, runtime, and engine-config lineage when supplied by the runner adapter. Trial records expose those fingerprints for resume checks, audit trails, and downstream OpenPine reporting.


## CLI boundary

The CLI is intentionally small and fail-closed. It supports only `run`, `resume`,
`dry-run`, `analyze`, and `export`. Previously advertised unsupported commands were
removed so `--help` reflects real standalone functionality. UI dashboards, plotting
services, champion-review workflows, and cross-repository comparisons are OpenPine
application concerns rather than optimizer-core responsibilities.

## Release hygiene model

`optimizer.distribution` builds deterministic source archives from an allow-listed
source tree and excludes bytecode, caches, local result directories, build outputs,
existing archives, VCS metadata, and egg-info metadata. The manifest reports release
artifacts such as `dist/`, `build/`, `optimizer_results/`, and nested source archives as
hygiene failures while tolerating VCS metadata, generated egg-info, and transient
interpreter caches because those entries are never included in the deterministic archive.

## Parameter validation

`ParameterSpace` validates parameter kinds, numeric bounds, integer integrality, bool
defaults, and grid step direction before candidate generation. This keeps invalid
optimization spaces out of algorithms and prevents bool values from silently passing
as ints/floats through Python's subclass relationship.
