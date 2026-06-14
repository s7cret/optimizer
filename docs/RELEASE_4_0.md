# Release 4.0.0

Release status target:

- package version: `4.0.0`;
- preferred runner contract: `pine.optimizer_runner.v1`;
- legacy `pain.optimizer_runner.v1` accepted only for migration compatibility;
- no hard dependency on sibling OpenPine repositories;
- architecture budget: no Python module above 700 lines;
- coverage gate: 100% standalone package coverage;
- deterministic source archive built by `python -m optimizer.distribution build-zip`;
- strict runner-response validation for structured `metrics` and `hashes`;
- runner lineage preserves content, data, runner, runtime, and engine-config fingerprints when supplied;
- safe-expression overflow is reported as `SafeExpressionError`, not raw Python overflow;
- standalone CLI help lists only implemented commands;
- parameter validation rejects unsupported parameter types, non-integral int bounds/defaults, bool-as-number inputs, invalid numeric defaults, and inverted ranges;
- deterministic distribution manifests report release artifacts without failing merely because local test runs produced interpreter caches that are excluded from archives.

Run before pushing:

```bash
bash scripts/release_gate.sh
bash scripts/wheel_smoke.sh
python -m optimizer.distribution build-zip --root . --output optimizer-4.0.0.zip
```

Optional external smoke after publishing sibling repositories:

```text
pine2ast -> ast2python -> pinelib/backtest-engine -> optimizer
```
