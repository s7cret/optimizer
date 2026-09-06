# OP-26 / OP-27: runner and result integrity

A failed, cancelled, incomplete or error-bearing result must not become a completed
profitable trial. This applies both to basic dictionaries and typed responses with
nested raw results. Existing engine diagnostic codes are retained. Required outputs
are checked for both the canonical and supported historical response contract;
string/number truthiness is not accepted as an availability flag.

Non-finite metric values are omitted by ordinary extraction and rejected by custom
extractors or objective calculation. A missing/non-finite primary objective fails
the trial. An optional infinite metric such as an undefined ratio does not poison
unrelated finite objectives. Actual zero profit, profit factor and drawdown are not
replaced by fallback values. Ranking ignores incomplete/non-finite scores and clears
previous ranks. This is a correctness boundary, not winning-trial replay acceptance.

The BacktestEngine runner now copies captured bars and nested static parameters,
and gives each call private copies of bars, parameters and a strategy instance
(when supplied instead of a class). The engine_factory must still construct a fresh
engine and avoid mutable external/global state; the adapter cannot prove arbitrary
user code is stateless. Pure source identity does not prove deterministic execution.

`_effective_pre_bars` in trial params overrides the static value, including zero.
It is passed only as the engine's named control, not a Pine parameter. Unsupported
warmup and unsupported range/seed/early-stop/output hints fail explicitly. Signature
inspection checks compatibility but never silently discards a meaningful control.
Object and dictionary engine results expose the same status/errors/outputs/hashes.

For durable production runners use `strict_identity=True` and explicitly supply
nonzero SHA256 runner_fingerprint, data_fingerprint and engine_config_hash. No source
or closure inference occurs for explicit identities. These are caller attestations,
not verification of wheel contents; the production host must bind them to admitted
artifacts, data and configuration. Development inference remains available, but
opaque/cyclic captured objects require explicit identities instead of collapsing to
a Python type. Nested bytecode constants are structural data, not repr addresses.
The versioned TrialIdentity contract remains the authority for durable cache keys.

No sandbox or timeout containment rule was relaxed. Full serial/parallel/seed,
process failure and winning-trial UI replay acceptance remains separate work.
