"""Durable TrialKey reservation helpers shared by optimizer execution paths."""

import time
import traceback
from typing import Literal, cast

from optimizer.core.diagnostic import Diagnostic
from optimizer.core.expression import stable_hash
from optimizer.core.objective import objective_direction
from optimizer.core.trial_key import (
    TrialIdentity,
    contract_schema_hashes,
    validate_critical_identity_hashes,
)
from optimizer.results.trial import Trial


def _direction(config) -> Literal["maximize", "minimize"]:
    value = (
        config.objective_direction
        if config.objective_expression and config.objective_direction != "auto"
        else (
            "maximize"
            if config.objective_expression
            else objective_direction(config.objective, config.objective_direction)
        )
    )
    return cast(Literal["maximize", "minimize"], value)


def pending_trial(tid, params, config, space_hash, config_hash):
    if not validate_critical_identity_hashes(config):
        raise ValueError("durable trial identity is not configured")
    direction = _direction(config)
    constraints = tuple(
        {
            "name": name,
            "operator": str(operator).upper(),
            "value": str(value),
        }
        for name, rules in sorted(config.constraints.items())
        for operator, value in sorted(rules.items())
        if not isinstance(value, bool)
    )
    identity = TrialIdentity(
        generated_artifact_hash=config.generated_artifact_hash or "",
        data_snapshot_series_hash=(
            config.data_snapshot_series_hash or config.data_fingerprint or ""
        ),
        parameters=params,
        engine_build_hash=config.engine_build_hash or "",
        engine_config_hash=config.engine_config_hash or "",
        semantic_profile=config.semantic_profile,
        finality_policy=config.finality_policy,
        warmup_policy=config.warmup_policy,
        score_policy=config.score_policy,
        end_policy=config.end_policy,
        contract_schema_hashes=contract_schema_hashes(),
        stack_manifest_hash=config.stack_manifest_hash or "",
        deterministic_seed=config.seed,
        fold_identity=config.fold_identity,
        walk_forward_identity=config.walk_forward_identity,
        objective_version=config.objective_version,
        constraints_version=config.constraints_version,
        producer_commit=config.optimizer_commit or "",
        optimizer_id=config.optimizer_id,
        strategy_id=config.strategy_id,
        source_hash=config.source_hash or config.generated_artifact_hash or "",
        emitted_module_hash=config.emitted_module_hash
        or config.generated_artifact_hash
        or "",
        numeric_policy=config.numeric_policy,
        fill_policy=config.fill_policy,
        objective_metric=config.objective,
        objective_direction=direction.upper(),
        constraints=constraints,
    ).seal()
    return Trial.pending(
        tid,
        dict(params),
        trial_key=identity.trial_key,
        identity_payload=identity.payload,
        params_hash=stable_hash(params),
        objective_direction=direction,
        parameter_space_hash=space_hash,
        optimizer_config_hash=config_hash,
        constraints_snapshot=dict(config.constraints),
    )


def failed_worker_trial(tid, params, exc, config, space_hash, config_hash):
    direction = _direction(config)
    params_hash = stable_hash(params)
    diagnostic = Diagnostic(
        "OPTIMIZER_WORKER_EXCEPTION",
        f"{exc.__class__.__name__}: {exc}",
        "error",
        tid,
        params_hash,
    )
    now = int(time.time() * 1000)
    return Trial(
        tid,
        dict(params),
        {},
        None,
        direction,
        None,
        False,
        {},
        0,
        None,
        None,
        None,
        None,
        None,
        0.0,
        "failed",
        parameter_space_hash=space_hash,
        optimizer_config_hash=config_hash,
        error_message=str(exc),
        traceback=traceback.format_exc(),
        diagnostics=[diagnostic],
        started_at=now,
        finished_at=now,
        params_hash=params_hash,
    )


def bind_trial_identity(trial, pending):
    trial.trial_key = pending.trial_key
    trial.identity_payload = pending.identity_payload
    trial.constraints_snapshot = pending.constraints_snapshot
    if trial.status == "completed":
        trial.lifecycle = "completed"
    elif any(d.code == "TRIAL_TIMEOUT" for d in trial.diagnostics):
        trial.lifecycle = "timeout"
    else:
        trial.lifecycle = "failed"
    return trial


__all__ = ["bind_trial_identity", "failed_worker_trial", "pending_trial"]
