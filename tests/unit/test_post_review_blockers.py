import json
import subprocess
import sys

from optimizer import OptimizerConfig, Parameter, RunnerCapabilities, optimize


def test_non_best_objective_recommendation_never_violates_hard_constraints(tmp_path):
    def runner(p):
        return {'net_profit': p['x'], 'max_drawdown_percent': 99, 'profit_factor': 1.0, 'sharpe_ratio': 1.0}

    cfg = OptimizerConfig(
        output_dir=tmp_path,
        storage_backend='json',
        selection_mode='balanced',
        max_trials=2,
        use_profile_auto_constraints=False,
        constraints={'max_drawdown_percent': {'max': 1, 'hard': True}},
    )
    res = optimize([Parameter('x', 'int', 1, 1, 2, 1)], runner, cfg)

    assert res.recommended_profile == 'best_balanced'
    assert res.recommended_trial is None
    assert all(not t.passed_constraints for t in res.all_trials)
    assert any(d.code == 'NO_TRIALS_PASSED_CONSTRAINTS' for d in res.diagnostics)


def test_best_objective_mode_can_recommend_best_even_if_constraints_fail(tmp_path):
    cfg = OptimizerConfig(
        output_dir=tmp_path,
        storage_backend='json',
        selection_mode='best_objective',
        max_trials=1,
        use_profile_auto_constraints=False,
        constraints={'max_drawdown_percent': {'max': 1, 'hard': True}},
    )
    res = optimize(
        [Parameter('x', 'int', 1, 1, 1, 1)],
        lambda p: {'net_profit': 10, 'max_drawdown_percent': 99, 'profit_factor': 1.0, 'sharpe_ratio': 1.0},
        cfg,
    )
    assert res.recommended_trial is not None
    assert res.recommended_trial.passed_constraints is False


def test_adaptive_grid_refinement_uses_minimize_direction(tmp_path):
    seen = []

    def runner(p):
        seen.append(p['x'])
        return {'max_drawdown_percent': p['x'], 'net_profit': 100 - p['x'], 'profit_factor': 1.1, 'sharpe_ratio': 1.0}

    cfg = OptimizerConfig(
        algorithm='adaptive_grid',
        output_dir=tmp_path,
        storage_backend='json',
        objective='max_drawdown_percent',
        objective_direction='auto',
        max_trials=4,
        grid_max_combinations=3,
        adaptive_grid_top_n=1,
        adaptive_grid_refinement_factor=0.5,
        use_profile_auto_constraints=False,
    )
    optimize([Parameter('x', 'int', 1, 1, 3, 1)], runner, cfg)

    assert seen[:3] == [1, 2, 3]
    assert seen[3] == 1


def test_random_invalid_cross_constraint_combos_are_persisted_as_skipped(tmp_path):
    cfg = OptimizerConfig(
        algorithm='random',
        output_dir=tmp_path,
        storage_backend='json',
        max_trials=6,
        random_trials=6,
        seed=0,
        use_profile_auto_constraints=False,
    )
    res = optimize(
        [Parameter('x', 'int', 1, 1, 3, 1), Parameter('y', 'int', 1, 1, 3, 1)],
        lambda p: {'net_profit': p['x'] + p['y'], 'max_drawdown_percent': 1, 'profit_factor': 1.1, 'sharpe_ratio': 1.0},
        cfg,
        cross_constraints=['x == y'],
    )

    skipped = [t for t in res.all_trials if t.status == 'skipped']
    assert skipped
    assert all(any(d.code == 'INVALID_PARAM_COMBINATION' for d in t.diagnostics) for t in skipped)
    rows = [json.loads(line) for line in (tmp_path / 'trials.jsonl').read_text().splitlines()]
    assert any(r['status'] == 'skipped' for r in rows)


def test_required_output_capability_gap_fails_before_runner_call(tmp_path):
    class R:
        capabilities = RunnerCapabilities(
            supports_runner_request=True,
            supports_required_outputs=True,
            supported_outputs=set(),
        )

        def __call__(self, req):  # pragma: no cover - must not be called
            raise AssertionError('runner should not be called')

    res = optimize(
        [Parameter('x', 'int', 1, 1, 1, 1)],
        R(),
        OptimizerConfig(output_dir=tmp_path, storage_backend='json', use_profile_auto_constraints=False),
    )

    assert res.trials_count_by_status['failed'] == 1
    assert any(d.code == 'RUNNER_REQUIRED_OUTPUT_UNSUPPORTED' for t in res.all_trials for d in t.diagnostics)


def test_runner_fingerprint_callable_is_invoked_and_persisted(tmp_path):
    class R:
        def fingerprint(self):
            return 'runner-v1'

        def __call__(self, params):
            return {'net_profit': params['x'], 'max_drawdown_percent': 1, 'profit_factor': 1.1, 'sharpe_ratio': 1.0}

    optimize(
        [Parameter('x', 'int', 1, 1, 1, 1)],
        R(),
        OptimizerConfig(output_dir=tmp_path, storage_backend='json', use_profile_auto_constraints=False),
    )

    meta = json.loads((tmp_path / 'run.json').read_text())
    assert meta['runner_fingerprint'] == 'runner-v1'


def test_placeholder_cli_commands_exit_nonzero(tmp_path):
    proc = subprocess.run(
        [sys.executable, '-m', 'optimizer.cli.main', 'plot', '--result-dir', str(tmp_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 2
    assert 'no standalone CLI implementation' in proc.stderr
