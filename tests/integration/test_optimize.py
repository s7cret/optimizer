from optimizer import OptimizerConfig, Parameter, optimize, RunnerCapabilities

def test_optimize_grid_json(tmp_path):
    def runner(p): return {'net_profit': p['x'], 'max_drawdown_percent': 10-p['x'], 'profit_factor': 1+p['x'], 'sharpe_ratio': p['x']}
    cfg=OptimizerConfig(max_trials=3, output_dir=tmp_path, storage_backend='json', constraints={'max_drawdown_percent': {'max': 9}})
    res=optimize([Parameter('x','int',1,1,3,1)], runner, cfg)
    assert res.recommended_trial.params['x']==3
    assert (tmp_path/'trials.jsonl').exists()

def test_advanced_runner_request(tmp_path):
    class R:
        capabilities=RunnerCapabilities(supports_runner_request=True,supports_seed=True)
        def __call__(self, req):
            assert req.seed==42 and 'net_profit' in req.required_metrics
            return {'net_profit': req.params['x'], 'max_drawdown_percent': 1}
    res=optimize([Parameter('x','int',1,1,1,1)], R(), OptimizerConfig(output_dir=tmp_path, storage_backend='json'))
    assert res.best_trial.id==1
