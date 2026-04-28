import argparse, importlib.util, json, sys
from pathlib import Path

from optimizer import OptimizerConfig, Parameter, optimize
from optimizer.reporting.csv_report import write_csv
from optimizer.reporting.json_report import to_json
from optimizer.reporting.markdown_report import to_markdown
from optimizer.storage.json_backend import JsonStorage
from optimizer.storage.sqlite_backend import SQLiteStorage


def load_obj(spec):
    file, name = spec.split(':', 1); modname = 'optimizer_user_' + Path(file).stem
    sp = importlib.util.spec_from_file_location(modname, file); mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod); return getattr(mod, name)


def load_params(path):
    data = json.loads(Path(path).read_text()); return [Parameter(**p) for p in data.get('parameters', data)]


def _storage(result_dir):
    d = Path(result_dir)
    if (d / 'optimizer.sqlite').exists(): return SQLiteStorage(d)
    return JsonStorage(d)


def _load_trials(result_dir):
    return _storage(result_dir).load_trials_raw()


def _cmd_export(result_dir, fmt):
    rows = _load_trials(result_dir)
    out = Path(result_dir) / f'trials_export.{"md" if fmt == "markdown" else fmt}'
    if fmt == 'json':
        out.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str))
    elif fmt == 'csv':
        keys = sorted({k for r in rows for k in r})
        import csv
        with out.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    else:
        lines = ['# Optimizer trials export', '', f'Trials: {len(rows)}', '']
        for r in rows[:50]: lines.append(f"- id={r.get('id')} status={r.get('status')} objective={r.get('objective_value')} params={r.get('params')}")
        out.write_text('\n'.join(lines) + '\n')
    print(out)


def _cmd_analyze(result_dir):
    rows = _load_trials(result_dir)
    counts = {}
    for r in rows: counts[r.get('status', 'unknown')] = counts.get(r.get('status', 'unknown'), 0) + 1
    best = sorted([r for r in rows if r.get('objective_value') is not None], key=lambda r: r['objective_value'], reverse=True)[:5]
    print(json.dumps({'trials': len(rows), 'counts': counts, 'top_objective_rows': best}, indent=2, sort_keys=True, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='optimizer')
    sub = ap.add_subparsers(dest='cmd', required=True)
    run = sub.add_parser('run'); run.add_argument('--params', required=True); run.add_argument('--runner', required=True); run.add_argument('--output-dir', default='./optimizer_results'); run.add_argument('--algorithm', default='grid'); run.add_argument('--objective', default='net_profit'); run.add_argument('--max-trials', type=int, default=1000)
    analyze = sub.add_parser('analyze'); analyze.add_argument('--result-dir', default='./optimizer_results')
    exp = sub.add_parser('export'); exp.add_argument('--result-dir', default='./optimizer_results'); exp.add_argument('--format', choices=['json', 'csv', 'markdown'], default='json')
    resume = sub.add_parser('resume'); resume.add_argument('--params', required=True); resume.add_argument('--runner', required=True); resume.add_argument('--output-dir', default='./optimizer_results'); resume.add_argument('--algorithm', default='grid'); resume.add_argument('--objective', default='net_profit'); resume.add_argument('--max-trials', type=int, default=1000)
    for name in ['walk-forward', 'champions', 'diff', 'plot', 'baseline']:
        p = sub.add_parser(name); p.add_argument('--result-dir', default='./optimizer_results')
    ns = ap.parse_args(argv)
    if ns.cmd in {'run', 'resume'}:
        cfg = OptimizerConfig(output_dir=Path(ns.output_dir), algorithm=ns.algorithm, objective=ns.objective, max_trials=ns.max_trials, resume=True)
        res = optimize(load_params(ns.params), load_obj(ns.runner), cfg)
        to_json(res, Path(ns.output_dir) / 'result.json'); write_csv(res.all_trials or [], Path(ns.output_dir) / 'trials.csv'); to_markdown(res, Path(ns.output_dir) / 'report.md')
        print(to_markdown(res))
    elif ns.cmd == 'analyze':
        _cmd_analyze(ns.result_dir)
    elif ns.cmd == 'export':
        _cmd_export(ns.result_dir, ns.format)
    else:
        print(f'{ns.cmd}: no standalone CLI implementation in this release; use the documented Python API/reporting module for this operation.', file=sys.stderr)
        return 2


if __name__ == '__main__': sys.exit(main())
