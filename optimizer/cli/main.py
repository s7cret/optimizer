import argparse, importlib.util, json
from pathlib import Path
from optimizer import OptimizerConfig, Parameter, optimize
from optimizer.reporting.json_report import to_json
from optimizer.reporting.csv_report import write_csv
from optimizer.reporting.markdown_report import to_markdown

def load_obj(spec):
    file, name = spec.split(':',1); modname='optimizer_user_'+Path(file).stem
    sp=importlib.util.spec_from_file_location(modname,file); mod=importlib.util.module_from_spec(sp); sp.loader.exec_module(mod); return getattr(mod,name)

def load_params(path):
    data=json.loads(Path(path).read_text()); return [Parameter(**p) for p in data.get('parameters',data)]

def main(argv=None):
    ap=argparse.ArgumentParser(prog='optimizer')
    sub=ap.add_subparsers(dest='cmd',required=True)
    run=sub.add_parser('run'); run.add_argument('--params',required=True); run.add_argument('--runner',required=True); run.add_argument('--output-dir',default='./optimizer_results'); run.add_argument('--algorithm',default='grid'); run.add_argument('--objective',default='net_profit'); run.add_argument('--max-trials',type=int,default=1000)
    sub.add_parser('analyze').add_argument('--result-dir',default='./optimizer_results')
    exp=sub.add_parser('export'); exp.add_argument('--result-dir',default='./optimizer_results'); exp.add_argument('--format',choices=['json','csv','markdown'],default='json')
    sub.add_parser('resume')
    sub.add_parser('walk-forward')
    sub.add_parser('champions')
    sub.add_parser('diff')
    sub.add_parser('plot')
    sub.add_parser('baseline')
    ns=ap.parse_args(argv)
    if ns.cmd=='run':
        cfg=OptimizerConfig(output_dir=Path(ns.output_dir), algorithm=ns.algorithm, objective=ns.objective, max_trials=ns.max_trials)
        res=optimize(load_params(ns.params), load_obj(ns.runner), cfg)
        to_json(res, Path(ns.output_dir)/'result.json'); write_csv(res.all_trials or [], Path(ns.output_dir)/'trials.csv'); to_markdown(res, Path(ns.output_dir)/'report.md')
        print(to_markdown(res))
    else:
        print(f'{ns.cmd}: placeholder/diagnostic command available; full implementation depends on stored result context')
if __name__=='__main__': main()
