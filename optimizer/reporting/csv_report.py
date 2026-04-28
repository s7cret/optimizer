import csv
def write_csv(trials, path):
    keys=sorted({k for t in trials for k in t.metrics.keys()})
    with open(path,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['id','status','objective_value','passed_constraints',*keys])
        for t in trials: w.writerow([t.id,t.status,t.objective_value,t.passed_constraints,*[t.metrics.get(k) for k in keys]])
