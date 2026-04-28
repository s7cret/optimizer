def evaluate_constraints(metrics:dict[str,float|None], constraints:dict[str,dict[str,float]]):
    violations={}
    for name, rules in (constraints or {}).items():
        v=metrics.get(name)
        if v is None: violations[name]='missing'; continue
        if 'min' in rules and v < rules['min']: violations[name]=f'{v} < min {rules["min"]}'
        if 'max' in rules and v > rules['max']: violations[name]=f'{v} > max {rules["max"]}'
    return violations
