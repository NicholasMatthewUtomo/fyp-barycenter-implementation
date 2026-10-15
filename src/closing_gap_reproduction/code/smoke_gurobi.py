#!/usr/bin/env python3
"""Tiny analytic checks for persistent column generation; requires a Gurobi license."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import gurobipy as gp
import wb_bary_algorithms as wb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--json',type=Path)
    args=p.parse_args()
    measures=[wb.Measure([[0.0]],[1.0]),wb.Measure([[2.0]],[1.0])]
    rows=[]
    for t,expected in [(1,2.0),(2,1.0)]:
        S,_=wb.build_W2_candidates_subset_enumeration(measures,t)
        costs=wb._stacked_costs(S,measures,2)
        for limit in [None,5.0]:
            z,Y,obj,model=wb.solve_barycenter_lp_cg_persistent(costs,measures,params={'OutputFlag':0},time_limit_sec=limit)
            assert np.isclose(obj,expected,atol=1e-9)
            assert np.isclose(z.sum(),1,atol=1e-9) and (z>=-1e-9).all()
            for plan,mu in zip(Y,measures):
                assert (plan>=-1e-9).all()
                assert np.allclose(plan.sum(1),z,atol=1e-9)
                assert np.allclose(plan.sum(0),mu.w,atol=1e-9)
            assert np.isclose(obj,sum(float((c*y).sum()) for c,y in zip(costs,Y))/2,atol=1e-9)
            rows.append(dict(case='two_diracs',t=t,time_limit_sec=limit,objective=float(obj),passed=True))
            model.dispose()
    S=np.array([[0.],[1.],[2.]])
    costs=wb._stacked_costs(S,measures,2)
    z,Y,obj,model=wb.solve_barycenter_lp_cg_persistent(costs,measures,params={'OutputFlag':0},init_L=[0],time_limit_sec=5)
    assert np.isclose(obj,1.0) and np.allclose(z,[0,1,0])
    rows.append(dict(case='add_missing_midpoint',objective=float(obj),passed=True))
    model.dispose()
    start=time.perf_counter()
    try:
        wb.solve_barycenter_lp_cg_persistent(costs,measures,params={'OutputFlag':0},init_L=[0],time_limit_sec=0)
    except RuntimeError as exc:
        assert str(exc)=='No feasible solution found before time limit.'
        assert time.perf_counter()-start<5
    else:
        raise AssertionError('Zero budget should report no feasible incumbent')
    rows.append(dict(case='exhausted_budget',passed=True))
    report=dict(status='passed',gurobi='.'.join(map(str,gp.gurobi.version())),checks=rows,
                scope='tiny analytical problems; not paper-scale reproduction')
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
