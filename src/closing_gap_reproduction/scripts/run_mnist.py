#!/usr/bin/env python3
"""Portable W2 MNIST experiment, derived from readMNIST.ipynb cells 50,69,73,76,81,84.

Uses the first k training examples of a digit, as the notebook actually does.
The digit-0, k=50, t=2 solve is retained in one-based notebook cell 50.
The default notebook-subsets candidate rule uses distinct measure indices.
The optional paper-multisets rule allows repeated measure indices as in the
paper; it is a documented alternative, not an exact historical result replay.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
from PIL import Image
import wb_bary_algorithms as wb
from mnist_to_ot import images_to_prob_vectors
from ot_scatter_to_grid import to_mnist_uint8


def build_paper_multiset_candidates(measures, t):
    """Enumerate the paper's t=1/2 means, allowing repeated measure indices.

    For t=2 this is every midpoint of the pooled input supports, including
    pairs drawn from a single measure. Use the original core's final
    12-decimal deduplication, with no earlier rounding of input coordinates.
    """
    if t not in (1, 2):
        raise ValueError("This runner supports t=1 or t=2.")
    pooled = np.unique(np.vstack([m.X for m in measures]), axis=0)
    if t == 1:
        return wb._dedup_rows(pooled)
    midpoints = (pooled[:, None, :] + pooled[None, :, :]).reshape(-1, pooled.shape[1]) / 2.0
    return wb._dedup_rows(midpoints)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--digit', type=int, choices=[0, 1, 2], required=True)
    p.add_argument('--t', type=int, choices=[1, 2], required=True)
    p.add_argument('--k', type=int, default=50, help='50 is the recovered paper setting; smaller values are demos')
    p.add_argument('--solver', choices=['gurobi', 'scipy'], default='gurobi')
    p.add_argument('--candidate-rule',choices=['notebook-subsets','paper-multisets'],
                   default='notebook-subsets',help='Recover the source notebook or allow repeated measure indices as in the paper.')
    p.add_argument('--data', type=Path, default=ROOT/'data/mnist/first50_digits012.npz')
    p.add_argument('--output', type=Path, default=ROOT/'results/mnist')
    p.add_argument('--inspect', action='store_true', help='read data and report sizes without building candidates or solving')
    args=p.parse_args()
    if not 1 <= args.k <= 50 or args.t > args.k:
        p.error('require 1 <= t <= k <= 50')
    with np.load(args.data, allow_pickle=False) as data:
        all_images, labels, training_indices = data['images'], data['labels'], data['training_indices']
        if (all_images.ndim != 3 or all_images.shape[1:] != (28, 28)
                or labels.shape != (len(all_images),)
                or training_indices.shape != labels.shape):
            p.error('data must contain images (N,28,28), labels (N,), and training_indices (N,)')
        if not np.isfinite(all_images).all() or (all_images < 0).any():
            p.error('images must contain finite nonnegative pixel values')
        selected=np.flatnonzero(labels==args.digit)[:args.k]
        if len(selected) != args.k:
            p.error(f'data contains fewer than {args.k} images of digit {args.digit}')
        images=all_images[selected]
        indices=training_indices[selected].tolist()
    weights, coords, _=images_to_prob_vectors(images)
    measures=[wb.Measure(coords[w>1e-8],w[w>1e-8]) for w in weights]
    sizes=[len(m.w) for m in measures]
    # Conservative grid-based upper bound; duplicates reduce this in practice.
    candidate_bound=(27*args.t+1)**2
    num_variables=candidate_bound*(sum(sizes)+1)
    constraints=args.k*candidate_bound+sum(sizes)+1
    dense_gib=8*constraints*num_variables/1024**3
    report=dict(experiment='MNIST',paper_figure=3,digit=args.digit,t=args.t,k=args.k,
                selection='first k occurrences in training-set order',training_indices=indices,
                positive_support_sizes=sizes,grid_domain=[-1,1],threshold=1e-8,
                marginal_weights='equal 1/k',candidate_grid_upper_bound=candidate_bound,
                candidate_rule=args.candidate_rule,
                measure_index_choice_count=(math.comb(args.k,args.t) if args.candidate_rule=='notebook-subsets'
                                             else math.comb(args.k+args.t-1,args.t)),
                dense_scipy_Aeq_upper_bound_GiB=dense_gib,
                solver=args.solver,paper_scale=args.k==50)
    if args.inspect:
        report['mode']='inspect; no solve'
        print(json.dumps(report,indent=2)); return
    if args.solver=='gurobi' and not wb._HAS_GUROBI:
        p.error('gurobipy is unavailable. Install requirements-gurobi.txt and configure your own license. Use --inspect to examine the inputs.')
    if args.solver=='scipy' and not wb._HAS_SCIPY:
        p.error('SciPy is unavailable. Install requirements.txt before using --solver scipy.')
    if args.solver=='scipy' and dense_gib>2:
        p.error(f'the original SciPy implementation uses dense constraints (upper bound {dense_gib:.1f} GiB); use Gurobi for this size or run code/smoke_test.py')
    stem=f'digit{args.digit}_k{args.k}_t{args.t}_{args.candidate_rule}'
    if args.output.exists() and not args.output.is_dir():
        p.error('--output must be a directory')
    if any((args.output/(stem+suffix)).exists() for suffix in ['.npz','.png','.json']):
        p.error('output files for this digit/k/t already exist; choose another --output directory')
    started=time.perf_counter()
    params={'OutputFlag':0} if args.solver=='gurobi' else None
    if args.candidate_rule == 'notebook-subsets':
        out=wb.algo2_W2_subset_enumeration(measures,t=args.t,prefer_solver=args.solver,
                                         params=params,seed=0,max_T=None,max_points_per_T=None)
    else:
        support=build_paper_multiset_candidates(measures,args.t)
        costs=wb._stacked_costs(support,measures,p=2)
        z, _, objective, _ = wb.solve_barycenter_lp(costs,measures,prefer=args.solver,params=params)
        out=dict(S=support,z=z,objective=objective,runtime=time.perf_counter()-started)
    report.update(mode='fresh optimization',objective=float(out['objective']),
                  runtime_seconds=float(out['runtime']),wall_seconds=time.perf_counter()-started,
                  support_size=len(out['z']),active_support_size=int((out['z']>1e-12).sum()),
                  mass_sum=float(out['z'].sum()),minimum_mass=float(out['z'].min()),
                  validation='solver completed; historical paper runtime not asserted',
                  versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','Pillow']})
    if args.solver=='gurobi': report['versions']['gurobipy']=importlib.metadata.version('gurobipy')
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/(stem+'.npz'),S=out['S'],z=out['z'],training_indices=np.array(indices))
    # Save this exact result object, avoiding the notebook's out2/out3 mix-up.
    xyw=np.column_stack([out['S'],out['z']])
    Image.fromarray(to_mnist_uint8(xyw,target_hw=(28,28),mode='bilinear',gaussian_sigma=0.8)).save(args.output/(stem+'.png'))
    (args.output/(stem+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
