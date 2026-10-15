#!/usr/bin/env python3
"""Execute one original package runner with recorded laptop thread limits."""
import argparse
from pathlib import Path
import runpy
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--threads',type=int,default=4)
p.add_argument('--solver',action='store_true')
p.add_argument('script',type=Path)
p.add_argument('args',nargs=argparse.REMAINDER)
a=p.parse_args()
if a.threads < 1: p.error('--threads must be positive')
if a.solver:
    import gurobipy as gp
    gp.setParam('Threads',a.threads)
sys.path.insert(0,str(a.script.resolve().parent))
sys.argv=[str(a.script)]+a.args
runpy.run_path(str(a.script),run_name='__main__')
