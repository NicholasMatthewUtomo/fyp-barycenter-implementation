#!/usr/bin/env python3
"""Check the real sign image pipeline without training or classification."""
import argparse
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--json',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent.parent/'closing_gap_reproduction'
sys.path.insert(0,str(root/'scripts'))
import run_sign
run_sign.load_runtime()
checks=[]
for folder in ['SignAlphaSubset','Sign_C','Sign_D']:
    for image in sorted((root/'data/sign'/folder).glob('*.jpg')):
        mu=run_sign.image_to_measure(str(image),**run_sign.preprocess_options())
        assert mu.X.shape[1]==5 and abs(float(mu.w.sum())-1)<1e-8
        rgb=run_sign.rasterize_barycenter(mu.X,mu.w,size=(64,64),coordinate_origin='xy',normalized_coords=True,color_space='rgb')
        assert rgb.shape==(64,64,3)
        checks.append(dict(image=image.name,atoms=len(mu.w),mass=float(mu.w.sum())))
assert len(checks)==9
report=dict(status='passed',scope='nine real training images; no optimization',checks=checks)
a.json.parent.mkdir(parents=True,exist_ok=True)
a.json.write_text(json.dumps(report,indent=2)+'\n')
print('PASS: all nine training images load, normalize and render.')
