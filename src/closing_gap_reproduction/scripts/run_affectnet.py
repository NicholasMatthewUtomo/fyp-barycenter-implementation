"""Small, reproducible AffectNet YOLO image barycenter experiment.

Download: python scripts/run_affectnet.py --download-only
Run: python scripts/run_affectnet.py --data PATH --output results/affectnet
Uses training images only; this is a numerical experiment, not a classifier.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import wb_bary_algorithms as wb

DATASET = 'fatihkgg/affectnet-yolo-format'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/affectnet')
    parser.add_argument('--class-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.data is None:
        os.environ.setdefault('KAGGLEHUB_CACHE', str(ROOT / 'data/kagglehub'))
        import kagglehub
        args.data = Path(kagglehub.dataset_download(DATASET))
    args.data = args.data.resolve()
    print(f'Dataset path: {args.data}', flush=True)
    if args.download_only:
        return
    if args.output.exists():
        parser.error('choose a new --output directory to preserve previous results')
    # Accept both train/images and images/train layouts.
    images = sorted(p for p in args.data.rglob('*')
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                    and 'train' in p.relative_to(args.data).parts)
    candidates = []
    for path in images:
        parts = list(path.relative_to(args.data).parts)
        if 'images' not in parts:
            continue
        parts[parts.index('images')] = 'labels'
        label = (args.data / Path(*parts)).with_suffix('.txt')
        if not label.exists():
            continue
        for row in label.read_text().splitlines():
            values = row.split()
            if len(values) == 5 and int(values[0]) == args.class_id:
                box = np.asarray(values[1:], dtype=float)
                if np.isfinite(box).all() and (box >= 0).all() and (box <= 1).all() and (box[2:] > 0).all():
                    candidates.append((path, label, box))
                    break
    if len(candidates) < 3:
        parser.error(f'need 3 training images for class {args.class_id}; found {len(candidates)}')
    selected = np.random.default_rng(args.seed).choice(len(candidates), 3, replace=False)
    args.output.mkdir(parents=True)
    size = 6
    axis = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(axis, axis)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    measures, records, tiles = [], [], []
    for number, index in enumerate(selected):
        path, label, (cx, cy, bw, bh) = candidates[index]
        with Image.open(path) as source:
            image = source.convert('RGB')
            width, height = image.size
            crop = (max(0, int((cx-bw/2)*width)), max(0, int((cy-bh/2)*height)),
                    min(width, int(np.ceil((cx+bw/2)*width))), min(height, int(np.ceil((cy+bh/2)*height))))
            face = image.crop(crop)
            face.save(args.output / f'input_{number}.png')
            low = face.convert('L').resize((size, size), Image.Resampling.BOX)
        mass = np.asarray(low, dtype=float).ravel()
        if mass.sum() <= 0:
            raise ValueError(f'zero intensity image: {path}')
        mass /= mass.sum()
        measures.append(wb.Measure(coords[mass > 0], mass[mass > 0]))
        low.save(args.output / f'input_{number}_6x6.png')
        tiles.append((f'Input {number+1}', low))
        records.append(dict(image=str(path.relative_to(args.data)),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            label=label.read_text(), crop_pixels=crop))
    results = []
    for t in [1, 2]:
        start = time.perf_counter()
        support, _ = wb.build_W2_candidates_subset_enumeration(measures, t)
        costs = wb._stacked_costs(support, measures, p=2)
        z, plans, objective, status = wb.solve_barycenter_lp_scipy(costs, measures)
        elapsed = time.perf_counter() - start
        residual = max(max(float(np.abs(plan.sum(axis=1)-z).max()),
                           float(np.abs(plan.sum(axis=0)-mu.w).max()))
                       for mu, plan in zip(measures, plans))
        reconstructed = sum(float(np.sum(c*p)) for c, p in zip(costs, plans))/len(measures)
        passed = (residual < 1e-7 and abs(z.sum()-1) < 1e-7 and z.min() >= -1e-9
                  and all(plan.min() >= -1e-9 for plan in plans)
                  and abs(reconstructed-objective) < 1e-8)
        if not passed:
            raise RuntimeError('transport feasibility or objective check failed')
        np.savez_compressed(args.output / f't{t}.npz', S=support, z=z,
                            **{f'plan_{i}': plan for i, plan in enumerate(plans)})
        grid = np.zeros((t*(size-1)+1,)*2)
        indices = np.rint(support*(grid.shape[0]-1)).astype(int)
        np.add.at(grid, (indices[:, 1], indices[:, 0]), z)
        display = Image.fromarray(np.rint(grid/grid.max()*255).astype('uint8'))
        display.save(args.output / f'barycenter_t{t}.png')
        tiles.append((f't={t}', display))
        results.append(dict(t=t, objective=float(objective), runtime_seconds=elapsed,
                            candidate_count=len(z), positive_atoms=int((z > 1e-9).sum()),
                            mass=float(z.sum()), max_marginal_residual=residual,
                            solver_message=status.message, passed=bool(passed)))
    if results[1]['objective'] > results[0]['objective'] + 1e-8:
        raise RuntimeError('t=2 objective unexpectedly exceeds t=1 on the shared image grid')
    canvas = Image.new('RGB', (200*len(tiles), 225), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (title, tile) in enumerate(tiles):
        canvas.paste(tile.resize((192,192), Image.Resampling.NEAREST).convert('RGB'), (200*i, 25))
        draw.text((200*i+5, 5), title, fill='black')
    canvas.save(args.output / 'comparison.png')
    report = dict(status='passed', dataset=DATASET, dataset_path=str(args.data),
                  class_id=args.class_id, seed=args.seed, eligible_training_images=len(candidates),
                  preprocessing='YOLO face crop; grayscale; BOX resize 6x6; normalized intensity masses; coordinates [0,1]^2',
                  marginal_weights='equal 1/3', candidate_rule='distinct measure subsets',
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  inputs=records, results=results,
                  limitations=['Three downsampled training images only.',
                               'No classification accuracy or full-resolution claim.',
                               'Each barycenter image is independently scaled for display.'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
