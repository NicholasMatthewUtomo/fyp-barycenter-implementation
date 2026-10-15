#!/usr/bin/env python3
"""Verify the recovered package's original delivered files using SHA-256."""
import hashlib
from pathlib import Path

root=Path(__file__).resolve().parents[1]
failures=[]
entries=(root/'MANIFEST.sha256').read_text().splitlines()
for entry in entries:
    expected,relative=entry.split('  ',1)
    path=root/relative
    if not path.is_file():
        failures.append(f'Missing: {relative}')
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        failures.append(f'Changed: {relative}')
if failures:
    raise SystemExit('\n'.join(failures))
print(f'PASS: {len(entries)} original package files match MANIFEST.sha256.')
