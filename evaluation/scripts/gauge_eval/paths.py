"""Separate checked-in sources from writable datasets and outputs."""
import os
import json
import hashlib
from functools import lru_cache
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('GAUGE_WORKSPACE', SOURCE)).expanduser().resolve()
DATA = ROOT / 'data/GAUGE-Dataset'

@lru_cache(maxsize=1)
def dataset_inventory():
    return {item['path']:item['sha256'] for item in json.loads((SOURCE/'recipes/dataset_files.json').read_text())['files']}

def verify_dataset_paths(names):
    inventory=dataset_inventory()
    for name in names:
        path=DATA/name
        if name not in inventory or hashlib.sha256(path.read_bytes()).hexdigest()!=inventory[name]:
            raise ValueError('External dataset differs from pinned inputs: '+name)
