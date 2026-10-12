"""Download only the pinned external files required by the 55-case recipes."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from urllib.parse import quote
from gauge_eval.paths import DATA, SOURCE

def fetch(item, revision):
    target = DATA/item['path']
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']:
        return
    url = 'https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset/resolve/'+revision+'/'+quote(item['path'])
    with urlopen(url, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest()!=item['sha256']:
        raise ValueError('Pinned dataset hash mismatch: '+item['path'])
    target.parent.mkdir(parents=True,exist_ok=True)
    temp = target.with_suffix(target.suffix+'.partial')
    temp.write_bytes(payload)
    temp.replace(target)

if __name__=='__main__':
    inventory=json.loads((SOURCE/'recipes/dataset_files.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item:fetch(item,inventory['revision']),inventory['files']))
    print('Verified',len(inventory['files']),'external dataset files at',inventory['revision'])
