"""Frozen protocol must preserve SDK independence and declared adaptations."""
from pathlib import Path
import json
from collections import Counter
import pytest

RECIPES=Path(__file__).resolve().parents[1]/'recipes'
CASES=json.loads((RECIPES/'manifest.json').read_text())['cases']

def test_full_scope_and_external_data_inventory():
    assert len(CASES)==55
    assert Counter(c['regime']for c in CASES)=={'rigid':29,'textile':18,'deformable':8}
    assert len({c['case_id']for c in CASES})==55
    inventory=json.loads((RECIPES/'dataset_files.json').read_text())
    assert len({f['path']for f in inventory['files']})==len(inventory['files'])
    for f in inventory['files']:
        assert not Path(f['path']).is_absolute() and '..'not in Path(f['path']).parts
        assert len(f['sha256'])==64

@pytest.mark.parametrize('case',CASES,ids=[c['case_id']for c in CASES])
def test_sdk_protocol_and_native_no_pd(case):
    recipe=json.loads((RECIPES/(case['case_id']+'.json')).read_text())
    for mode in ['xie','native']:
        s=recipe[mode]['scene'];e=recipe[mode]['evaluation']
        assert abs(s['dt']-1/(900 if case['regime']=='deformable'else 180))<1e-14
        assert e['rollout_frames']>=e['frames']
        assert 'reference'not in e and 'real_observables'not in e
        for b in s['bodies']:
            if b['source'].get('path'):assert b['source']['path'].startswith('${PROFILE}/')
            if mode=='native':assert not b.get('controller')
        if mode=='xie':assert s['solver']['reduced_primal_recovery']=={'enabled':case['regime']=='rigid'}
        for d in e['drivers']:
            assert d.get('mode')=='pd'if mode=='xie'else d.get('mode')!='pd'
            assert 'measurement_anchor'in d and 'positions'not in d
