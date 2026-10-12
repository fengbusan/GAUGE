"""Freeze a SAP comparison using the installed 0.5.14 SDK's native policy.

Only numerical solver selection changes. Original measured bodies, attachments,
drivers, collision geometry, time step, materials and score windows stay frozen.
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from gauge_eval.paths import ROOT, SOURCE
from gauge_eval.scenes import obj
from gauge_eval.mesh_topology import split_disconnected_vertex_fans

def load_sdk():
    path=Path(os.environ['GAUGE_XIE_SDK_PATH']).expanduser().resolve()
    sys.meta_path[:]=[f for f in sys.meta_path if not type(f).__module__.startswith('_editable')]
    for name in list(sys.modules):
        if name=='xiephy_gpu' or name.startswith('xiephy_gpu.'):del sys.modules[name]
    sys.path.insert(0,str(path));importlib.invalidate_caches()
    sdk=importlib.import_module('xiephy_gpu')
    if sdk.__version__!='0.5.14' or not Path(sdk.__file__).resolve().is_relative_to(path):
        raise ValueError('Preparation requires the explicitly selected 0.5.14 SDK')
    return sdk

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-profile',default='xie_sdk_v1')
    p.add_argument('--profile',default='xie_sap_native_0514_v2')
    p.add_argument('--sap-policy',choices=['native','usd'],default='native',help='Native JSON defaults, or explicit USD scene-dependent policy for diagnostics')
    p.add_argument('--normalize-rigid-topology',action='store_true',help='Losslessly split disconnected rigid vertex fans; preserve every triangle coordinate')
    a=p.parse_args();sdk=load_sdk()
    from xiephy_gpu.sim.usd_solver_policy import apply_sap_scene_defaults
    from pxr import Usd
    stage=Usd.Stage.CreateInMemory();prim=stage.DefinePrim('/Physics')
    target=ROOT/'configs'/a.profile;target.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((SOURCE/'recipes/manifest.json').read_text());records=[]
    for case in manifest['cases']:
        cid=case['case_id'];base=ROOT/'configs'/a.base_profile/cid
        s=json.loads((base/'scene.json').read_text());e=json.loads((base/'evaluation.json').read_text())
        if hashlib.sha256((base/'scene.json').read_bytes()).hexdigest()!=e['scene_sha256']:
            raise ValueError('Base scene receipt mismatch: '+cid)
        original=copy.deepcopy(s);solver=s['solver']
        solver.update(integrator='bdf1',contact_solver={'model':'velocity_primal'},
                      reduced_primal_recovery={'enabled':False},
                      velocity_primal={'strategy':'frozen_sap'},
                      linear_solver={'type':'auto','convergence':'fixed_iterations','preconditioner':{'type':'block_jacobi'}})
        defaults=[]
        policy=(apply_sap_scene_defaults(prim,s,lambda _p,k,v:defaults.append({'field':k,'value':v})) if a.sap_policy=='usd'
                else {'selection':'native_json','resolved':'native_json','overrides':{},'note':'Leave all SAP numerical and swept-candidate fields unauthored; native scene parser owns defaults'})
        out=target/cid;out.mkdir(exist_ok=True)
        topology_receipts=[]
        if a.normalize_rigid_topology:
            for body in s['bodies']:
                source=body['source']
                if body['kind']!='rigid' or source.get('type')!='mesh_obj' or source.get('collision_representation')!='triangle_surface':continue
                path=Path(source['path']);v,f=obj(path)
                vv,ff,splits=split_disconnected_vertex_fans(v,f)
                if not splits:continue
                asset=out/'assets'/path.name;asset.parent.mkdir(exist_ok=True)
                asset.write_text('\n'.join(['v '+' '.join(map(str,p))for p in vv]+['f '+' '.join(str(int(k)+1)for k in t)for t in ff])+'\n')
                topology_receipts.append({'body':body['name'],'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'normalized_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'vertices_before':len(v),'vertices_after':len(vv),'triangle_coordinates_exactly_unchanged':True,'splits':splits})
                source['path']=str(asset)
        # The sole non-solver difference is a hash-recorded, lossless mesh path.
        comparable=copy.deepcopy(s)
        for changed in topology_receipts:
            next(b for b in comparable['bodies']if b['name']==changed['body'])['source']['path']=changed['source']
        assert {k:v for k,v in original.items() if k!='solver'}=={k:v for k,v in comparable.items() if k!='solver'}
        (out/'scene.json').write_text(json.dumps(s,indent=2))
        e.update(profile=a.profile,backend='sap',source_base_profile=a.base_profile,
                 source_base_scene_sha256=e['scene_sha256'],sap_scene_policy=policy,
                 scene_sha256=hashlib.sha256((out/'scene.json').read_bytes()).hexdigest(),rigid_topology_normalization=topology_receipts)
        (out/'evaluation.json').write_text(json.dumps(e,indent=2))
        records.append({**case,'scene_sha256':e['scene_sha256'],'evaluation_sha256':hashlib.sha256((out/'evaluation.json').read_bytes()).hexdigest(),'sap_scene_policy':policy,'rigid_topology_normalization':topology_receipts})
    native=Path(sdk._C.__file__)
    receipt={'profile':a.profile,'base_profile':a.base_profile,'version':sdk.__version__,
             'native_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),
             'solver':'velocity_primal/frozen_sap','sap_policy':a.sap_policy,
             'standard_policy_source':('native JSON parser sap_solver_defaults.h' if a.sap_policy=='native' else 'installed SDK usd_solver_policy.apply_sap_scene_defaults'),
             'rigid_policy':('Native JSON production defaults: 10/10, relative 1e-3, velocity 0.05, swept candidates.' if a.sap_policy=='native' else 'USD profile: 80/30, relative 1e-4, velocity 1e-5, swept_point_contacts=false'),
             'mixed_policy':'Native production defaults: 10/10, relative 1e-3, velocity 0.05, swept candidates.',
             'physical_inputs_and_drivers_unchanged':True,'no_per_task_score_tuning':True,'cases':records}
    receipt['normalize_rigid_topology']=a.normalize_rigid_topology
    (target/'manifest.json').write_text(json.dumps(receipt,indent=2))
    print('Frozen',len(records),'SAP cases; native SHA256',receipt['native_sha256'])

if __name__=='__main__':main()
