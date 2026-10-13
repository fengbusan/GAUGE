"""SAP from the native-peer protocol, never from the IPC boundary adaptation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from prepare_sap import load_sdk
from gauge_eval.paths import ROOT, SOURCE
from gauge_eval.scenes import obj
from gauge_eval.mesh_topology import split_disconnected_vertex_fans
from gauge_eval.sap_protocol import freeze_solver, native_geometry_kind

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',default='sap_native_protocol_0514_v4')
    p.add_argument('--moving-grip',choices=['strict','compliant'],default='strict')
    p.add_argument('--endpoint-ccd',action=argparse.BooleanOptionalAction,default=True,
                   help='Only toggle SAP endpoint point CCD; keep the other frozen settings')
    a=p.parse_args();sdk=load_sdk();target=ROOT/'configs'/a.profile;target.mkdir(parents=True,exist_ok=True)
    records=[]
    for case in json.loads((SOURCE/'recipes/manifest.json').read_text())['cases']:
        cid=case['case_id'];base=ROOT/'configs/native_sdk_v1'/cid
        s=json.loads((base/'scene.json').read_text());e=json.loads((base/'evaluation.json').read_text())
        if sha(base/'scene.json')!=e['scene_sha256']:raise ValueError('Native base hash mismatch')
        legacy=json.loads((ROOT/'configs/xie_sdk_v1'/cid/'scene.json').read_text())
        out=target/cid;out.mkdir(exist_ok=True);geometry=[]
        # Native-peer recipes omit engine-specific numerical schema fields.
        # Retain the previously audited SAP/SDK material-clearance block, while
        # replacing the boundary/asset protocol and the requested budget.
        s['solver']=freeze_solver(legacy['solver'],endpoint_ccd=a.endpoint_ccd)
        # Authored USD contact exclusions are independent of boundary force
        # models. Reuse their audited vertex/face translation into SDK-native
        # attachment ownership, with ZERO stiffness for exact static grips.
        moving_names={d['name']for d in e['drivers']}
        if legacy.get('collision_pairs'):s['collision_pairs']=copy.deepcopy(legacy['collision_pairs'])
        if legacy.get('attachments'):
            s['attachments']=copy.deepcopy(legacy['attachments'])
            for z in s['attachments']:
                if z['rigid_body']not in moving_names or a.moving_grip=='strict':
                    z['constraints']['position']['stiffness']=0.0
        # SDK spelling for a static rigid actor is a stationary kinematic actor.
        for body in s['bodies']:
            body.pop('controller',None)
            if body['motion']=='static':body['motion']='kinematic'
            if body['kind']=='rigid' and body['motion']=='dynamic':
                # JSON schema requires the key; null creates NO controller.
                body['controller']=None
            if body['kind']=='deformable':
                # Material conversion into this SDK's hinge energy, not a PD.
                src=next(b['source']for b in legacy['bodies']if b['name']==body['name'])
                if 'extra_constitutions'in src:body['source']['extra_constitutions']=copy.deepcopy(src['extra_constitutions'])
                continue
            source=body['source']
            if source.get('type')!='mesh_obj':continue
            path=Path(source['path']);v,f=obj(path);kind,info=native_geometry_kind(v,f)
            receipt={'body':body['name'],'original_source_sha256':sha(path),'representation':kind,**info}
            if kind=='primitive_box':
                body['source']={k:copy.deepcopy(source[k])for k in ['mass','inertia','center_of_mass']if k in source}
                body['source'].update(type='primitive_box',half_extents=info['half_extents'])
            else:
                source['collision_representation']=kind
                if kind=='triangle_surface':
                    vv,ff,splits=split_disconnected_vertex_fans(v,f)
                    if splits:
                        asset=out/'assets'/path.name;asset.parent.mkdir(exist_ok=True)
                        asset.write_text('\n'.join(['v '+' '.join(map(str,z))for z in vv]+['f '+' '.join(str(int(k)+1)for k in t)for t in ff])+'\n')
                        source['path']=str(asset);receipt.update(topology_only=True,vertices_before=len(v),vertices_after=len(vv),triangle_coordinates_exactly_unchanged=True,output_sha256=sha(asset))
            geometry.append(receipt)
        for d in e['drivers']:
            if d.get('mode')!='kinematic':raise ValueError('Native driver must be kinematic')
            body=next(b for b in s['bodies']if b['name']==d['name'])
            if body['motion']!='kinematic':raise ValueError('Driver body must be kinematic')
        blocked=None
        if e.get('controls'):
            if a.moving_grip=='strict':
                blocked=f'SDK {sdk.__version__} has no exact moving soft-rigid grip: native fixed_vertices are static Dirichlet only; moving position/attachment APIs are compliant. No PD, moving fixed-mask rewrite or free-vertex projection used.'
            else:
                # The already declared, uniform constitutive attachment stiffness
                # remains unchanged. Only moving grip groups have nonzero stiffness.
                soft=next(b for b in s['bodies']if b['kind']=='deformable')
                for z in s['attachments']:
                    if z['rigid_body']not in moving_names:continue
                    ids=z['soft_region']['cell_indices']
                    soft['source']['fixed_vertices']=[i for i in soft['source'].get('fixed_vertices',[])if i not in set(ids)]
                e['comparison_flags']=sorted(set(e['comparison_flags'])|{'native_compliant_moving_grip'})
        # These were incorrectly carried in native recipe metadata, despite the
        # native URDF/pivots already using the official 20 mm spacing/zero gap.
        e['comparison_flags']=[f for f in e['comparison_flags']if f!='cradle_gap_adaptation']
        e.update(profile=a.profile,backend='sap',source_base_profile='native_sdk_v1',
                 source_base_scene_sha256=e['scene_sha256'],driver_protocol='native_kinematic_begin_step',
                 moving_grip_policy=a.moving_grip,protocol_blocked_reason=blocked,
                 collision_mapping=geometry,ipc_adaptations_used=False,
                 cradle_surface_gap_m=0 if e['metric']=='cradle'else None,
                 endpoint_ccd_scope='Soft/rod position-only local filter; rigid/joint poses are not clipped; SDF/EE/EF coverage is limited',
                 native_sap_parameters=s['solver']['velocity_primal'])
        (out/'scene.json').write_text(json.dumps(s,indent=2));e['scene_sha256']=sha(out/'scene.json')
        (out/'evaluation.json').write_text(json.dumps(e,indent=2))
        records.append({**case,'scene_sha256':e['scene_sha256'],'evaluation_sha256':sha(out/'evaluation.json'),'protocol_blocked_reason':blocked,'collision_mapping':geometry})
    receipt={'profile':a.profile,'base_profile':'native_sdk_v1','version':sdk.__version__,'native_sha256':sha(sdk._C.__file__),'solver':'velocity_primal/frozen_sap','requested_parameters':freeze_solver({},endpoint_ccd=a.endpoint_ccd)['velocity_primal'],'moving_grip_policy':a.moving_grip,'pd_used':False,'rp_used':False,'cradle_gap_adaptation':False,'no_per_case_score_tuning':True,'cases':records}
    (target/'manifest.json').write_text(json.dumps(receipt,indent=2));print('Frozen',len(records),'native SAP cases;',sum(bool(c['protocol_blocked_reason'])for c in records),'exact-grip protocol blocks')

if __name__=='__main__':main()
