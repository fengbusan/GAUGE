"""Xie SDK-only rollout. No engine repository, test tools, or source build needed."""
import argparse
import hashlib
import importlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from gauge_eval.paths import ROOT, DATA

def sample_driver(d,t):
    p=np.asarray(d['positions']);rot=Rotation.from_quat(d['quaternions_xyzw'])
    if len(p)<2:
        q=rot[0].as_quat()
        return np.r_[p[0],q[3],q[:3]],np.zeros(6)
    u=np.clip(t*d['fps'],0,len(p)-1);i=min(int(u),len(p)-2);a=u-i
    r=Slerp([0,1],rot[i:i+2])([a])[0];q=r.as_quat()
    linear=(p[i+1]-p[i])*d['fps'];angular=(rot[i+1]*rot[i].inv()).as_rotvec()*d['fps']
    if t>=(len(p)-1)/d['fps']:
        linear=np.zeros(3);angular=np.zeros(3)
    return np.r_[(1-a)*p[i]+a*p[i+1],q[3],q[:3]],np.r_[r.inv().apply(linear),r.inv().apply(angular)]

def run(cid,smoke=False,record_state=False):
    folder=ROOT/'configs/xie_sdk_v1'/cid
    e=json.loads((folder/'evaluation.json').read_text());scene=json.loads((folder/'scene.json').read_text())
    out=ROOT/('results/smoke' if smoke else 'results')/'xiephysic'/cid
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();sim=None;points=[];tracking=[];state_file=None
    status=dict(engine='xiephysic',case_id=cid,status='running',frames_required=e['rollout_frames'],score_frames=e['frames'],sample_hz=30,physics_hz=round(1/scene['dt']),source_scene_sha256=e['scene_sha256'],source_evaluation_sha256=hashlib.sha256((folder/'evaluation.json').read_bytes()).hexdigest(),official_equivalence_verified=False,rp_enabled=scene['solver'].get('reduced_primal_recovery',{}).get('enabled',False))
    def save():
        status.update(frames_completed=len(points),wall_seconds=time.monotonic()-start)
        np.save(out/'marker_positions_m.npy',np.asarray(points))
        (out/'status.json').write_text(json.dumps(status,indent=2))
    try:
        if hashlib.sha256((folder/'scene.json').read_bytes()).hexdigest()!=e['scene_sha256']:raise ValueError('Scene differs from prepared receipt')
        module=os.environ.get('GAUGE_XIE_SDK_MODULE','xiephy_gpu')
        sdk_path=os.environ.get('GAUGE_XIE_SDK_PATH')
        if sdk_path:
            sdk_path=Path(sdk_path).expanduser().resolve()
            # Isolate this SDK process from stale editable-install finders and
            # packages eagerly imported by site initialization. No source build.
            sys.meta_path[:]=[f for f in sys.meta_path if not type(f).__module__.startswith('_editable')]
            for name in list(sys.modules):
                if name==module or name.startswith(module+'.'):del sys.modules[name]
            sys.path.insert(0,str(sdk_path));importlib.invalidate_caches()
        sdk=importlib.import_module(module)
        if sdk_path and not Path(sdk.__file__).resolve().is_relative_to(sdk_path):raise ValueError('Loaded SDK is outside GAUGE_XIE_SDK_PATH')
        status.update(version=sdk.__version__,sdk_module=sdk.__name__)
        native=Path(sdk._C.__file__) if hasattr(sdk,'_C') else None
        if native:status['sdk_binary_sha256']=hashlib.sha256(native.read_bytes()).hexdigest()
        expected=os.environ.get('GAUGE_XIE_SDK_VERSION','0.5.13')
        if sdk.__version__!=expected:raise ValueError(f'SDK version {sdk.__version__} differs from requested {expected}')
        sim=sdk.Sim.from_json(str(folder/'scene.json'))
        if record_state:state_file=(out/'state_q.jsonl').open('w')
        def observe():
            snap=sim.snapshot()
            if 'articulated'in e:
                p=np.asarray([np.asarray(b['pivot'])+Rotation.from_rotvec(np.asarray(b['axis'])*snap.q(b['name'])[0]).apply(b['offset']) for b in e['articulated']])
            elif e['regime']=='rigid':p=np.asarray(snap.q(e['body'])[:3])[None].copy()
            else:p=np.asarray(snap.q(e['body'])).reshape(-1,3)[e['marker_vertex_ids']].copy()
            if not np.isfinite(p).all():raise FloatingPointError('Nonfinite material markers')
            points.append(p)
            if state_file:
                state_file.write(json.dumps({'time':float(snap.time),'q':{b['name']:np.asarray(snap.q(b['name'])).tolist() for b in scene['bodies']}})+'\n')
                state_file.flush()
        observe();substeps=round(1/(30*scene['dt']))
        if e['regime']!='rigid':
            target=np.asarray(json.loads((DATA/'metadata'/e['regime']/(e['task']+'.json')).read_text())['markers']['translation'])
            error=float(np.max(np.linalg.norm(points[0]-target,axis=-1)))
            mapping={'algorithm':'Frozen Hungarian native vertex IDs; verified against actual initial SDK positions','native_ids':e['marker_vertex_ids'],'max_error_m':error,'passed':error<.01}
            (out/'marker_mapping.json').write_text(json.dumps(mapping,indent=2))
            if not mapping['passed']:raise ValueError('Native marker mapping exceeds 1 cm')
        if not np.isclose(substeps*scene['dt'],1/30,rtol=0,atol=1e-12):raise ValueError('dt must divide 30 Hz output interval')
        for frame in range(1,min(3,e['rollout_frames']) if smoke else e['rollout_frames']):
            for sub in range(substeps):
                t=((frame-1)*substeps+sub+1)*scene['dt']
                for d in e['drivers']:
                    q,qd=sample_driver(d,t)
                    if d.get('mode')!='pd':raise ValueError('Xie moving driver must use the declared PD adaptation')
                    sim.set_control({d['name']:{'q_target':q.tolist(),'qd_target':qd.tolist()}})
                if not sim.step():raise RuntimeError('SDK step failed')
            observe()
            for d in e['drivers']:
                target,_=sample_driver(d,frame/30);actual=np.asarray(sim.snapshot().q(d['name']))
                tracking.append({'frame':frame,'body':d['name'],'position_error_m':float(np.linalg.norm(actual[:3]-target[:3])),'rotation_error_rad':float((Rotation.from_quat(actual[[4,5,6,3]])*Rotation.from_quat(target[[4,5,6,3]]).inv()).magnitude())})
            if frame%30==0:save();print('FRAME',frame,'/',e['rollout_frames'],flush=True)
        status['status']='smoke_passed' if smoke else 'completed'
    except Exception as ex:
        status.update(status='timeout' if isinstance(ex,TimeoutError) else 'numerical_failure' if isinstance(ex,FloatingPointError) else 'adapter_error',error=str(ex),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if state_file:state_file.close()
        if sim:sim.close()
        (out/'control_tracking.json').write_text(json.dumps(tracking))
        save()
    return 0 if status['status'] in ['completed','smoke_passed'] else 1

if __name__=='__main__':
    def timeout(signum,frame):raise TimeoutError('Case wall budget exceeded')
    signal.signal(signal.SIGTERM,timeout)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case');p.add_argument('--smoke',action='store_true');p.add_argument('--record-state',action='store_true')
    args=p.parse_args();raise SystemExit(run(args.case,args.smoke,args.record_state))
