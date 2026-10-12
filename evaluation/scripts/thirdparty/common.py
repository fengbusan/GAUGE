import os
"""Frozen GAUGE inputs and engine-independent observations for local native baselines."""
import json,hashlib,time,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(os.environ.get('GAUGE_WORKSPACE',Path(__file__).resolve().parents[2])).expanduser().resolve()
PROFILE=os.environ.get('GAUGE_NATIVE_PROFILE',os.environ.get('GAUGE_PROFILE','native_sdk_v1'))

class ProtocolBlocked(RuntimeError):
 pass

def require_genesis_native_drive(e):
 if e['drivers']:
  raise ProtocolBlocked('Official metadata requires moving kinematic colliders. Installed Genesis Kinematic material is visualization-only; fixed-root pose setters do not provide verified moving-contact velocity. No PD or dynamic teleport fallback is permitted. Official runner or a verified native kinematic-contact path is required.')

def native_marker_mapping(e, positions, engine, cid):
 from scipy.optimize import linear_sum_assignment
 metadata_path=ROOT/'data/GAUGE-Dataset/metadata'/e['regime']/(e['task']+'.json')
 target=np.asarray(json.load(open(metadata_path))['markers']['translation'],dtype=float)
 positions=np.asarray(positions,dtype=float)
 costs=np.linalg.norm(target[:,None]-positions[None],axis=-1)
 rows,ids=linear_sum_assignment(costs)
 assert len(rows)==len(target) and np.array_equal(rows,np.arange(len(target)))
 distances=costs[rows,ids]
 out=ROOT/os.environ.get('THIRDPARTY_OUTPUT_ROOT','results')/engine/cid
 out.mkdir(parents=True,exist_ok=True)
 record={'source':str(metadata_path),'source_sha256':hashlib.sha256(metadata_path.read_bytes()).hexdigest(),'target':'official metadata markers.translation, world meters','algorithm':'direct Hungarian Euclidean-cost unique assignment to native positions; no intermediate proxy','native_count':len(positions),'marker_ids':list(range(len(target))),'native_ids':ids.tolist(),'target_positions_m':target.tolist(),'native_positions_m':positions[ids].tolist(),'errors_m':distances.tolist(),'max_error_m':float(distances.max()),'passed':bool(distances.max()<.01)}
 (out/'marker_mapping.json').write_text(json.dumps(record,indent=2))
 if not record['passed']:raise ValueError('Direct official-marker assignment exceeds 1 cm; see marker_mapping.json')
 return ids,float(distances.max())
def load(cid):
 p=ROOT/'configs'/PROFILE/cid
 s=json.load(open(p/'scene.json'));e=json.load(open(p/'evaluation.json'))
 if hashlib.sha256((p/'scene.json').read_bytes()).hexdigest()!=e['scene_sha256']:raise ValueError('Scene differs from prepared receipt')
 return s,e
def sample_driver(d,t):
 p=np.asarray(d['positions']);rots=Rotation.from_quat(d['quaternions_xyzw']);u=np.clip(t*d['fps'],0,len(p)-1);i=min(int(u),len(p)-2);a=u-i;r=Slerp([0,1],rots[i:i+2])([a])[0];q=r.as_quat()
 return np.r_[(1-a)*p[i]+a*p[i+1],q[3],q[:3]]
def target_points(c,t):
 q=sample_driver(c['driver'],t);a=np.asarray(c['anchor']);r0=Rotation.from_quat(a[[4,5,6,3]]);r=Rotation.from_quat(q[[4,5,6,3]])
 return r.apply(r0.inv().apply(np.asarray(c['rest_world'])-a[:3]))+q[:3]
def mesh(path):
 lines=Path(path).read_text().splitlines();i=lines.index('$Nodes');n=int(lines[i+1]);v=np.array([list(map(float,x.split()[1:]))for x in lines[i+2:i+2+n]]);i=lines.index('$Elements');n=int(lines[i+1]);f=[]
 for row in lines[i+2:i+2+n]:
  a=list(map(int,row.split()));f.append([x-1 for x in a[3+a[2]:]])
 return v,np.array(f)
def save_result(engine,cid,points,status,extra=None):
 out=ROOT/os.environ.get('THIRDPARTY_OUTPUT_ROOT','results')/engine/cid;out.mkdir(parents=True,exist_ok=True);s,e=load(cid);p=np.asarray(points);np.save(out/'marker_positions_m.npy',p)
 data={'engine':engine,'case_id':cid,'status':status,'frames_completed':len(p),'frames_required':e['rollout_frames'],'score_frames':e['frames'],'sample_hz':30,'physics_hz':round(1/s['dt']),'source_scene_sha256':e['scene_sha256'],'source_evaluation_sha256':hashlib.sha256((ROOT/'configs'/PROFILE/cid/'evaluation.json').read_bytes()).hexdigest(),'observation':'rigid body origin or material marker positions, meters','trajectory_sha256':hashlib.sha256((out/'marker_positions_m.npy').read_bytes()).hexdigest(),**(extra or {})}
 (out/'status.json').write_text(json.dumps(data,indent=2));return data
def boundary_components(f,ids):
 remaining=set(ids);adj={i:set()for i in ids}
 for cell in f:
  nodes=set(map(int,cell))&remaining
  for i in nodes:adj[i].update(nodes-{i})
 groups=[]
 while remaining:
  todo=[remaining.pop()];component=[]
  while todo:
   i=todo.pop();component.append(i);n=adj[i]&remaining;remaining-=n;todo.extend(n)
  groups.append(component)
 return groups
def select_boundary(points,reference,ids,tolerance=1e-5):
 cloud=reference[ids];center=cloud.mean(0);_,sv,axes=np.linalg.svd(cloud-center,full_matrices=True);rank=int(np.sum(sv>1e-6));p=(points-center)@axes.T;c=(cloud-center)@axes.T
 return np.flatnonzero(np.all((p[:,:rank]>=c[:,:rank].min(0)-tolerance)&(p[:,:rank]<=c[:,:rank].max(0)+tolerance),axis=1)&(np.linalg.norm(p[:,rank:],axis=1)<tolerance))


def load_obj_mesh(path):
 from types import SimpleNamespace
 vertices=[];faces=[]
 for line in Path(path).read_text().splitlines():
  parts=line.split()
  if not parts:continue
  if parts[0]=='v':vertices.append(list(map(float,parts[1:4])))
  if parts[0]=='f':
   ids=[int(x.split('/')[0])for x in parts[1:]]
   ids=[i-1 if i>0 else len(vertices)+i for i in ids]
   faces.extend([[ids[0],ids[j],ids[j+1]]for j in range(1,len(ids)-1)])
 return SimpleNamespace(vertices=np.asarray(vertices),faces=np.asarray(faces,dtype=int))


def vertices_in_authored_grip(points, cloud, tolerance=1e-6):
 """Transfer a convex volume grip / planar cap; never hull a textile frame."""
 from scipy.spatial import ConvexHull
 points=np.asarray(points);cloud=np.asarray(cloud);center=cloud.mean(0);_,sv,axes=np.linalg.svd(cloud-center,full_matrices=False);rank=int(np.sum(sv>1e-6));projected=(points-center)@axes.T;support=(cloud-center)@axes.T
 if rank==3:
  equations=ConvexHull(support[:,:3]).equations;inside=np.all(projected[:,:3]@equations[:,:3].T+equations[:,3]<=tolerance,axis=1)
 elif rank==2:
  equations=ConvexHull(support[:,:2]).equations;inside=np.all(projected[:,:2]@equations[:,:2].T+equations[:,2]<=tolerance,axis=1)&(np.abs(projected[:,2])<=tolerance)
 else:raise ValueError('Authored volume grip must span a plane or volume')
 ids=np.flatnonzero(inside);assert len(ids),'No native nodes in authored grip';return ids
