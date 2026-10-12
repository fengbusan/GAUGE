import os
import argparse,json,time,traceback,os
import numpy as np,genesis as gs,trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from common import ProtocolBlocked,require_genesis_native_drive, native_marker_mapping,ROOT,load,mesh,target_points,sample_driver,save_result
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--smoke',action='store_true');a=p.parse_args();s,e=load(a.case);factor=int(os.environ.get('GENESIS_MPM_SUBSTEPS','1'));s['dt']/=factor;points=[];started=time.monotonic()
try:
 require_genesis_native_drive(e)
 gs.init(backend=gs.gpu,logging_level='warning');body=next(b for b in s['bodies']if b['kind']=='deformable');src=body['source'];v,f=mesh(src['path']);volume=np.abs(np.linalg.det(v[f[:,1:]]-v[f[:,0,None]]))/6;weights=np.zeros(len(v));np.add.at(weights,f.ravel(),np.repeat(volume/4,4));spacing=float(np.median(np.cbrt(weights)));scene=gs.Scene(sim_options=gs.options.SimOptions(dt=s['dt'],substeps=1,gravity=(0,0,-9.81)),show_viewer=False,mpm_options=gs.options.MPMOptions());faces=np.concatenate([f[:,[0,1,2]],f[:,[0,1,3]],f[:,[0,2,3]],f[:,[1,2,3]]]);keys,idx,count=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True);tm=trimesh.Trimesh(v,faces[idx[count==1]],process=True);tm.fix_normals();mat=gs.materials.MPM.Elastic(E=src['material']['young_modulus'],nu=src['material']['poisson_ratio'],rho=src['density']);out=ROOT/os.environ.get('THIRDPARTY_OUTPUT_ROOT','results')/'genesis'/a.case;out.mkdir(parents=True,exist_ok=True);asset=out/'surface.obj';tm.export(str(asset));ent=scene.add_entity(morph=gs.morphs.Mesh(file=str(asset)),material=mat);rigid={}
 for rb in s['bodies']:
  if rb['kind']!='rigid':continue
  q=rb['initial']['q'];rigid[rb['name']]=scene.add_entity(morph=gs.morphs.Mesh(file=rb['source']['path'],pos=tuple(q[:3]),quat=tuple(q[3:]),fixed=False,decimate=False),material=gs.materials.Rigid(friction=max(.01,rb['contact_material']['friction']),coup_friction=rb['contact_material']['friction']))
 scene.build();spacing=scene.sim.mpm_solver.particle_size
 initial=ent.get_particles_pos().cpu().numpy();ids,error=native_marker_mapping(e,initial,'genesis',a.case)
 def nearest_cap(refids):return np.flatnonzero(cKDTree(v[refids]).query(initial)[0]<=spacing)
 fixed=nearest_cap(src['fixed_vertices']);controls=[]
 for c in e['controls']:
  ci=nearest_cap(c['vertices']);assert len(ci);controls.append((ci,{**c,'rest_world':initial[ci].tolist()}))
 assert len(fixed);points.append(initial[ids]);n=3 if a.smoke else e['rollout_frames'];steps=round(1/30/s['dt'])
 def boundaries(t):
  ent.set_particles_pos(initial[fixed],fixed);ent.set_particles_vel(np.zeros((len(fixed),3)),fixed)
  for ci,c in controls:
   tar=target_points(c,t);vel=(target_points(c,t+s['dt'])-tar)/s['dt'];ent.set_particles_pos(tar,ci);ent.set_particles_vel(vel,ci)
 for frame in range(1,n):
  for sub in range(steps):
   t=((frame-1)*steps+sub)*s['dt'];boundaries(t)
   for d in e['drivers']:
    require_genesis_native_drive(e)
   scene.step();boundaries(t+s['dt'])
  all_positions=ent.get_particles_pos().cpu().numpy();val=all_positions[ids];
  if np.any(all_positions<=scene.sim.mpm_solver.boundary.lower+1e-6) or np.any(all_positions>=scene.sim.mpm_solver.boundary.upper-1e-6):raise FloatingPointError('MPM reached computational-domain boundary; trajectory invalid')

  if not np.isfinite(val).all():raise FloatingPointError('Non-finite simulated marker positions')
  points.append(val)
  if frame%30==0:print('FRAME',frame,'/',n,flush=True)
 save_result('genesis',a.case,points,'smoke_passed'if a.smoke else'completed',{'version':gs.__version__,'solver':'MPM.Elastic corotation defaults','internal_substeps':factor,'physics_hz':900*factor,'particles':len(initial),'particle_spacing_m':spacing,'grid_spacing_m':1/scene.sim.mpm_solver.grid_density,'marker_mapping_max_m':error,'boundary_rule':'Position/velocity projection on one particle-spacing end layer before and after each physical step','material_mapping':src['material'],'wall_seconds':time.monotonic()-started})
except Exception as ex:traceback.print_exc();save_result('genesis',a.case,points,'protocol_blocked'if isinstance(ex,ProtocolBlocked)or 'Entity has particles outside solver boundary' in str(ex)or 'assignment exceeds 1 cm' in str(ex)else'numerical_failure'if isinstance(ex,FloatingPointError)or 'Invalid constraint forces' in str(ex) or 'out of grid' in str(ex).lower() else'adapter_error',{'error':str(ex),'traceback':traceback.format_exc(),'internal_substeps':factor,'physics_hz':900*factor})
