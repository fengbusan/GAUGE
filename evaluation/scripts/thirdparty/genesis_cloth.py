import os
import argparse,json,time,traceback
import numpy as np,genesis as gs,trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from common import ProtocolBlocked,require_genesis_native_drive, native_marker_mapping,ROOT,load,mesh,target_points,sample_driver,save_result,boundary_components,select_boundary
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--smoke',action='store_true');a=p.parse_args();s,e=load(a.case);points=[];started=time.monotonic()
try:
 require_genesis_native_drive(e)
 gs.init(backend=gs.gpu,logging_level='warning');scene=gs.Scene(sim_options=gs.options.SimOptions(dt=s['dt'],substeps=1,gravity=(0,0,-9.81)),show_viewer=False);body=next(b for b in s['bodies']if b['kind']=='deformable');src=body['source'];v,f=mesh(src['path']);meta=json.load(open(ROOT/'data/GAUGE-Dataset/metadata/textile'/f"{e['task']}.json"));m=meta['assets']['textile']['material'][e['material_parameter_key']];mat=gs.materials.PBD.Cloth(rho=m['density'],stretch_compliance=1/m['stretch stiffness'],bending_compliance=1/m['bend stiffness'],static_friction=m['friction'],kinetic_friction=m['friction']);out=ROOT/os.environ.get('THIRDPARTY_OUTPUT_ROOT','results')/'genesis'/a.case;out.mkdir(parents=True,exist_ok=True);asset=out/'surface.obj';trimesh.Trimesh(v,f,process=False).export(str(asset));ent=scene.add_entity(morph=gs.morphs.Mesh(file=str(asset)),material=mat);rigid={}
 for rb in s['bodies']:
  if rb['kind']!='rigid':continue
  q=rb['initial']['q'];rigid[rb['name']]=scene.add_entity(morph=gs.morphs.Mesh(file=rb['source']['path'],pos=tuple(q[:3]),quat=tuple(q[3:]),fixed=False,decimate=False),material=gs.materials.Rigid(friction=rb['contact_material']['friction'],coup_friction=rb['contact_material']['friction']))
 scene.build()
 initial=ent.get_particles_pos().cpu().numpy();ids,error=native_marker_mapping(e,initial,'genesis',a.case);fixed=[];controls=[]
 for component in boundary_components(f,src.get('fixed_vertices',[])):fixed.extend(select_boundary(initial,v,component).tolist())
 for c in e['controls']:
  ci=select_boundary(initial,v,c['vertices']);assert len(ci);controls.append((ci,{**c,'rest_world':initial[ci].tolist()}))
 pin=sorted(set(fixed+[int(i)for ci,c in controls for i in ci]));assert len(pin);ent.fix_particles(pin)
 points.append(initial[ids]);n=3 if a.smoke else e['rollout_frames'];steps=round(1/30/s['dt'])
 for frame in range(1,n):
  for sub in range(steps):
   t=((frame-1)*steps+sub+1)*s['dt']
   for ci,c in controls:ent.set_particles_pos(target_points(c,t),ci)
   for d in e['drivers']:
    require_genesis_native_drive(e)
   scene.step()
  val=ent.get_particles_pos().cpu().numpy()[ids];
  if not np.isfinite(val).all():raise FloatingPointError('Non-finite simulated marker positions')
  points.append(val)
  if frame%30==0:print('FRAME',frame,'/',n,flush=True)
 save_result('genesis',a.case,points,'smoke_passed'if a.smoke else'completed',{'version':gs.__version__,'solver':'PBD defaults','particles':len(initial),'marker_mapping_max_m':error,'material_mapping':{'areal_density':m['density'],'stretch_compliance':1/m['stretch stiffness'],'bending_compliance':1/m['bend stiffness']},'wall_seconds':time.monotonic()-started})
except Exception as ex:traceback.print_exc();save_result('genesis',a.case,points,'protocol_blocked'if isinstance(ex,ProtocolBlocked)else'numerical_failure'if isinstance(ex,FloatingPointError)or 'Invalid constraint forces' in str(ex) or 'out of grid' in str(ex).lower() else'adapter_error',{'error':str(ex),'traceback':traceback.format_exc()})
