import os
import argparse,json,time,traceback,xml.etree.ElementTree as ET
import numpy as np,genesis as gs
from scipy.spatial.transform import Rotation
from common import ProtocolBlocked,require_genesis_native_drive, ROOT,load,sample_driver,save_result
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--smoke',action='store_true');a=p.parse_args();s,e=load(a.case);points=[];driver_errors=[];started=time.monotonic()
try:
 require_genesis_native_drive(e)
 gs.init(backend=gs.gpu,logging_level='warning');scene=gs.Scene(sim_options=gs.options.SimOptions(dt=s['dt'],substeps=1,gravity=(0,0,-9.81)),show_viewer=False);entities={};artnames={x['name']for x in e.get('articulated',[])};driven={d['name']for d in e['drivers']}
 for b in s['bodies']:
  src=b['source'];q=b['initial']['q'];mat=gs.materials.Rigid(friction=b['contact_material']['friction'],coup_restitution=b['contact_material'].get('restitution',0));name=b['name']
  if name in artnames:
   tree=ET.parse(src['path'])
   for inertia in tree.findall('.//inertia'):
    for field in ['ixy','ixz','iyz']:
     if field not in inertia.attrib:inertia.set(field,'0')
   out=ROOT/os.environ.get('THIRDPARTY_OUTPUT_ROOT','results')/'genesis'/a.case;out.mkdir(parents=True,exist_ok=True);urdf=out/(name+'.urdf');tree.write(urdf);morph=gs.morphs.URDF(file=str(urdf),fixed=True,merge_fixed_links=False)
  else:
   opts=dict(pos=tuple(q[:3]),quat=tuple(q[3:]),fixed=b['motion']!='dynamic')
   if src['type']=='mesh_obj':morph=gs.morphs.Mesh(file=src['path'],decimate=False,align=False,**opts)
   elif src['type']in ['sphere','primitive_sphere']:morph=gs.morphs.Sphere(radius=src['radius'],**opts)
   else:raise ValueError(src['type'])
  entities[name]=scene.add_entity(morph=morph,material=mat)
 if s['solver'].get('ground_enabled'):
  gm=s['solver']['ground_material'];scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,s['solver']['ground_height'])),material=gs.materials.Rigid(friction=gm['friction'],coup_restitution=gm['restitution']))
 scene.build()
 for b in s['bodies']:
  ent=entities[b['name']];src=b['source']
  if b['name']in artnames:ent.set_dofs_position(b['initial']['q'])
  elif b['motion']=='dynamic':
   ent.set_mass(src['mass'])
   if 'inertia'in src:
    link=ent.links[0];link.set_COM(np.array(src['center_of_mass']));link.set_inertia(np.array(src['inertia']))
 def observe():
  names=[x['name']for x in e['articulated']]if 'articulated'in e else[e['body']]
  return np.array([(entities[n].get_link('bob').get_pos()if n in artnames else entities[n].get_pos()).detach().cpu().numpy().reshape(3)for n in names])
 points.append(observe());n=3 if a.smoke else e['rollout_frames'];steps=round(1/30/s['dt'])
 for frame in range(1,n):
  for sub in range(steps):
   t=((frame-1)*steps+sub+1)*s['dt']
   for d in e['drivers']:
    require_genesis_native_drive(e)
   scene.step()
  val=observe();
  if not np.isfinite(val).all():raise FloatingPointError('Non-finite simulated marker positions')
  points.append(val)
  for d in e['drivers']:driver_errors.append(float(np.linalg.norm(entities[d['name']].get_pos().detach().cpu().numpy().reshape(3)-sample_driver(d,frame/30)[:3])))
  if frame%30==0:print('FRAME',frame,'/',n,flush=True)
 save_result('genesis',a.case,points,'smoke_passed'if a.smoke else'completed',{'version':gs.__version__,'solver':'native rigid solver defaults','driver':'No moving kinematic collider in this case; no PD','driver_max_position_error_m':max(driver_errors,default=0.),'wall_seconds':time.monotonic()-started,'restitution_note':'coup_restitution governs cross-solver coupling only; native rigid contact has no coefficient-of-restitution input'})
except Exception as ex:traceback.print_exc();save_result('genesis',a.case,points,'protocol_blocked'if isinstance(ex,ProtocolBlocked)else'numerical_failure'if isinstance(ex,FloatingPointError)or 'Invalid constraint forces' in str(ex) or 'out of grid' in str(ex).lower() else'adapter_error',{'error':str(ex),'traceback':traceback.format_exc()})
