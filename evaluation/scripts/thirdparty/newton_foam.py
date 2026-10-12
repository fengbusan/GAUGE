import argparse,json,time,traceback
import numpy as np,newton,warp as wp,trimesh
from scipy.spatial.transform import Rotation
from common import load_obj_mesh, native_marker_mapping,ROOT,load,mesh,target_points,sample_driver,save_result
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--smoke',action='store_true');a=p.parse_args();s,e=load(a.case);points=[];start=time.monotonic();boundary_checks=[];pre_projection={'fixed_max_error_m':0.,'moving_max_error_m':0.}
try:
 wp.init();b=newton.ModelBuilder();body=next(x for x in s['bodies']if x['kind']=='deformable');src=body['source'];v,f=mesh(src['path']);volume=np.abs(np.linalg.det(v[f[:,1:]]-v[f[:,0,None]]))/6;weights=np.zeros(len(v));np.add.at(weights,f.ravel(),np.repeat(volume/4,4));assert np.all(weights>0)
 newton.solvers.SolverImplicitMPM.register_custom_attributes(b);b.add_particles(pos=v.tolist(),vel=np.zeros_like(v).tolist(),mass=(weights*src['density']).tolist(),radius=(np.cbrt(weights)/2).tolist())
 fixed=set(src.get('fixed_vertices',[]));moving=set(i for c in e['controls']for i in c['vertices'])
 for i in fixed|moving:b.particle_mass[i]=0
 names={}
 for rb in s['bodies']:
  if rb['kind']!='rigid':continue
  q=rb['initial']['q'];idx=b.add_body(xform=wp.transform(q[:3],q[4:]+q[3:4]),is_kinematic=True,mass=rb['source']['mass']);names[rb['name']]=idx;tm=load_obj_mesh(rb['source']['path']);b.add_shape_mesh(idx,mesh=newton.Mesh(np.asarray(tm.vertices,np.float32),np.asarray(tm.faces,np.int32).ravel()),cfg=b.ShapeConfig(density=0,mu=rb['contact_material']['friction'],restitution=rb['contact_material']['restitution']))
 model=b.finalize();model.mpm.young_modulus.fill_(src['material']['young_modulus']);model.mpm.poisson_ratio.fill_(src['material']['poisson_ratio']);model.mpm.tensile_yield_ratio.fill_(1.0);cfg=newton.solvers.SolverImplicitMPM.Config();solver=newton.solvers.SolverImplicitMPM(model,config=cfg);state=model.state();nxt=model.state();control=model.control();contacts=None;ids,marker_error=native_marker_mapping(e,state.particle_q.numpy(),'newton',a.case);points.append(state.particle_q.numpy()[ids]);n=3 if a.smoke else e['rollout_frames'];steps=round(1/30/s['dt'])
 for frame in range(1,n):
  for sub in range(steps):
   t=((frame-1)*steps+sub+1)*s['dt'];t0=t-s['dt'];q=state.particle_q.numpy();qd=state.particle_qd.numpy();grad=state.mpm.particle_qd_grad.numpy()
   # Native MPM infinite-mass boundary particles are advected. Prescribe the
   # beginning-of-step state and rigid velocity gradient, as in Newton's native
   # example_mpm_beam_twist; prescribing the end pose here advances it twice.
   fi=sorted(fixed);q[fi]=v[fi];qd[fi]=0;grad[fi]=0
   for c in e['controls']:
    target=target_points(c,t0);ci=c['vertices'];qa=sample_driver(c['driver'],t0);qb=sample_driver(c['driver'],t);omega=(Rotation.from_quat(qb[[4,5,6,3]])*Rotation.from_quat(qa[[4,5,6,3]]).inv()).as_rotvec()/s['dt'];linear=(qb[:3]-qa[:3])/s['dt'];q[ci]=target;qd[ci]=linear+np.cross(omega,target-qa[:3]);wx,wy,wz=omega;grad[ci]=np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]])
   state.particle_q.assign(q);state.particle_qd.assign(qd);state.mpm.particle_qd_grad.assign(grad)
   if e['drivers']:
    rq=state.body_q.numpy();rv=state.body_qd.numpy()
    for d in e['drivers']:
     tar=sample_driver(d,t0);nex=sample_driver(d,t);rq[names[d['name']]]=np.r_[tar[:3],tar[4:],tar[3]];omega=(Rotation.from_quat(nex[[4,5,6,3]])*Rotation.from_quat(tar[[4,5,6,3]]).inv()).as_rotvec()/s['dt'];rv[names[d['name']]]=np.r_[(nex[:3]-tar[:3])/s['dt'],omega]
    state.body_q.assign(rq);state.body_qd.assign(rv)
   state.clear_forces();solver.step(state,nxt,control,contacts,s['dt']);state,nxt=nxt,state
   # Exact prescribed boundary state at the recording instant; no forces or PD.
   q=state.particle_q.numpy()
   if not np.isfinite(q).all():raise FloatingPointError('Non-finite native particle positions before boundary projection')
   pre_projection['fixed_max_error_m']=max(pre_projection['fixed_max_error_m'],float(np.linalg.norm(q[fi]-v[fi],axis=-1).max())if fi else 0.)
   pre_projection['moving_max_error_m']=max(pre_projection['moving_max_error_m'],max((float(np.linalg.norm(q[c['vertices']]-target_points(c,t),axis=-1).max())for c in e['controls']),default=0.))
   q[fi]=v[fi]
   for c in e['controls']:q[c['vertices']]=target_points(c,t)
   state.particle_q.assign(q)
   if e['drivers']:
    rq=state.body_q.numpy()
    for d in e['drivers']:
     tar=sample_driver(d,t);rq[names[d['name']]]=np.r_[tar[:3],tar[4:],tar[3]]
    state.body_q.assign(rq)
  val=state.particle_q.numpy()[ids];
  if not np.isfinite(val).all():raise FloatingPointError('Non-finite simulated marker positions')
  q=state.particle_q.numpy();boundary_checks.append({'frame':frame,'fixed_max_error_m':float(np.linalg.norm(q[sorted(fixed)]-v[sorted(fixed)],axis=-1).max())if fixed else 0.,'moving_max_error_m':max((float(np.linalg.norm(q[c['vertices']]-target_points(c,frame/30),axis=-1).max())for c in e['controls']),default=0.)})
  points.append(val)
  if frame%30==0:print('FRAME',frame,'/',n,flush=True)
 save_result('newton',a.case,points,'smoke_passed'if a.smoke else'completed',{'version':newton.__version__,'solver':'SolverImplicitMPM','material_mapping':src['material'],'voxel_size_m':cfg.voxel_size,'particles':len(v),'sampling':'Official tet vertices with lumped tetrahedral volume; radius=cuberoot(volume)/2; voxel uses native Config default; elastic tensile_yield_ratio=1','boundary_mapping':'Native zero-mass infinite-mass MPM velocity BC; beginning-of-step poses, rigid skew(omega) APIC gradients, exact prescribed end poses; no PD','boundary_mapping_version':'native_mpm_begin_step_v2','boundary_checks':boundary_checks,'native_before_projection':pre_projection,'boundary_projection_notice':'Post-step Dirichlet position projection applies only to prescribed boundary particles; pre-projection native drift is reported separately. Free particles are never projected.','wall_seconds':time.monotonic()-start})
except Exception as ex:traceback.print_exc();save_result('newton',a.case,points,'numerical_failure'if isinstance(ex,FloatingPointError)else'adapter_error',{'error':str(ex),'traceback':traceback.format_exc(),'boundary_mapping_version':'native_mpm_begin_step_v2','boundary_checks':boundary_checks,'native_before_projection':pre_projection,'wall_seconds':time.monotonic()-start})
