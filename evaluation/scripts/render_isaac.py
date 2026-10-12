"""Isaac RTX batch renderer for canonical world-vertex caches (no physics step)."""
from pathlib import Path
import argparse,json,os,subprocess,time,sys
ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--follow',action='store_true');args,_=ap.parse_known_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','width':960,'height':720,'anti_aliasing':0,'sync_loads':True})
import numpy as np
from pxr import UsdGeom,UsdLux,UsdShade,Gf,Sdf,Vt
import omni.usd
import omni.replicator.core as rep
import carb
carb.settings.get_settings().set('/app/asyncRendering',False)
carb.settings.get_settings().set('/rtx/post/aa/op',0)
rep.orchestrator.set_capture_on_play(False)
jobs=json.load(open(args.jobs))
if args.smoke:jobs=jobs[:1]
def queued_jobs():
 seen=set()
 while True:
  current=json.load(open(args.jobs))if args.follow else jobs
  for job in current:
   if job['case_id']not in seen:
    seen.add(job['case_id']);yield job
  if not args.follow:break
  if Path(args.jobs).with_suffix('.done').exists()and all(j['case_id']in seen for j in json.load(open(args.jobs))):break
  time.sleep(3)
for j in queued_jobs():
 out=Path(j['outdir']);manifest=out/'render_manifest.json'
 if manifest.exists() and not args.smoke:continue
 cache=Path(j['cache']);meta=json.load(open(cache/'metadata.json'));verts=np.load(cache/'world_vertices.npy',mmap_mode='r');faces=np.load(cache/'faces.npy');ids=np.load(cache/'vertex_body_ids.npy')
 omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage();UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1)
 center=np.array(j['center']);radius=j['radius'];meshes=[]
 palette=[(.08,.48,.43),(.82,.37,.14),(.30,.48,.65),(.66,.54,.28),(.49,.33,.60)]
 for b in meta['bodies']:
  bid=b['body_index'];ix=np.flatnonzero(ids==bid);tri=faces[np.all(ids[faces]==bid,axis=1)];mapping=np.full(len(ids),-1,dtype=np.int32);mapping[ix]=np.arange(len(ix));tri=mapping[tri]
  mesh=UsdGeom.Mesh.Define(stage,f'/World/Body_{bid}');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(verts[0,ix],dtype=np.float32)));mesh.CreateFaceVertexCountsAttr([3]*len(tri));mesh.CreateFaceVertexIndicesAttr(tri.reshape(-1).tolist());mesh.CreateDoubleSidedAttr(True);mesh.CreateSubdivisionSchemeAttr('none')
  color=(.29,.34,.36)if b['name']in ['ground','ground_mesh']else palette[bid%len(palette)]
  if b['name']=='cube':color=(.90,.31,.06)
  if b['name']=='plate':color=(.21,.41,.53)
  mat=UsdShade.Material.Define(stage,f'/World/Materials/M_{bid}');shader=UsdShade.Shader.Define(stage,f'/World/Materials/M_{bid}/Shader');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.5);shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.15 if b['kind']=='rigid'else 0);mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat);meshes.append((mesh,ix))
 # Display passive joint constraints from the frozen scene, without adding physics.
 links=[];evaluation=Path(meta['evaluation']) if meta.get('evaluation') else Path(meta['source']).parents[3]/'configs'/Path(meta['source']).parents[1].name/j['case_id']/'evaluation.json'
 articulated=json.load(open(evaluation)).get('articulated',[])if evaluation.exists()else[]
 for index,link in enumerate(articulated):
  body=next(b for b in meta['bodies']if b['name']==link['name']);ix=np.flatnonzero(ids==body['body_index']);curve=UsdGeom.BasisCurves.Define(stage,f'/Presentation/Joint_{index}');curve.CreateTypeAttr('linear');curve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray([link['pivot'],np.asarray(verts[0,ix]).mean(axis=0)],dtype=np.float32)));curve.CreateCurveVertexCountsAttr([2]);curve.CreateWidthsAttr([.0008]);curve.SetWidthsInterpolation('constant');curve.CreateDisplayColorAttr([(.12,.15,.18)]);links.append((curve,ix,link['pivot']))
 # A neutral presentation floor at the scene's actual support height, never a physics collider.
 z=float(j['ground_height']);extent=max(radius*8,2);floor=UsdGeom.Mesh.Define(stage,'/Presentation/Floor');floor.CreatePointsAttr([(-extent,-extent,z),(extent,-extent,z),(extent,extent,z),(-extent,extent,z)]);floor.CreateFaceVertexCountsAttr([4]);floor.CreateFaceVertexIndicesAttr([0,1,2,3]);floor.CreateDisplayColorAttr([(.32,.36,.38)]);floor.CreateDoubleSidedAttr(True)
 dome=UsdLux.DomeLight.Define(stage,'/Lighting/Fill');dome.CreateIntensityAttr(700);dome.CreateColorAttr((.8,.88,1))
 key=UsdLux.DistantLight.Define(stage,'/Lighting/Key');key.CreateIntensityAttr(2500);key.CreateAngleAttr(12);UsdGeom.Xformable(key).AddRotateXYZOp().Set((25,-30,-35))
 cam=UsdGeom.Camera.Define(stage,'/Camera');cam.CreateClippingRangeAttr((.001,1000));cam.CreateFocalLengthAttr(36);cam.CreateHorizontalApertureAttr(36);cam.CreateVerticalApertureAttr(27)
 direction=np.array([.75,-1.15,.8]);direction/=np.linalg.norm(direction);eye=center+direction*radius*3.35
 matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*center),Gf.Vec3d(0,0,1)).GetInverse();cam.AddTransformOp().Set(matrix)
 rp=rep.create.render_product('/Camera',(960,720));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([rp])
 for _ in range(8):app.update()
 rep.orchestrator.step(rt_subframes=4,delta_time=0.0,pause_timeline=True)
 indices=list(range(min(5,len(verts))))if args.smoke else list(range(len(verts)))
 video=out/('smoke.mp4'if args.smoke else'simulation.mp4');proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
 started=time.monotonic()
 try:
  for i in indices:
   for mesh,ix in meshes:mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(verts[i,ix],dtype=np.float32)))
   for curve,ix,pivot in links:curve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray([pivot,np.asarray(verts[i,ix]).mean(axis=0)],dtype=np.float32)))
   rep.orchestrator.step(rt_subframes=1,delta_time=0.0,pause_timeline=True)
   image=np.asarray(rgb.get_data());assert image.shape[:2]==(720,960),image.shape;proc.stdin.write(np.ascontiguousarray(image[:,:,:3]).tobytes())
   if i in [0,len(verts)-1]:
    from PIL import Image
    Image.fromarray(image[:,:,:3]).save(out/f'keyframe_{i:05}.png')
   if i%300==0:print('PROGRESS',j['case_id'],i,len(verts),flush=True)
 finally:proc.stdin.close()
 assert proc.wait()==0
 rgb.detach([rp]);rp.destroy()
 receipt={'case_id':j['case_id'],'renderer':'Isaac Sim 6.0.1 RTX RaytracedLighting','physics':'none: recorded SDK trajectory replay','runtime_bundle':'Isaac Sim replay host (physics disabled); independent of engine SDK release','host_python':sys.executable,'input':meta['source'],'source_sha256':meta['source_sha256'],'frames':[meta['frames'][i]for i in indices],'joint_visualization':'Frozen official pivot to recorded bob center; display only','fps':30,'resolution':[960,720],'camera':{'center':center.tolist(),'radius':radius,'eye':eye.tolist()},'video':str(video),'elapsed_seconds':time.monotonic()-started}
 (out/('smoke_manifest.json'if args.smoke else'render_manifest.json')).write_text(json.dumps(receipt,indent=2));print('COMPLETED',j['case_id'],len(indices),round(receipt['elapsed_seconds'],1),flush=True)
app.close()
