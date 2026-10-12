"""Build renderer caches from recorded SDK state, without loading any engine.

Rigid mesh poses are SDK-recorded forward kinematics; soft surface vertices are
fixed SDK topology indices. Analytical spheres are displayed with official OBJ
assets of identical radius. Presentation meshes never feed back into physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from gauge_eval.paths import ROOT, DATA
from gauge_eval.scenes import obj

def body_meshes(topology, scene, evaluation):
    metadata=json.loads((DATA/'metadata'/evaluation['regime']/(evaluation['task']+'.json')).read_text())
    groups=[];faces=[];body_ids=[];offset=0
    for body in topology['rigid_bodies']:
        v=np.asarray(body['vertices_local'],dtype=float);f=np.asarray(body['surface_triangles'],dtype=int)
        links=np.asarray(body['vertex_link_ids'],dtype=int)
        source=next(b['source']for b in scene['bodies']if b['name']==body['name'])
        if len(v)==6 and len(f)==8 and body['name']in metadata['assets']:
            asset=(DATA/'metadata'/evaluation['regime']/metadata['assets'][body['name']]['model']).resolve()
            official,triangles=obj(asset)
            if source.get('type')=='primitive_sphere':
                radius=source['radius']
                if not np.isclose(np.max(np.linalg.norm(official,axis=1)),radius,rtol=1e-4,atol=1e-7):
                    raise ValueError('Official display sphere does not match analytical radius')
            if len(set(links))!=1:raise ValueError('Analytical display vertices span multiple links')
            v,f,links=official,triangles,np.full(len(official),links[0],dtype=int)
        if not len(v):continue
        groups.append({'name':body['name'],'body_index':body['body_index'],'kind':'rigid','local':v,'links':links,'count':len(v)})
        faces.append(f+offset);body_ids.extend([body['body_index']]*len(v));offset+=len(v)
    for body in topology['soft_bodies']:
        ids=np.asarray(body['surface_vertex_ids'],dtype=int);f=np.asarray(body['surface_triangles'],dtype=int)
        groups.append({'name':body['name'],'body_index':body['body_index'],'kind':'soft','ids':ids,'count':len(ids)})
        faces.append(f+offset);body_ids.extend([body['body_index']]*len(ids));offset+=len(ids)
    return groups,np.concatenate(faces),np.asarray(body_ids,dtype=int),offset

def world_vertices(frame,groups):
    vertices=[]
    for body in groups:
        if body['kind']=='soft':
            v=np.asarray(frame['q'][body['name']],dtype=float).reshape(-1,3)[body['ids']]
        else:
            poses=frame['rigid_link_poses'][body['name']];q=np.asarray(poses['quaternions'],dtype=float)
            rotations=Rotation.from_quat(q[body['links']][:,[1,2,3,0]])
            v=rotations.apply(body['local'])+np.asarray(poses['positions'])[body['links']]
        vertices.append(v)
    result=np.concatenate(vertices)
    if not np.isfinite(result).all():raise ValueError('Nonfinite replay surface')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',default='xie_sap_native_0514_v3');p.add_argument('--result-key',default='xie_sap_native_v3');p.add_argument('--case',action='append');p.add_argument('--output',default='results/sap_native_replay')
    a=p.parse_args();root=ROOT/a.output;root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'configs'/a.profile/'manifest.json').read_text());jobs=[]
    for case in manifest['cases']:
        cid=case['case_id']
        if a.case and cid not in a.case:continue
        result=ROOT/'results'/a.result_key/cid
        if not(result/'status.json').exists():continue
        status=json.loads((result/'status.json').read_text())
        if status['status']=='running':continue
        state=result/'state_q.jsonl'
        if not state.exists() or not state.stat().st_size:continue
        folder=ROOT/'configs'/a.profile/cid
        scene=json.loads((folder/'scene.json').read_text());evaluation=json.loads((folder/'evaluation.json').read_text())
        topology=json.loads((result/'render_topology.json').read_text());groups,faces,ids,nverts=body_meshes(topology,scene,evaluation)
        with state.open()as f:nframes=sum(1 for _ in f)
        if nframes!=status['frames_completed']:raise ValueError('Recording/receipt frame count differs')
        output=root/('full_render_'+cid);cache=output/'geometry';cache.mkdir(parents=True,exist_ok=True)
        verts=np.lib.format.open_memmap(cache/'world_vertices.npy',mode='w+',dtype=np.float32,shape=(nframes,nverts,3))
        frame_ids=[];lo=np.full(3,np.inf);hi=np.full(3,-np.inf)
        with state.open()as f:
            for i,line in enumerate(f):
                frame=json.loads(line);v=world_vertices(frame,groups);verts[i]=v;frame_ids.append(frame['frame'])
                if i<evaluation['frames']:lo=np.minimum(lo,v.min(0));hi=np.maximum(hi,v.max(0))
        verts.flush();del verts
        np.save(cache/'faces.npy',faces);np.save(cache/'vertex_body_ids.npy',ids)
        metadata={'source':str(state),'source_sha256':hashlib.sha256(state.read_bytes()).hexdigest(),'evaluation':str(folder/'evaluation.json'),'frames':frame_ids,'fps':30,'bodies':[{k:b[k]for k in ['name','body_index','kind']}for b in groups],'geometry_source':'SDK render_topology + SDK forward_kinematics + recorded q; official sphere OBJ is presentation only','status':status['status']}
        (cache/'metadata.json').write_text(json.dumps(metadata,indent=2))
        jobs.append({**case,'cache':str(cache),'outdir':str(output),'frames':nframes,'center':((lo+hi)/2).tolist(),'radius':max(float(np.linalg.norm(hi-lo)/2)*1.12,.09),'ground_height':scene['solver'].get('ground_height',0)})
        print('REPLAY',cid,nframes,nverts,flush=True)
    (root/'jobs.json').write_text(json.dumps(jobs,indent=2))

if __name__=='__main__':main()
