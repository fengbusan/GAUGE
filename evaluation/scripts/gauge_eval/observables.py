"""Engine-blind observables for the literal text and published-table replication.

The table profile is a declared interpretation verified ONLY with real trials.
It is not an official implementation certificate and never uses simulator scores.
Input positions are millimetres; marker order is the frozen metadata order.
"""
import numpy as np
from .metrics import angle_deficits,face_areas

def observe(points,e,profile='paper_formula'):
 p=np.asarray(points,dtype=float)
 if e['regime']=='rigid':
  if p.ndim==3:p=p[:,0]
  return p[:,2:3]if profile=='published_table'and e['task']=='bouncing ball'else p
 if e['regime']=='textile':
  if profile=='published_table'and e['task']=='textile bending':
   a=p[:,2:-2]-p[:,:-4];b=p[:,4:]-p[:,2:-2];denom=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)*np.linalg.norm(a+b,axis=-1)
   if np.any(denom<=0):raise ValueError('Degenerate bending marker chain')
   return 2000*np.linalg.norm(np.cross(a,b),axis=-1)/denom
  return angle_deficits(p,e['marker_faces'],**e['marker_groups'])[0]
 faces=np.array(e['marker_faces'],dtype=int)
 if e['task']!='foam bending'and np.array_equal(faces[:12],faces[12:]):faces[12:]+=14
 return face_areas(p,faces)
