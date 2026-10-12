"""Exact retessellation of a verified axis-aligned box surface; no hull collision proxy."""
import itertools
import numpy as np

def exact_box_surface(vertices,faces):
 v=np.asarray(vertices);f=np.asarray(faces);lo=v.min(0);hi=v.max(0);tol=1e-10
 assert np.all(hi>lo)
 # Every original triangle must belong to one of the six exact box planes.
 assert all(any(np.all(np.abs(v[t,axis]-bound)<tol)for axis in range(3)for bound in (lo[axis],hi[axis]))for t in f)
 out=np.array(list(itertools.product(*zip(lo,hi))));tris=[]
 assert all(np.any(np.linalg.norm(v-corner,axis=1)<tol)for corner in out)
 for axis in range(3):
  other=[i for i in range(3)if i!=axis]
  for sign,bound in [(-1,lo[axis]),(1,hi[axis])]:
   corners=[]
   for a,b in [(0,0),(1,0),(1,1),(0,1)]:
    point=lo.copy();point[axis]=bound;point[other[0]]=(lo,hi)[a][other[0]];point[other[1]]=(lo,hi)[b][other[1]];corners.append(int(np.flatnonzero(np.all(out==point,axis=1))[0]))
   for tri in [corners[:3],[corners[0],corners[2],corners[3]]]:
    if np.cross(out[tri[1]]-out[tri[0]],out[tri[2]]-out[tri[0]])[axis]*sign<0:tri[1],tri[2]=tri[2],tri[1]
    tris.append(tri)
 tris=np.asarray(tris)
 def area(p,t):return np.linalg.norm(np.cross(p[t[:,1]]-p[t[:,0]],p[t[:,2]]-p[t[:,0]]),axis=1).sum()/2
 def volume(p,t):return abs(np.einsum('ij,ij->i',p[t[:,0]],np.cross(p[t[:,1]],p[t[:,2]])).sum()/6)
 assert np.isclose(area(v,f),area(out,tris),rtol=1e-10,atol=1e-12)
 assert np.isclose(volume(v,f),volume(out,tris),rtol=1e-10,atol=1e-12)
 return out,tris,{'original_vertices':len(v),'original_triangles':len(f),'vertices':len(out),'triangles':len(tris),'bounds':[lo.tolist(),hi.tolist()],'area_m2':float(area(out,tris)),'volume_m3':float(volume(out,tris)),'validation':'All original faces lie on the same six planes; corners, surface area and enclosed volume are preserved. Triangle-surface collision, not convex-hull collision.'}
