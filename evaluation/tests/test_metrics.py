import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gauge_eval.metrics import (rmse,dtw,angle_deficits,face_areas,
    longest_stationary_duration,period_fft,momentum_transfer,energy_loss,normalize)

def test_rmse_feature_norm_and_shape():
    assert rmse([[0,0,0],[0,0,0]],[[3,4,0],[3,4,0]]) == 5
    with pytest.raises(ValueError): rmse([1,2],[1])
    with pytest.raises(ValueError): rmse([np.nan],[0])

def test_dtw_against_exhaustive_paths():
    a=np.array([0.,2.,4.]);b=np.array([1.,3.,3.])
    paths=[]
    def walk(i,j,cost,length):
        cost+=abs(a[i]-b[j]);length+=1
        if (i,j)==(len(a)-1,len(b)-1): paths.append((cost,length));return
        for di,dj in [(1,1),(1,0),(0,1)]:
            if i+di<len(a) and j+dj<len(b):walk(i+di,j+dj,cost,length)
    walk(0,0,0,0)
    best=min(c for c,l in paths)
    expected={c/l for c,l in paths if c==best}
    assert dtw(a,b) in expected
    assert dtw([0,1,2],[0,0,1,2,2]) == 0
    assert dtw([[0,0]],[[3,4]]) == 5

def test_geometry_units_and_rigid_invariance():
    p=np.array([[[0.,0,0],[1,0,0],[1,1,0],[0,1,0]]])
    f=[[0,1,2],[0,2,3]]
    k,omit=angle_deficits(p,f,[0,1,2,3],[],[])
    np.testing.assert_allclose(k,0,atol=1e-14)
    assert omit[0]==0
    q=p@np.array([[0,-1,0],[1,0,0],[0,0,1]])*1000+[8,9,10]
    np.testing.assert_allclose(angle_deficits(q,f,[0,1,2,3],[],[])[0],k,atol=1e-14)
    np.testing.assert_allclose(face_areas(q,f),face_areas(p,f)*1e6)
    with pytest.raises(ValueError):angle_deficits(p,f,[0,1],[],[])

def test_curvature_missing_angles_are_counted():
    p=np.array([[[0.,0,0],[1,0,0],[np.nan,np.nan,np.nan]]])
    k,omit=angle_deficits(p,[[0,1,2]],[0,1,2],[],[])
    np.testing.assert_allclose(k,np.pi/2)
    assert omit[0]==3
    with pytest.raises(ValueError):face_areas(p,[[0,1,2]])

def test_closed_octahedron_gauss_bonnet():
    p=np.array([[[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]])
    faces=[[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]]
    k,omit=angle_deficits(p,faces,[],[],list(range(6)))
    assert k.sum()==pytest.approx(4*np.pi)
    assert omit.sum()==0

def test_lsd_requires_global_range_not_just_small_frame_differences():
    assert longest_stationary_duration(np.zeros(30),30)==1
    z=np.arange(30)*.0001 # velocity 3mm/s, but total drift >0.5mm
    assert longest_stationary_duration(z,30) <= 6/30
    assert longest_stationary_duration(np.arange(30)*.01,30)==0

def test_period_and_task_specific_directions():
    t=np.arange(300)/30
    assert period_fft(np.cos(2*np.pi*t/2),30)==2
    assert momentum_transfer(1,.81)==.9
    assert energy_loss(10,10)==0
    assert energy_loss(10,11)==-.1
    assert normalize(5,10)==.5
    with pytest.raises(ValueError):normalize(0,0)
    with pytest.raises(ValueError):period_fft(np.ones(300),30)
