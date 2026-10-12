import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gauge_eval.scoring import competition_ranks, matches_printed, rank_summary

def test_recording_reuse_rejects_changed_physics_drivers_and_windows():
    from gauge_eval.scoring import validate_reused_evaluation
    import copy
    actual={'profile':'previous','scene_sha256':'abc','frames':380,'rollout_frames':500,'marker_vertex_ids':[2,4], 'drivers':[{'positions':[[0,0,0],[1,0,0]]}]}
    target={**actual,'profile':'normalized','rigid_topology_normalization':[]}
    validate_reused_evaluation(actual,target)
    for field,value in [('scene_sha256','different'),('frames',300),('marker_vertex_ids',[4,2]),('drivers',[])]:
        altered=copy.deepcopy(target);altered[field]=value
        with pytest.raises(ValueError,match='different physical/evaluation inputs'):
            validate_reused_evaluation(actual,altered)
from gauge_eval.observables import observe
from run_xie import sample_driver

def test_ranking_rounding_ties_and_printed_precision():
    assert competition_ranks({'a':1.004,'b':1.,'c':1.02},2)=={'a':1,'b':1,'c':3}
    assert competition_ranks({'a':1.005,'b':1.,'c':1.02},2)=={'a':2,'b':1,'c':3}
    assert matches_printed(1.2649,'1.26')
    assert not matches_printed(1.2651,'1.26')

def test_observable_profiles_and_units():
    points=np.array([[[3.,4.,12.]]])
    e={'regime':'rigid','task':'bouncing ball'}
    np.testing.assert_equal(observe(points,e,'published_table'),[[12]])
    np.testing.assert_equal(observe(points,e,'paper_formula'),[[3,4,12]])
    # Three points per staggered chain on a 1 m radius circle: curvature 1/m.
    angles=np.linspace(0,np.pi/2,3)
    chain=np.stack([np.cos(angles)*1000,np.sin(angles)*1000,np.zeros(3)],axis=-1)
    p=np.repeat(chain,2,axis=0)[None]
    np.testing.assert_allclose(observe(p,{'regime':'textile','task':'textile bending'},'published_table'),1)

def test_driver_endpoint_stops_and_one_sample_is_static():
    d={'positions':[[0,0,0],[1,0,0]],'quaternions_xyzw':[[0,0,0,1]]*2,'fps':30}
    q,v=sample_driver(d,2)
    np.testing.assert_equal(q[:3],[1,0,0]);np.testing.assert_equal(v,0)
    q,v=sample_driver({**d,'positions':[[0,0,0]],'quaternions_xyzw':[[0,0,0,1]]},0)
    np.testing.assert_equal(v,0)

def test_missing_published_peer_is_excluded_not_zero_ranked():
    row={'case_id':'case','regime':'rigid','ranking_eligible':True,
         'baseline_audit':{'RMSE':{'mean_matches':True,'std_matches':True}},
         'published':{'RMSE':{'engines':{'IsaacSim':'1.00','Genesis':'—','Newton':'2.00'}}},
         'engines':{'xiephysic':{'scores':{'RMSE':{'normalized':.5}}}}}
    assert rank_summary([row],'published')['RMSE']['all']['n']==0
    row['published']['RMSE']['engines']['Genesis']='3.00'
    result=rank_summary([row],'published')['RMSE']
    assert result['all']['engines']['xiephysic']['first_pct']==100
    assert result['rigid']['n']==1 and result['textile']['n']==0

def test_serialized_integer_does_not_reduce_official_display_precision():
    row={'case_id':'case','regime':'rigid','ranking_eligible':True,
         'baseline_audit':{'RMSE':{'mean_matches':True,'std_matches':True}},
         'published':{'RMSE':{'engines':{'IsaacSim':'1','Genesis':'0.83','Newton':'1.15'}}},
         'engines':{'xiephysic':{'scores':{'RMSE':{'normalized':0.81045}}}}}
    result=rank_summary([row],'published')['RMSE']['all']['cases'][0]
    assert result['decimal_places']==2
    assert result['ranks']=={'xiephysic':1,'isaac':3,'genesis':2,'newton':4}
