import numpy as np
from gauge_eval.sap_protocol import freeze_solver, native_geometry_kind, SAP_PARAMETERS

def test_budget_override_retains_material_clearance_and_linear_policy():
    original={'d_hat':.001,'friction_mu':.7,'ground_restitution':.6,'max_newton_iters':80}
    result=freeze_solver(original)
    assert original=={'d_hat':.001,'friction_mu':.7,'ground_restitution':.6,'max_newton_iters':80}
    assert result['d_hat']==.001 and result['friction_mu']==.7 and result['ground_restitution']==.6
    assert result['velocity_primal']==SAP_PARAMETERS
    assert result['max_newton_iters']==32 and result['max_line_search_iters']==12
    assert result['reduced_primal_recovery']=={'enabled':False}

def test_convex_admission_never_fills_a_concavity_or_moves_a_box():
    from itertools import product
    v=np.array(list(product([-1.,1.],repeat=3)))
    # Same box topology used by every backend's exact shape test.
    faces=np.array([[0,2,3],[0,3,1],[4,5,7],[4,7,6],[0,1,5],[0,5,4],[2,6,7],[2,7,3],[0,4,6],[0,6,2],[1,3,7],[1,7,5]])
    kind,info=native_geometry_kind(v,faces)
    assert kind=='primitive_box'and info['half_extents']==[1,1,1]
    # A recessed corner is nonconvex; a hull would erase real geometry.
    dented=v.copy();dented[7]=[0,0,0]
    assert native_geometry_kind(dented,faces)[0]=='triangle_surface'
    # A translated exact box cannot use a centered primitive silently.
    assert native_geometry_kind(v+[.1,0,0],faces)[0]=='convex_hull'
