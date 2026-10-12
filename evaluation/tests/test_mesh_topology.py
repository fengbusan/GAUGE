import numpy as np
import pytest
from gauge_eval.mesh_topology import split_disconnected_vertex_fans


def test_touching_solids_keep_exact_surface_and_face_order():
    # Two closed tetrahedra share only one authored vertex ID.
    v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]], dtype=float)
    f = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3],[0,4,5],[0,6,4],[0,5,6],[4,6,5]])
    vv, ff, splits = split_disconnected_vertex_fans(v, f)
    assert len(vv) == 8 and len(splits) == 1
    assert np.array_equal(v[f], vv[ff])
    assert set(ff[:4].flat).isdisjoint(set(ff[4:].flat))
    v2, f2, again = split_disconnected_vertex_fans(vv, ff)
    assert not again and np.array_equal(v2,vv) and np.array_equal(f2,ff)


def test_connected_sheet_is_unchanged_and_invalid_edge_is_rejected():
    v = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1]],dtype=float)
    f = np.array([[0,1,2],[1,3,2]])
    vv, ff, splits = split_disconnected_vertex_fans(v,f)
    assert not splits and np.array_equal(vv,v) and np.array_equal(ff,f)
    with pytest.raises(ValueError,match='Nonmanifold edges'):
        split_disconnected_vertex_fans(v,np.array([[0,1,2],[0,1,3],[0,1,4]]))
