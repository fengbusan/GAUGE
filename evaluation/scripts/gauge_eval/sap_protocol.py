"""Native GAUGE boundary mapping and one explicitly frozen SAP budget."""
import copy
import itertools
import numpy as np
from scipy.spatial import ConvexHull, QhullError

SAP_PARAMETERS = {
    'strategy': 'frozen_sap',
    'max_inner_iterations': 32,
    'max_line_search_iterations': 12,
    'swept_point_contacts': True,
    'endpoint_point_ccd': True,
    'inner_relative_tolerance': 1e-4,
    'inner_velocity_tolerance': 1e-4,
}


def freeze_solver(solver):
    result = copy.deepcopy(solver)
    result.update(integrator='bdf1', contact_solver={'model': 'velocity_primal'},
                  reduced_primal_recovery={'enabled': False},
                  max_newton_iters=32, max_line_search_iters=12,
                  velocity_primal=copy.deepcopy(SAP_PARAMETERS),
                  linear_solver={'type': 'auto', 'convergence': 'fixed_iterations',
                                 'preconditioner': {'type': 'block_jacobi'}})
    return result


def native_geometry_kind(vertices, faces):
    """Select an equivalent native solid; never hull nonconvex input.

    Admission checks all triangle support planes, area and volume. No simulator
    trajectory or score participates in the representation choice.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1).sum()/2
    volume = abs(np.einsum('ij,ij->i', tri[:,0], np.cross(tri[:,1], tri[:,2])).sum())/6
    try:
        hull = ConvexHull(v)
    except QhullError:
        return 'triangle_surface', {}
    on_hull = all(np.any(np.all(np.abs(t@hull.equations[:,:3].T+hull.equations[:,3]) < 1e-9, axis=0)) for t in tri)
    if not (on_hull and np.isclose(volume,hull.volume,rtol=1e-6,atol=1e-15) and np.isclose(area,hull.area,rtol=1e-6,atol=1e-12)):
        return 'triangle_surface', {}
    lo, hi = v.min(0), v.max(0)
    corners = np.array(list(itertools.product(*zip(lo,hi))))
    exact_box = (np.allclose(lo+hi,0,atol=1e-10,rtol=0) and
                 all(np.any(np.linalg.norm(v-p,axis=1)<1e-10)for p in corners) and
                 all(any(np.all(np.abs(t[:,axis]-bound)<1e-10)for axis in range(3)for bound in [lo[axis],hi[axis]])for t in tri))
    info = {'area_m2':float(area),'volume_m3':float(volume),
            'convex_area_m2':float(hull.area),'convex_volume_m3':float(hull.volume),
            'all_original_faces_on_convex_boundary':True}
    if exact_box:
        info['half_extents'] = ((hi-lo)/2).tolist()
        return 'primitive_box', info
    return 'convex_hull', info
