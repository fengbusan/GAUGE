"""GAUGE equations 2 and 13–17, with explicit conventions for ambiguities.

No engine names, reference scores, fitted constants, or implicit time alignment
belong in this module. Quantities retain the units supplied by the caller.
"""
from collections import deque
import numpy as np
from scipy.spatial.distance import cdist


def trajectory(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError('trajectory must be nonempty finite [time, features]')
    return x


def rmse(a, b):
    a, b = trajectory(a), trajectory(b)
    if a.shape != b.shape:
        raise ValueError('RMSE needs explicitly aligned equal shapes')
    return float(np.sqrt(np.mean(np.sum((a-b)**2, axis=1))))


def dtw(a, b):
    """Minimum SUM Euclidean cost path, then divide by its path length.

    Standard unconstrained steps (1,1), (1,0), (0,1), fixed endpoints.
    Ties prefer diagonal, then vertical, then horizontal. These path conventions
    are not specified by GAUGE; callers must report them. No squared local cost.
    """
    a, b = trajectory(a), trajectory(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError('DTW feature dimensions differ')
    distance = cdist(a, b)
    n, m = distance.shape
    costs = np.full((n+1,m+1), np.inf)
    lengths = np.zeros((n+1,m+1), dtype=np.int32)
    costs[0,0] = 0
    # Anti-diagonals have no internal dependency and can be evaluated together.
    for total in range(2,n+m+1):
        i = np.arange(max(1,total-m), min(n,total-1)+1)
        j = total-i
        candidates = np.stack((costs[i-1,j-1],costs[i-1,j],costs[i,j-1]))
        which = candidates.argmin(axis=0)
        parent_lengths = np.stack((lengths[i-1,j-1],lengths[i-1,j],lengths[i,j-1]))
        indices = np.arange(len(i))
        costs[i,j] = candidates[which,indices] + distance[i-1,j-1]
        lengths[i,j] = parent_lengths[which,indices] + 1
    return float(costs[n,m]/lengths[n,m])


def _mesh(points, faces):
    p = np.asarray(points, dtype=np.float64)
    f = np.asarray(faces)
    if p.ndim != 3 or p.shape[-1] != 3 or not p.size:
        raise ValueError('points must be [time, marker, xyz]')
    if f.ndim != 2 or f.shape[1] != 3 or not f.size or f.dtype.kind not in 'iu':
        raise ValueError('faces must be nonempty integer [face, 3]')
    if f.min() < 0 or f.max() >= p.shape[1] or np.any(np.diff(np.sort(f,axis=1),axis=1)==0):
        raise ValueError('invalid face indices')
    return p, f


def face_areas(points, faces):
    p, f = _mesh(points, faces)
    if not np.isfinite(p).all():
        raise ValueError('area metric has no specified missing-marker policy')
    a, b, c = (p[:,f[:,i]] for i in range(3))
    return .5*np.linalg.norm(np.cross(b-a,c-a),axis=-1)


def angle_deficits(points, faces, corner, edge, inner):
    """Angular deficit in radians, WITHOUT dual-area normalization (eq. 14).

    Paper policy: omit angles with missing vertices or zero-length edges.
    Returns omitted-angle counts too; no silent imputation or frame deletion.
    """
    p, f = _mesh(points, faces)
    groups = [list(corner),list(edge),list(inner)]
    ids = sum(groups,[])
    if sorted(ids) != list(range(p.shape[1])):
        raise ValueError('corner/edge/inner must partition every marker exactly once')
    theta = np.empty(p.shape[1])
    for group, value in zip(groups,[np.pi/2,np.pi,2*np.pi]):
        theta[group] = value
    out = np.broadcast_to(theta,(p.shape[0],p.shape[1])).copy()
    omitted = np.zeros(p.shape[0], dtype=np.int64)
    for k in range(3):
        a = p[:,f[:,(k+1)%3]]-p[:,f[:,k]]
        b = p[:,f[:,(k+2)%3]]-p[:,f[:,k]]
        denom = np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
        valid = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1) & (denom > 0)
        cosine = np.zeros_like(denom)
        np.divide(np.sum(a*b,axis=-1),denom,out=cosine,where=valid)
        angles = np.where(valid,np.arccos(np.clip(cosine,-1,1)),0)
        for face_id, vertex in enumerate(f[:,k]):
            out[:,vertex] -= angles[:,face_id]
        omitted += (~valid).sum(axis=1)
    return out, omitted


def longest_stationary_duration(z, fps, position_threshold=.0005, velocity_threshold=.005):
    """Eq. 15 in metres/seconds; velocity uses numpy.gradient, explicit convention.

    Duration is NUMBER OF SAMPLES / fps, as printed, not (samples-1)/fps.
    """
    z = trajectory(z)[:,0]
    if len(z) < 2 or fps <= 0 or position_threshold < 0 or velocity_threshold < 0:
        raise ValueError('invalid LSD input')
    speed = np.abs(np.gradient(z,1/fps))
    lo, hi = deque(), deque()
    left, best = 0, 0
    for right in range(len(z)):
        if speed[right] > velocity_threshold:
            left = right+1
            lo.clear(); hi.clear()
            continue
        while lo and z[lo[-1]] >= z[right]: lo.pop()
        while hi and z[hi[-1]] <= z[right]: hi.pop()
        lo.append(right); hi.append(right)
        while z[hi[0]]-z[lo[0]] > position_threshold:
            left += 1
            while lo and lo[0] < left: lo.popleft()
            while hi and hi[0] < left: hi.popleft()
        best = max(best,right-left+1)
    return best/fps


def momentum_transfer(height_in, height_out):
    """Eq. 16 returns a fraction; multiply by 100 for Table 3's real MTE %."""
    if not np.isfinite([height_in,height_out]).all() or height_in <= 0 or height_out < 0:
        raise ValueError('invalid first-impact apex heights')
    return float(np.sqrt(height_out/height_in))


def period_fft(signal, fps):
    x = trajectory(signal)[:,0]
    if fps <= 0 or len(x) < 3:
        raise ValueError('invalid period input')
    power = np.abs(np.fft.rfft(x-x.mean()))**2
    power[0] = 0
    if power.max() == 0:
        raise ValueError('stationary signal has no oscillation period')
    return float(1/np.fft.rfftfreq(len(x),1/fps)[power.argmax()])


def energy_loss(start, end):
    if not np.isfinite([start,end]).all() or start <= 0 or end < 0:
        raise ValueError('invalid mechanical energies')
    return float((start-end)/start)


def normalize(value, baseline):
    if not np.isfinite([value,baseline]).all() or baseline <= 0:
        raise ValueError('normalization requires positive baseline; EL must remain raw')
    return float(value/baseline)
