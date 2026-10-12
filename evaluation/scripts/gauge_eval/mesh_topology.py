"""Lossless rigid-surface normalization; never weld or approximate geometry."""
import numpy as np


def split_disconnected_vertex_fans(vertices, faces):
    """Separate face fans sharing only a vertex, retaining face order/coordinates.

    This resolves pinched OBJ vertex identities without repairing nonmanifold
    edges, removing faces, changing winding, or replacing the surface by a hull.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=int)
    incident = [[] for _ in vertices]
    edges = {}
    for i, triangle in enumerate(faces):
        if len(set(triangle)) != 3 or min(triangle) < 0 or max(triangle) >= len(vertices):
            raise ValueError('Invalid triangle indices')
        for vertex in triangle:
            incident[vertex].append(i)
        for a, b in zip(triangle, np.roll(triangle, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    if any(len(fs) > 2 for fs in edges.values()):
        raise ValueError('Nonmanifold edges require explicit asset review')
    result = faces.copy()
    points = list(vertices)
    splits = []
    for vertex, face_ids in enumerate(incident):
        remaining = set(face_ids)
        fan_count = 0
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            stack, fan = [seed], []
            while stack:
                i = stack.pop()
                fan.append(i)
                for other in faces[i]:
                    if other == vertex:
                        continue
                    for j in edges[tuple(sorted((vertex, int(other))))]:
                        if j in remaining:
                            remaining.remove(j)
                            stack.append(j)
            if fan_count:
                new_id = len(points)
                points.append(vertices[vertex].copy())
                for i in fan:
                    result[i, faces[i] == vertex] = new_id
                splits.append({'source_vertex': vertex, 'new_vertex': new_id, 'faces': sorted(fan)})
            fan_count += 1
    points = np.asarray(points)
    if not np.array_equal(vertices[faces], points[result]):
        raise ValueError('Topology normalization changed triangle coordinates')
    return points, result, splits
