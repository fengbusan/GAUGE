"""Small, engine-independent geometry and motion import utilities."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .paths import ROOT, DATA

def obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            vertices.append(list(map(float, parts[1:4])))
        elif parts[0] == 'f':
            ids = [int(x.split('/')[0]) for x in parts[1:]]
            ids = [i-1 if i > 0 else len(vertices)+i for i in ids]
            faces.extend([[ids[0], ids[j], ids[j+1]] for j in range(1, len(ids)-1)])
    return np.asarray(vertices), np.asarray(faces, dtype=int)

def pose(task, asset, material):
    def resolve(x):
        return x[material] if isinstance(x, dict) else x
    q = Rotation.from_euler('xyz', resolve(task['orientation'][asset]), degrees=True).as_quat()
    return [*resolve(task['translation'][asset]), q[3], *q[:3]]

def write_msh(path, vertices, cells, kind):
    lines = ['$MeshFormat', '2.2 0 8', '$EndMeshFormat', '$Nodes', str(len(vertices))]
    lines += [str(i+1)+' '+' '.join(map(str, p)) for i, p in enumerate(vertices)]
    lines += ['$EndNodes', '$Elements', str(len(cells))]
    lines += [f'{i+1} {kind} 0 '+' '.join(str(int(j)+1) for j in c) for i, c in enumerate(cells)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines+['$EndElements'])+'\n')

def driver(raws, name, anchor):
    positions = [np.stack([r[name][k] for k in 'xyz'], axis=-1)/1000 for r in raws]
    quats = [np.stack([r[name][k] for k in ['qx','qy','qz','qw']], axis=-1) for r in raws]
    n = min(map(len, positions))
    p = np.mean([a[:n] for a in positions], axis=0)
    rotations = Rotation.from_quat(np.array([
        Rotation.from_quat([q[i] for q in quats]).mean().as_quat() for i in range(n)]))
    relative = rotations*rotations[0].inv()
    initial = Rotation.from_quat(np.asarray(anchor)[[4,5,6,3]])
    return dict(name=name, positions=(p-p[0]+np.asarray(anchor[:3])).tolist(),
                quaternions_xyzw=(relative*initial).as_quat().tolist(), fps=30)
