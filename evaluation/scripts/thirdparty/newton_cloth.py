import argparse, json, time, traceback
import numpy as np, newton, warp as wp
from scipy.spatial.transform import Rotation
from common import (
    load_obj_mesh,
    native_marker_mapping,
    ROOT,
    load,
    mesh,
    target_points,
    sample_driver,
    save_result,
    runtime_identity,
)

p = argparse.ArgumentParser()
p.add_argument("case")
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
s, e = load(a.case)
points = []
start = time.monotonic()
try:
    runtime_identity("newton")
    wp.init()
    b = newton.ModelBuilder()
    body = next(x for x in s["bodies"] if x["kind"] == "deformable")
    src = body["source"]
    v, f = mesh(src["path"])
    meta = json.load(
        open(ROOT / "data/GAUGE-Dataset/metadata/textile" / f"{e['task']}.json")
    )
    m = meta["assets"]["textile"]["material"][e["material_parameter_key"]]
    ks = m["stretch stiffness"]
    mu = m["shear stiffness"]
    mu = mu if 2 < ks / mu < 4 else ks / 2.7
    lam = 2 * mu * (ks - 2 * mu) / (4 * mu - ks)
    edges = np.unique(
        np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1),
        axis=0,
    )
    ell = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1).mean()
    bend = m["bend stiffness"] / ell
    print(
        "NATIVE_DEFAULTS",
        json.dumps(
            {
                "particle_radius_m": float(b.default_particle_radius),
                "tri_kd": float(b.default_tri_kd),
                "edge_kd": float(b.default_edge_kd),
                "material_mapping": "GAUGE Appendix A.2.2 equations 10-12 including Ks/2.7",
            }
        ),
        flush=True,
    )
    b.add_cloth_mesh(
        pos=wp.vec3(0),
        rot=wp.quat_identity(),
        scale=1,
        vel=wp.vec3(0),
        vertices=v.tolist(),
        indices=f.ravel().tolist(),
        density=m["density"],
        tri_ke=mu,
        tri_ka=lam,
        edge_ke=bend,
    )
    fixed = set(src.get("fixed_vertices", []))
    moving = set(i for c in e["controls"] for i in c["vertices"])
    for i in fixed | moving:
        b.particle_mass[i] = 0
        b.particle_flags[i] = 0
    names = {}
    for rb in s["bodies"]:
        if rb["kind"] != "rigid":
            continue
        q = rb["initial"]["q"]
        idx = b.add_body(
            xform=wp.transform(q[:3], q[4:] + q[3:4]),
            is_kinematic=True,
            mass=rb["source"]["mass"],
        )
        names[rb["name"]] = idx
        tm = load_obj_mesh(rb["source"]["path"])
        b.add_shape_mesh(
            idx,
            mesh=newton.Mesh(
                np.asarray(tm.vertices, np.float32),
                np.asarray(tm.faces, np.int32).ravel(),
            ),
            cfg=b.ShapeConfig(
                density=0,
                mu=rb["contact_material"]["friction"],
                restitution=rb["contact_material"]["restitution"],
            ),
        )
    b.color(include_bending=True)
    model = b.finalize()
    solver = newton.solvers.SolverVBD(model)
    state = model.state()
    nxt = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    ids, marker_error = native_marker_mapping(
        e, state.particle_q.numpy(), "newton", a.case
    )
    points.append(state.particle_q.numpy()[ids])
    n = 3 if a.smoke else e["rollout_frames"]
    steps = round(1 / 30 / s["dt"])
    for frame in range(1, n):
        for sub in range(steps):
            t = ((frame - 1) * steps + sub + 1) * s["dt"]
            q = state.particle_q.numpy()
            qd = state.particle_qd.numpy()
            for c in e["controls"]:
                target = target_points(c, t)
                ci = c["vertices"]
                qd[ci] = (target - q[ci]) / s["dt"]
                q[ci] = target
            state.particle_q.assign(q)
            state.particle_qd.assign(qd)
            if e["drivers"]:
                rq = state.body_q.numpy()
                rv = state.body_qd.numpy()
                for d in e["drivers"]:
                    tar = sample_driver(d, t)
                    rq[names[d["name"]]] = np.r_[tar[:3], tar[4:], tar[3]]
                    prev = sample_driver(d, t - s["dt"])
                    omega = (
                        Rotation.from_quat(tar[[4, 5, 6, 3]])
                        * Rotation.from_quat(prev[[4, 5, 6, 3]]).inv()
                    ).as_rotvec() / s["dt"]
                    rv[names[d["name"]]] = np.r_[(tar[:3] - prev[:3]) / s["dt"], omega]
                state.body_q.assign(rq)
                state.body_qd.assign(rv)
            state.clear_forces()
            pipeline.collide(state, contacts)
            solver.step(state, nxt, control, contacts, s["dt"])
            state, nxt = nxt, state
        val = state.particle_q.numpy()[ids]
        if not np.isfinite(val).all():
            raise FloatingPointError("Non-finite simulated marker positions")
        points.append(val)
        if frame % 30 == 0:
            print("FRAME", frame, "/", n, flush=True)
    save_result(
        "newton",
        a.case,
        points,
        "smoke_passed" if a.smoke else "completed",
        {
            "version": newton.__version__,
            "solver": "SolverVBD defaults",
            "material_mapping": {
                "tri_ke": mu,
                "tri_ka": lam,
                "edge_ke": bend,
                "areal_density": m["density"],
                "collision_radius_m": float(b.default_particle_radius),
                "physical_thickness_m": m["thickness"],
                "radius_policy": "particle_radius omitted; unmodified Newton 1.3.0 ModelBuilder default",
            },
            "wall_seconds": time.monotonic() - start,
        },
    )
except Exception as ex:
    traceback.print_exc()
    save_result(
        "newton",
        a.case,
        points,
        "numerical_failure" if isinstance(ex, FloatingPointError) else "adapter_error",
        {"error": str(ex), "traceback": traceback.format_exc()},
    )
