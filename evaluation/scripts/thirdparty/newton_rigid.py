import argparse, time, traceback, xml.etree.ElementTree as ET
import numpy as np
import newton, warp as wp
from scipy.spatial.transform import Rotation
from common import load_obj_mesh, load, sample_driver, save_result, runtime_identity

p = argparse.ArgumentParser()
p.add_argument("case")
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
s, e = load(a.case)
points = []
driver_errors = []
start = time.monotonic()
try:
    runtime_identity("newton")
    wp.init()
    b = newton.ModelBuilder()
    names = {}
    drive_joints = {}
    driven = {d["name"] for d in e["drivers"]}
    joint_angles = []

    def xf(q):
        return wp.transform(q[:3], q[4:7] + q[3:4])

    for body in s["bodies"]:
        src = body["source"]
        name = body["name"]
        q = body["initial"]["q"]
        art = next((x for x in e.get("articulated", []) if x["name"] == name), None)
        if art:
            angle = q[0]
            rot = Rotation.from_rotvec(np.array(art["axis"]) * angle)
            pos = np.array(art["pivot"]) + rot.apply(art["offset"])
            qr = rot.as_quat()
            q = [*pos, qr[3], *qr[:3]]
            bob = ET.parse(src["path"]).getroot().find("link[@name='bob']")
            mass = float(bob.find("inertial/mass").get("value"))
            radius = float(bob.find(".//sphere").get("radius"))
            inertia = np.eye(3) * 0.4 * mass * radius**2
        else:
            mass = src["mass"]
            inertia = np.array(
                src.get(
                    "inertia", np.eye(3) * 0.4 * mass * src.get("radius", 0.01) ** 2
                )
            )
        dynamic = body["motion"] == "dynamic" or name in driven
        idx = (
            b.add_link(
                xform=xf(q),
                mass=mass,
                inertia=wp.mat33(inertia),
                com=wp.vec3(src.get("center_of_mass", [0, 0, 0])),
                lock_inertia=True,
                is_kinematic=name in driven,
                label=name,
            )
            if dynamic
            else -1
        )
        names[name] = idx
        mat = body["contact_material"]
        cfg = b.ShapeConfig(
            density=0, mu=mat["friction"], restitution=mat.get("restitution", 0)
        )
        local = wp.transform_identity() if dynamic else xf(q)
        if art or src["type"] in ("sphere", "primitive_sphere"):
            b.add_shape_sphere(
                idx, xform=local, radius=radius if art else src["radius"], cfg=cfg
            )
        elif src["type"] == "mesh_obj":
            mesh = load_obj_mesh(src["path"])
            b.add_shape_mesh(
                idx,
                xform=local,
                mesh=newton.Mesh(
                    np.array(mesh.vertices, dtype=np.float32),
                    np.array(mesh.faces, dtype=np.int32).ravel(),
                ),
                cfg=cfg,
            )
        else:
            raise ValueError(src["type"])
        if art:
            j = b.add_joint_revolute(
                -1,
                idx,
                parent_xform=wp.transform(art["pivot"], wp.quat_identity()),
                child_xform=wp.transform(
                    (-np.array(art["offset"])).tolist(), wp.quat_identity()
                ),
                axis=art["axis"],
            )
            b.add_articulation([j])
            b.joint_q[-1] = angle
        elif dynamic:
            j = b.add_joint_free(idx)
            b.add_articulation([j])
            drive_joints[name] = j
    if s["solver"].get("ground_enabled"):
        gm = s["solver"]["ground_material"]
        b.add_ground_plane(
            height=s["solver"]["ground_height"],
            cfg=b.ShapeConfig(mu=gm["friction"], restitution=gm["restitution"]),
        )
    model = b.finalize()
    solver = newton.solvers.SolverMuJoCo(model)
    state = model.state()
    nxt = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    observed = (
        [names[x["name"]] for x in e["articulated"]]
        if "articulated" in e
        else [names[e["body"]]]
    )

    def obs():
        return state.body_q.numpy()[observed, :3].copy()

    points.append(obs())
    n = 3 if a.smoke else e["rollout_frames"]
    steps = round(1 / 30 / s["dt"])
    for frame in range(1, n):
        for sub in range(steps):
            t = ((frame - 1) * steps + sub + 1) * s["dt"]
            if e["drivers"]:
                q = state.body_q.numpy()
                jq = state.joint_q.numpy()
                jv = state.joint_qd.numpy()
                qs = model.joint_q_start.numpy()
                vs = model.joint_qd_start.numpy()
                for d in e["drivers"]:
                    target = sample_driver(d, t)
                    q[names[d["name"]]] = np.r_[target[:3], target[4:7], target[3]]
                    j = drive_joints[d["name"]]
                    jq[qs[j] : qs[j] + 7] = q[names[d["name"]]]
                    prev = sample_driver(d, t - s["dt"])
                    omega = (
                        Rotation.from_quat(target[[4, 5, 6, 3]])
                        * Rotation.from_quat(prev[[4, 5, 6, 3]]).inv()
                    ).as_rotvec() / s["dt"]
                    jv[vs[j] : vs[j] + 6] = np.r_[
                        (target[:3] - prev[:3]) / s["dt"], omega
                    ]
                state.body_q.assign(q)
                state.joint_q.assign(jq)
                state.joint_qd.assign(jv)
            state.clear_forces()
            solver.step(state, nxt, control, None, s["dt"])
            state, nxt = nxt, state
        val = obs()
        if not np.isfinite(val).all():
            raise FloatingPointError("Non-finite simulated marker positions")
        points.append(val)
        for d in e["drivers"]:
            target = sample_driver(d, frame / 30)
            actual = state.body_q.numpy()[names[d["name"]]]
            driver_errors.append(float(np.linalg.norm(actual[:3] - target[:3])))
            assert driver_errors[-1] < 1e-5, "kinematic apparatus target not applied"
        if frame % 30 == 0:
            print("FRAME", frame, "/", n, flush=True)
    save_result(
        "newton",
        a.case,
        points,
        "smoke_passed" if a.smoke else "completed",
        {
            "version": newton.__version__,
            "solver": "SolverMuJoCo native defaults",
            "driver_max_position_error_m": max(driver_errors, default=0.0),
            "wall_seconds": time.monotonic() - start,
            "restitution_note": "ShapeConfig authored; MuJoCo contact law does not guarantee Newtonian restitution equivalence",
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
