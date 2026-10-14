"""Native PhysX rigid baseline. No XiePhysic adapter is imported."""

import argparse, json, time, traceback, os, xml.etree.ElementTree as ET
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("case")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "disable_viewport_updates": True})
import numpy as np
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
import omni.usd, carb
from common import ROOT, load, sample_driver, save_result, runtime_identity
from scipy.spatial.transform import Rotation

s, e = load(args.case)
out = ROOT / os.environ.get("THIRDPARTY_OUTPUT_ROOT", "results") / "isaac" / args.case
out.mkdir(parents=True, exist_ok=True)
points = []
driver_checks = []
started = time.monotonic()
sim = None
try:
    runtime_identity("isaac")
    assert e["regime"] == "rigid"
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    ps = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    ps.CreateTimeStepsPerSecondAttr(180)
    names = {}
    driven = {d["name"] for d in e["drivers"]}

    def transform(prim, q):
        x = UsdGeom.Xformable(prim)
        if not prim.GetAttribute("xformOp:translate"):
            x.AddTranslateOp()
        if not prim.GetAttribute("xformOp:orient"):
            x.AddOrientOp()
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, q[:3])))
        prim.GetAttribute("xformOp:orient").Set(
            Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[4:7])))
        )

    def material(geom, name, friction, restitution):
        from pxr import UsdShade

        mat = UsdShade.Material.Define(stage, "/World/materials/" + name)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(float(friction))
        api.CreateDynamicFrictionAttr(float(friction))
        api.CreateRestitutionAttr(float(restitution))
        UsdShade.MaterialBindingAPI.Apply(geom).Bind(mat, materialPurpose="physics")

    for b in s["bodies"]:
        name = b["name"].replace("-", "_")
        path = "/World/" + name
        names[b["name"]] = path
        body = UsdGeom.Xform.Define(stage, path).GetPrim()
        src = b["source"]
        q = b["initial"]["q"]
        art = next(
            (a for a in e.get("articulated", []) if a["name"] == b["name"]), None
        )
        if art:
            rot = Rotation.from_rotvec(np.array(art["axis"]) * q[0])
            p = np.array(art["pivot"]) + rot.apply(art["offset"])
            r = rot.as_quat()
            q = [*p, r[3], *r[:3]]
            root = ET.parse(src["path"]).getroot()
            bob = root.find("link[@name='bob']")
            mass = float(bob.find("inertial/mass").get("value"))
            radius = float(bob.find(".//sphere").get("radius"))
            geom = UsdGeom.Sphere.Define(stage, path + "/collider")
            geom.CreateRadiusAttr(radius)
        elif src["type"] == "mesh_obj":
            vertices = []
            faces = []
            for line in Path(src["path"]).read_text().splitlines():
                a = line.split()
                if not a:
                    continue
                if a[0] == "v":
                    vertices.append(list(map(float, a[1:4])))
                if a[0] == "f":
                    ids = [int(x.split("/")[0]) - 1 for x in a[1:]]
                    for i in range(1, len(ids) - 1):
                        faces.append([ids[0], ids[i], ids[i + 1]])
            geom = UsdGeom.Mesh.Define(stage, path + "/collider")
            geom.CreateSubdivisionSchemeAttr("none")
            geom.CreatePointsAttr(vertices)
            geom.CreateFaceVertexCountsAttr([3] * len(faces))
            geom.CreateFaceVertexIndicesAttr(np.array(faces).ravel().tolist())
            mass = src["mass"]
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim())
            mesh_api.CreateApproximationAttr(
                "convexHull" if b["motion"] == "dynamic" else "none"
            )
            vv = np.asarray(vertices)
            lo = vv.min(0)
            hi = vv.max(0)
            corners = np.array(
                [
                    [x, y, z]
                    for x in [lo[0], hi[0]]
                    for y in [lo[1], hi[1]]
                    for z in [lo[2], hi[2]]
                ]
            )
            unique = np.unique(vv, axis=0)
            tri = vv[np.asarray(faces)]
            volume = abs(
                np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
                / 6
            )
            exact_box = (
                len(unique) == 8
                and np.allclose(unique, corners, atol=1e-9, rtol=0)
                and np.isclose(volume, np.prod(hi - lo), rtol=1e-6, atol=1e-14)
            )
            if exact_box:
                stage.RemovePrim(path + "/collider")
                geom = UsdGeom.Cube.Define(stage, path + "/collider")
                geom.CreateSizeAttr(1.0)
                gx = UsdGeom.Xformable(geom)
                gx.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
                gx.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        elif src["type"] in ("sphere", "primitive_sphere"):
            geom = UsdGeom.Sphere.Define(stage, path + "/collider")
            geom.CreateRadiusAttr(src["radius"])
            mass = src["mass"]
        else:
            raise ValueError("unsupported source " + src["type"])
        transform(body, q)
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        material(
            geom.GetPrim(),
            name,
            b["contact_material"].get("friction", 0.5),
            b["contact_material"].get("restitution", 0),
        )
        if b["motion"] == "dynamic" or b["name"] in driven:
            api = UsdPhysics.RigidBodyAPI.Apply(body)
            api.CreateKinematicEnabledAttr(b["name"] in driven)
            ma = UsdPhysics.MassAPI.Apply(body)
            ma.CreateMassAttr(mass)
            if "inertia" in src:
                values, vectors = np.linalg.eigh(np.asarray(src["inertia"]))
                vectors[:, 2] *= np.linalg.det(vectors)
                qr = Rotation.from_matrix(vectors).as_quat()
                ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, values)))
                ma.CreatePrincipalAxesAttr(
                    Gf.Quatf(float(qr[3]), Gf.Vec3f(*map(float, qr[:3])))
                )
                ma.CreateCenterOfMassAttr(Gf.Vec3f(*src["center_of_mass"]))
        if art:
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/joint_" + name)
            joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
            joint.CreateAxisAttr("XYZ"[int(np.argmax(np.abs(art["axis"])))])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*art["pivot"]))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*(-np.array(art["offset"]))))
            joint.CreateLocalRot0Attr(Gf.Quatf(1))
            joint.CreateLocalRot1Attr(Gf.Quatf(1))
    if s["solver"].get("ground_enabled"):
        z = s["solver"]["ground_height"]
        g = UsdGeom.Mesh.Define(stage, "/World/ground")
        g.CreatePointsAttr([(-10, -10, z), (10, -10, z), (10, 10, z), (-10, 10, z)])
        g.CreateFaceVertexCountsAttr([4])
        g.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        gm = s["solver"]["ground_material"]
        material(g.GetPrim(), "ground", gm["friction"], gm["restitution"])
    stage.GetRootLayer().Export(str(out / "input.usda"))
    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    from omni.physx import get_physx_interface

    sim = get_physx_interface()

    def observe():
        cache = UsdGeom.XformCache()
        bodies = (
            [a["name"] for a in e["articulated"]] if "articulated" in e else [e["body"]]
        )
        return np.array(
            [
                list(
                    cache.GetLocalToWorldTransform(
                        stage.GetPrimAtPath(names[n])
                    ).ExtractTranslation()
                )
                for n in bodies
            ]
        )

    points.append(observe())
    sim.start_simulation()
    import omni.physics.tensors

    tv = omni.physics.tensors.create_simulation_view(
        "numpy", stage_id=omni.usd.get_context().get_stage_id()
    )
    views = {name: tv.create_rigid_body_view(path) for name, path in names.items()}
    n = 3 if args.smoke else e["rollout_frames"]
    for frame in range(1, n):
        for sub in range(6):
            t = ((frame - 1) * 6 + sub + 1) * s["dt"]
            for d in e["drivers"]:
                q = sample_driver(d, t)
                views[d["name"]].set_kinematic_targets(
                    np.array([np.r_[q[:3], q[4:], q[3]]], np.float32),
                    np.array([0], np.int32),
                )
            sim.update_simulation(s["dt"], t - s["dt"])
            sim.update_transformations(False, True, True, False)
        p = observe()
        assert np.isfinite(p).all()
        points.append(p)
        for d in e["drivers"]:
            mat = UsdGeom.XformCache().GetLocalToWorldTransform(
                stage.GetPrimAtPath(names[d["name"]])
            )
            q = mat.ExtractRotationQuat()
            actual = np.r_[mat.ExtractTranslation(), q.GetReal(), q.GetImaginary()]
            target = sample_driver(d, frame / 30)
            driver_checks.append(
                {
                    "frame": frame,
                    "name": d["name"],
                    "actual": actual.tolist(),
                    "target": target.tolist(),
                    "physics_pose_xyzw": views[d["name"]].get_transforms().tolist(),
                    "physics_velocity": views[d["name"]].get_velocities().tolist(),
                    "observed_actor_pose": views[e["body"]].get_transforms().tolist(),
                }
            )
        if frame % 30 == 0:
            print("FRAME", frame, "/", n, flush=True)
    (out / "driver_checks.json").write_text(json.dumps(driver_checks))
    save_result(
        "isaac",
        args.case,
        points,
        "smoke_passed" if args.smoke else "completed",
        {
            "solver": "native PhysX",
            "driver": "PhysX tensor kinematic targets",
            "collision_rule": "Exact axis-aligned closed boxes use native box primitives with identical bounds and volume; other shapes retain mesh/native hull conversion",
            "wall_seconds": time.monotonic() - started,
        },
    )
except Exception as ex:
    traceback.print_exc()
    save_result(
        "isaac",
        args.case,
        points,
        "adapter_error",
        {"error": str(ex), "traceback": traceback.format_exc()},
    )
finally:
    if sim:
        sim.reset_simulation()
    app.close()
