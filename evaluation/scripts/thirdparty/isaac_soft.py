"""Native PhysX FEM, native automatic attachments, and CPU kinematic targets.

FEM still runs on GPU. CPU readback is the supported path for native rigid
kinematic targets in this installed version; no PD and no synthetic clamp body.
"""

import os, argparse, json, time, traceback, sys
from pathlib import Path
import numpy as np
from isaacsim import SimulationApp

p = argparse.ArgumentParser()
p.add_argument("case")
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
app = SimulationApp({"headless": True, "disable_viewport_updates": True})
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf, UsdShade
import omni.usd, carb, omni.physics.tensors
from omni.physx import get_physx_interface
from omni.physx.scripts import deformableUtils
from common import (
    ROOT,
    load,
    sample_driver,
    native_marker_mapping,
    save_result,
    runtime_identity,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gauge_eval.scenes import pose

s, e = load(a.case)
points = []
driver_checks = []
started = time.monotonic()
sim = None
out = ROOT / os.environ.get("THIRDPARTY_OUTPUT_ROOT", "results") / "isaac" / a.case
out.mkdir(parents=True, exist_ok=True)


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


try:
    runtime_identity("isaac")
    source = ROOT / "data/GAUGE-Dataset/assets/usd" / (e["task"] + ".usd")
    omni.usd.get_context().open_stage(str(source))
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    body = next(b for b in s["bodies"] if b["kind"] == "deformable")
    meta = json.load(
        open(ROOT / "data/GAUGE-Dataset/metadata" / e["regime"] / (e["task"] + ".json"))
    )
    owner = stage.GetPrimAtPath("/World/assets/" + e["body"])
    transform(owner, pose(meta["tasks"]["default"], e["body"], e["material"]))
    # The published auto-attachment settings are the native recipe. Generated vertex
    # IDs can refer to a cooked mesh, so never treat them as raw points indices.
    recipes = []
    old_recipe_paths = []
    for pr in stage.Traverse():
        if pr.HasAPI("PhysxAutoDeformableAttachmentAPI"):
            actor = str(
                pr.GetRelationship(
                    "physxAutoDeformableAttachment:attachable1"
                ).GetTargets()[0]
            )
            attributes = [
                (attr.GetName(), attr.GetTypeName(), attr.Get())
                for attr in pr.GetAttributes()
                if attr.GetName().startswith("physxAutoDeformableAttachment:")
                and "Crc" not in attr.GetName()
                and attr.Get() is not None
            ]
            recipes.append((actor, attributes))
            old_recipe_paths.append(str(pr.GetPath()))
    # Removing a prim spec in the session layer leaves the weaker asset prim
    # active. Deactivate old cooked attachments explicitly before native recooking.
    for path in old_recipe_paths:
        stage.GetPrimAtPath(path).SetActive(False)
    old_attachments = stage.GetPrimAtPath("/World/attachments")
    if old_attachments:
        old_attachments.SetActive(False)
    for rb in s["bodies"]:
        if rb["kind"] != "rigid":
            continue
        prim = stage.GetPrimAtPath(e["usd_asset_paths"][rb["name"]])
        assert prim
        transform(prim, rb["initial"]["q"])
        api = UsdPhysics.RigidBodyAPI.Apply(prim)
        api.CreateRigidBodyEnabledAttr(True)
        api.CreateKinematicEnabledAttr(True)
        ma = UsdPhysics.MassAPI.Apply(prim)
        ma.CreateMassAttr(float(rb["source"]["mass"]))
        rmat = UsdShade.Material.Define(
            stage, "/World/calibrated_" + rb["name"].replace("-", "_")
        )
        rma = UsdPhysics.MaterialAPI.Apply(rmat.GetPrim())
        rma.CreateStaticFrictionAttr(float(rb["contact_material"]["friction"]))
        rma.CreateDynamicFrictionAttr(float(rb["contact_material"]["friction"]))
        rma.CreateRestitutionAttr(float(rb["contact_material"]["restitution"]))
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(child).Bind(
                    rmat, materialPurpose="physics"
                )
    scenes = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    assert len(scenes) == 1
    physics = UsdPhysics.Scene(scenes[0])
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    pa = PhysxSchema.PhysxSceneAPI.Apply(scenes[0])
    pa.CreateEnableGPUDynamicsAttr(True)
    pa.CreateBroadphaseTypeAttr("GPU")
    pa.CreateTimeStepsPerSecondAttr(round(1 / s["dt"]))
    mk = e["material_parameter_key"]
    m = meta["assets"][e["body"]]["material"][mk]
    mat = UsdShade.Material.Define(stage, "/World/measuredMaterial")
    mat.GetPrim().SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["OmniPhysicsDeformableMaterialAPI"]),
    )
    values = {
        "density": body["source"]["density"],
        "youngsModulus": body["source"]["material"]["young_modulus"],
        "poissonsRatio": body["source"]["material"]["poisson_ratio"],
        "dynamicFriction": m.get("friction", 0),
        "staticFriction": m.get("friction", 0),
    }
    if e["regime"] == "textile":
        values.update(
            surfaceThickness=m["thickness"],
            surfaceBendStiffness=m["bend stiffness"] / (4 * m["thickness"] ** 3),
        )
        assert deformableUtils.add_surface_deformable_material(
            stage,
            mat.GetPath(),
            density=values["density"],
            youngs_modulus=values["youngsModulus"],
            poissons_ratio=values["poissonsRatio"],
            static_friction=values["staticFriction"],
            dynamic_friction=values["dynamicFriction"],
            surface_thickness=values["surfaceThickness"],
            surface_bend_stiffness=values["surfaceBendStiffness"],
        )
        assert mat.GetPrim().HasAPI("OmniPhysicsSurfaceDeformableMaterialAPI")
        geom = UsdGeom.PointBased(
            stage.GetPrimAtPath(str(owner.GetPath()) + "/simulation_mesh")
        )
    else:
        geom = UsdGeom.PointBased(
            stage.GetPrimAtPath(str(owner.GetPath()) + "/simulation_mesh")
        )
    for name, val in values.items():
        mat.GetPrim().CreateAttribute(
            "omniphysics:" + name, Sdf.ValueTypeNames.Float
        ).Set(float(val))
    for pr in Usd.PrimRange(owner):
        UsdShade.MaterialBindingAPI.Apply(pr).Bind(mat, materialPurpose="physics")
    for i, (actor, attributes) in enumerate(recipes):
        path = Sdf.Path("/World/native_attachments/native_" + str(i))
        assert deformableUtils.create_auto_deformable_attachment(
            stage, path, owner.GetPath(), Sdf.Path(actor)
        )
        for name, kind, value in attributes:
            stage.GetPrimAtPath(path).CreateAttribute(name, kind).Set(value)
    # Supported CPU rigid-control API with GPU FEM and native mesh readback.
    carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    app.update()
    sim = get_physx_interface()
    sim.start_simulation()
    sim.update_transformations(True, True, True, False)
    tv = omni.physics.tensors.create_simulation_view(
        "numpy", stage_id=omni.usd.get_context().get_stage_id()
    )
    views = {
        e["usd_asset_paths"][rb["name"]]: tv.create_rigid_body_view(
            e["usd_asset_paths"][rb["name"]]
        )
        for rb in s["bodies"]
        if rb["kind"] == "rigid"
    }

    def native_world_points():
        p = np.asarray(geom.GetPointsAttr().Get(), dtype=float)
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(geom.GetPrim())
        return np.array([matrix.Transform(Gf.Vec3d(*map(float, x))) for x in p])

    initial = native_world_points()
    ids, error = native_marker_mapping(e, initial, "isaac", a.case)
    points.append(initial[ids])
    n = 3 if a.smoke else e["rollout_frames"]
    steps = round(1 / 30 / s["dt"])
    stage.Flatten().Export(str(out / "input.usda"))
    for frame in range(1, n):
        for sub in range(steps):
            t = ((frame - 1) * steps + sub + 1) * s["dt"]
            for d in e["drivers"]:
                q = sample_driver(d, t)
                views[e["usd_asset_paths"][d["name"]]].set_kinematic_targets(
                    np.array([np.r_[q[:3], q[4:], q[3]]], np.float32),
                    np.array([0], np.int32),
                )
            sim.update_simulation(s["dt"], t - s["dt"])
            sim.update_transformations(True, True, True, False)
        current = native_world_points()
        assert current.shape == initial.shape, (
            "Native mesh topology changed during fixed-vertex tracking"
        )
        value = current[ids]
        assert np.isfinite(value).all()
        points.append(value)
        for d in e["drivers"]:
            view = views[e["usd_asset_paths"][d["name"]]]
            actual = view.get_transforms()[0]
            velocity = view.get_velocities()[0]
            target = sample_driver(d, frame / 30)
            previous = sample_driver(d, frame / 30 - s["dt"])
            error_position = float(np.linalg.norm(actual[:3] - target[:3]))
            expected = (target[:3] - previous[:3]) / s["dt"]
            driver_checks.append(
                {
                    "frame": frame,
                    "name": d["name"],
                    "physics_pose_xyzw": actual.tolist(),
                    "physics_velocity": velocity.tolist(),
                    "target": target.tolist(),
                    "position_error_m": error_position,
                    "velocity_error_m_s": float(
                        np.linalg.norm(velocity[:3] - expected)
                    ),
                }
            )
            assert error_position < 1e-5, (
                "Native kinematic pose mismatch",
                d["name"],
                error_position,
            )
        if frame % 30 == 0:
            print("FRAME", frame, "/", n, flush=True)
    assert np.max(np.abs(np.asarray(points)[-1] - points[0])) > 1e-8, (
        "Native soft body never moved"
    )
    save_result(
        "isaac",
        a.case,
        points,
        "smoke_passed" if a.smoke else "completed",
        {
            "solver": "PhysX SurfaceFEM" if e["regime"] == "textile" else "PhysX FEM",
            "native_material": values,
            "material_mapping_version": "surface_api_fixed_v1",
            "marker_mapping_max_m": error,
            "native_vertex_source": "Canonical PhysX simulation-mesh vertices after native cooking; fixed IDs, no intermediate proxy",
            "vertices": len(initial),
            "native_auto_attachment_groups": len(recipes),
            "driver": "Native PhysX CPU kinematic targets with GPU FEM; no PD",
            "driver_max_position_error_m": max(
                (c["position_error_m"] for c in driver_checks), default=0.0
            ),
            "driver_max_velocity_error_m_s": max(
                (c["velocity_error_m_s"] for c in driver_checks), default=0.0
            ),
            "wall_seconds": time.monotonic() - started,
        },
    )
except Exception as ex:
    traceback.print_exc()
    save_result(
        "isaac",
        a.case,
        points,
        "adapter_error",
        {"error": str(ex), "traceback": traceback.format_exc()},
    )
finally:
    (out / "driver_checks.json").write_text(json.dumps(driver_checks))
    if sim:
        sim.reset_simulation()
    app.close()
