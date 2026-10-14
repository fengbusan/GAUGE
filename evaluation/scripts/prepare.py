"""Materialize all frozen recipes using an external, hash-checked official dataset."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from gauge_eval.paths import ROOT, DATA, SOURCE, verify_dataset_paths
from gauge_eval.scenes import obj, pose, write_msh, driver


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand(value, folder):
    if isinstance(value, dict):
        return {k: expand(v, folder) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, folder) for v in value]
    if isinstance(value, str):
        return value.replace("${DATASET}", str(DATA)).replace("${PROFILE}", str(folder))
    return value


def official_soft_mesh(spec, target):
    source = Path(spec["source"])
    metadata = json.loads(
        (DATA / "metadata" / spec["regime"] / (spec["task"] + ".json")).read_text()
    )
    q = np.asarray(pose(metadata["tasks"]["default"], spec["body"], spec["material"]))
    if spec["regime"] == "textile":
        vertices, cells = obj(source)
        kind = 2
    else:
        from pxr import Usd

        stage = Usd.Stage.Open(str(source))
        prim = stage.GetPrimAtPath("/World/assets/foam/simulation_mesh")
        points = np.asarray(prim.GetAttribute("points").Get(), dtype=float)
        rest = np.asarray(
            prim.GetAttribute("omniphysics:restShapePoints").Get(), dtype=float
        )
        if not np.array_equal(points, rest):
            raise ValueError("Authored foam rest shape differs from points")
        vertices, inverse = np.unique(points, axis=0, return_inverse=True)
        cells = np.unique(
            np.sort(
                inverse[
                    np.asarray(prim.GetAttribute("tetVertexIndices").Get(), dtype=int)
                ],
                axis=1,
            ),
            axis=0,
        )
        kind = 4
    world = Rotation.from_quat(q[[4, 5, 6, 3]]).apply(vertices) + q[:3]
    if kind == 4:
        det = np.linalg.det(world[cells[:, 1:]] - world[cells[:, 0, None]])
        if np.any(abs(det) <= 1e-18):
            raise ValueError("Degenerate official tetrahedron")
        neg = det < 0
        cells[neg, 1], cells[neg, 2] = cells[neg, 2].copy(), cells[neg, 1].copy()
    write_msh(target, world, cells, kind)
    return world, cells


def bending_edges(vertices, faces, modulus, thickness):
    areas = (
        np.linalg.norm(
            np.cross(
                vertices[faces[:, 1]] - vertices[faces[:, 0]],
                vertices[faces[:, 2]] - vertices[faces[:, 0]],
            ),
            axis=1,
        )
        / 2
    )
    edges = {}
    for j, tri in enumerate(faces):
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(j)
    if any(len(fs) > 2 for fs in edges.values()):
        raise ValueError("Non-manifold textile edge")
    return [
        {
            "vertices": list(edge),
            "stiffness": float(
                12
                * areas[fs].sum()
                * modulus
                / (thickness**3 * np.sum((vertices[edge[0]] - vertices[edge[1]]) ** 2))
            ),
        }
        for edge, fs in edges.items()
        if len(fs) == 2
    ]


def prepare(case_ids=None, mode="both"):
    manifest = json.loads((SOURCE / "recipes/manifest.json").read_text())
    selected = [
        c for c in manifest["cases"] if case_ids is None or c["case_id"] in case_ids
    ]
    if case_ids and len(selected) != len(set(case_ids)):
        raise ValueError("Unknown case ID")
    receipts = []
    for case in selected:
        cid = case["case_id"]
        bundle = json.loads((SOURCE / "recipes" / f"{cid}.json").read_text())
        for variant, profile in [("xie", "xie_sdk_v1"), ("native", "native_sdk_v1")]:
            if mode != "both" and variant != mode:
                continue
            folder = ROOT / "configs" / profile
            out = folder / cid
            out.mkdir(parents=True, exist_ok=True)
            data = expand(copy.deepcopy(bundle[variant]), folder)
            scene, e = data["scene"], data["evaluation"]
            verify_dataset_paths(
                e.get("trial_paths", []) + [f"metadata/{e['regime']}/{e['task']}.json"]
            )
            meshes = {}
            for key, raw_spec in bundle["assets"].items():
                if not key.startswith(variant + "-"):
                    continue
                spec = expand(raw_spec, folder)
                target = out / "assets" / key
                target.parent.mkdir(exist_ok=True)
                if (
                    "source_sha256" in spec
                    and sha(spec["source"]) != spec["source_sha256"]
                ):
                    raise ValueError(f"Dataset hash mismatch: {spec['source']}")
                if spec["kind"] == "official_soft_mesh":
                    meshes[str(target)] = official_soft_mesh(spec, target)
                elif spec["kind"] == "copy_dataset":
                    target.write_bytes(Path(spec["source"]).read_bytes())
                else:
                    target.write_text(spec["text"])
                if (
                    spec.get("expected_sha256")
                    and sha(target) != spec["expected_sha256"]
                ):
                    raise ValueError(
                        f"Regenerated mesh differs from frozen input: {cid}/{key}"
                    )
            raws = [
                json.loads((DATA / name).read_text())
                for name in e.get("trial_paths", [])
            ]
            for d in e["drivers"]:
                raw_name = d.pop("measurement_name", d["name"])
                generated = driver(raws, raw_name, d.pop("measurement_anchor"))
                generated["name"] = d["name"]
                d.update(generated)
            for c in e.get("controls", []):
                c["driver"] = copy.deepcopy(
                    next(d for d in e["drivers"] if d["name"] == c["driver"]["name"])
                )
            for b in scene["bodies"]:
                for constitution in b["source"].get("extra_constitutions", []):
                    if constitution.pop("regenerate_official_edge_stiffnesses", False):
                        v, f = meshes[b["source"]["path"]]
                        meta = json.loads(
                            (
                                DATA / "metadata" / e["regime"] / (e["task"] + ".json")
                            ).read_text()
                        )
                        material = meta["assets"][e["body"]]["material"][
                            e["material_parameter_key"]
                        ]
                        constitution["edge_stiffnesses"] = bending_edges(
                            v,
                            f,
                            material["bend stiffness"],
                            b["source"]["shell_thickness"],
                        )
            (out / "scene.json").write_text(json.dumps(scene, indent=2))
            e.update(
                profile=profile,
                scene_sha256=sha(out / "scene.json"),
                official_equivalence_verified=False,
            )
            (out / "evaluation.json").write_text(json.dumps(e, indent=2))
            receipts.append(
                dict(
                    case_id=cid,
                    profile=profile,
                    scene_sha256=e["scene_sha256"],
                    evaluation_sha256=sha(out / "evaluation.json"),
                )
            )
        print("PREPARED", cid, flush=True)
    (ROOT / "configs/import_receipt.json").write_text(json.dumps(receipts, indent=2))
    return receipts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case", action="append", help="Repeat to prepare a subset; default all 55"
    )
    p.add_argument("--mode", choices=["native", "xie", "both"], default="both")
    args = p.parse_args()
    prepare(args.case, args.mode)
