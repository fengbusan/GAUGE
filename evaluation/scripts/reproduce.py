"""Paper-version reproduction: native SDK runs, source audit and literal scoring."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gauge_eval.paths import ROOT, SOURCE, verify_dataset_paths
from gauge_eval.scoring import matches_printed, real_reference
from gauge_eval.metrics import rmse, dtw
from gauge_eval.observables import observe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selections(ids):
    cases = json.loads((SOURCE / "recipes/manifest.json").read_text())["cases"]
    selected = [c for c in cases if not ids or c["case_id"] in ids]
    if ids and set(ids) != {c["case_id"] for c in selected}:
        raise ValueError("Unknown case ID")
    return selected


def load_case(case):
    folder = ROOT / "configs/native_sdk_v1" / case["case_id"]
    e = json.loads((folder / "evaluation.json").read_text())
    if sha(folder / "scene.json") != e["scene_sha256"]:
        raise ValueError("Prepared scene hash mismatch: " + case["case_id"])
    return folder, e


def portable_fingerprint(scene, evaluation, workspace):
    """Compare physical inputs independently of their storage directory."""
    root = str(Path(workspace).resolve())

    def canonical(value):
        if isinstance(value, dict):
            return {
                canonical(k): canonical(v)
                for k, v in value.items()
                if k != "scene_sha256"
            }
        if isinstance(value, list):
            return [canonical(v) for v in value]
        if isinstance(value, str):
            if value.startswith("workspace-relative:"):
                return "${WORKSPACE}" + value[len("workspace-relative:") :]
            # Original receipts may reference prepared inputs in a sibling
            # workspace. Preserve profile/case/filename and normalize only
            # the storage prefix, including provenance dictionary keys.
            if value.startswith("/"):
                for marker in ["/configs/", "/data/GAUGE-Dataset/"]:
                    if marker in value:
                        return "${WORKSPACE}" + value[value.index(marker) :]
            if value.startswith(root + "/"):
                return "${WORKSPACE}" + value[len(root) :]
        return value

    payload = json.dumps(
        canonical({"scene": scene, "evaluation": evaluation}),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def asset_fingerprint(path):
    """Keep mesh bytes exact; normalize only URDF display-file locations."""
    path = Path(path)
    if path.suffix != ".urdf":
        return sha(path)
    import xml.etree.ElementTree as ET

    tree = ET.fromstring(path.read_text())
    for element in tree.iter("mesh"):
        filename = element.get("filename", "")
        marker = "/data/GAUGE-Dataset/"
        if filename.startswith("/") and marker in filename:
            element.set("filename", "${WORKSPACE}" + filename[filename.index(marker) :])
    return hashlib.sha256(ET.tostring(tree)).hexdigest()


def import_results(archive, cases):
    """Import generated recordings; reject changed physical inputs and unsafe paths."""
    import zipfile

    selected = {c["case_id"]: c for c in cases}
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("archive_manifest.json"))
        for name, expected in manifest["files"].items():
            parts = Path(name).parts
            if (
                len(parts) != 3
                or parts[0] not in ["isaac", "newton"]
                or parts[1] not in selected
            ):
                if len(parts) == 3 and parts[1] not in selected:
                    continue
                raise ValueError("Invalid archive member: " + name)
            if parts[2] not in [
                "status.json",
                "marker_positions_m.npy",
                "marker_mapping.json",
                "run.log",
            ]:
                raise ValueError("Unexpected archive filename: " + name)
            payload = z.read(name)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Archive member checksum mismatch: " + name)
            out = ROOT / "results/paper_versions" / name
            if out.exists() and out.read_bytes() != payload:
                raise ValueError(
                    "Refusing to replace a different recording: " + str(out)
                )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(payload)
    lock = json.loads((SOURCE / "reproduction/runtime_lock.json").read_text())
    for case in cases:
        _, e = load_case(case)
        for engine in ["isaac", "newton"]:
            folder = ROOT / "results/paper_versions" / engine / case["case_id"]
            validate_recording(folder, e, lock["engines"][engine]["version"])
    print(
        "IMPORTED generated recordings for",
        len(cases),
        "cases; no real-data files included",
    )


def audit(cases):
    """Compare real repeats with their inclusive mean using paper observables."""
    published = json.loads((SOURCE / "recipes/published_scores.json").read_text())
    output = ROOT / "results/paper_references"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        path, e = load_case(case)
        if e["metric"] in ["cradle", "pendulum"]:
            continue
        reference, baseline = real_reference(e, "paper_formula")
        # This separate profile is retained as a real-data diagnostic only.
        diagnostic, diagnostic_baseline = real_reference(e, "published_table")
        np.savez_compressed(
            output / (case["case_id"] + ".npz"),
            literal=reference,
            published_table_diagnostic=diagnostic,
        )
        comparisons = {}
        for metric, values in baseline.items():
            mean, std = (
                published[case["case_id"]][metric]["baseline_printed"]
                .replace(")", "")
                .split(" (")
            )
            comparisons[metric] = {
                "published_mean": mean,
                "published_std": std,
                "computed_mean": values["mean"],
                "computed_std_ddof1": values["std"],
                "mean_matches": matches_printed(values["mean"], mean),
                "std_matches": matches_printed(values["std"], std),
            }
        rows.append(
            {
                "case_id": case["case_id"],
                "paper_trial_paths": e["trial_paths"],
                "released_trials": len(e["trial_paths"]),
                "paper_formula_baseline": baseline,
                "published_table_diagnostic_baseline": diagnostic_baseline,
                "audit": comparisons,
                "input_evaluation_sha256": sha(path / "evaluation.json"),
            }
        )
        print("REFERENCE", case["case_id"], flush=True)
    report = {
        "equations": "Eq2,13,14: XYZ, per-marker angle deficits, marker-triangle areas",
        "real_trial_selection": "All released trials; author exact inclusion list unconfirmed",
        "std_convention": "ddof=1; denominator not explicitly specified by paper",
        "dtw_convention": "Minimum Euclidean-sum fixed-endpoint path, divide by path length; diagonal-first ties",
        "matched_mean_and_std": sum(
            a["mean_matches"] and a["std_matches"]
            for r in rows
            for a in r["audit"].values()
        ),
        "generic_metric_rows": sum(len(r["audit"]) for r in rows),
        "rows": rows,
        "official_equivalence_verified": False,
    }
    (ROOT / "results/paper_reference_audit.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        "BASELINES", report["matched_mean_and_std"], "/", report["generic_metric_rows"]
    )
    return report


def validate_recording(folder, e, required_version):
    receipt = json.loads((folder / "status.json").read_text())
    if receipt["version"] != required_version:
        raise ValueError("Recorded runtime version differs from lock")
    source = ROOT / "configs/native_sdk_v1" / e["case_id"]
    if "portable_input_fingerprint" in receipt:
        scene = json.loads((source / "scene.json").read_text())
        if (
            portable_fingerprint(scene, e, ROOT)
            != receipt["portable_input_fingerprint"]
        ):
            raise ValueError(
                "Imported recording has different physical scene/driver/evaluation inputs"
            )
        for body in scene["bodies"]:
            asset = body["source"].get("path")
            if (
                asset
                and asset_fingerprint(asset)
                != receipt["prepared_asset_fingerprints"][body["name"]]
            ):
                raise ValueError(
                    "Generated asset differs from recorded batch: " + body["name"]
                )
    elif receipt["source_scene_sha256"] != e["scene_sha256"] or receipt[
        "source_evaluation_sha256"
    ] != sha(source / "evaluation.json"):
        raise ValueError(
            "Trajectory belongs to different scene/driver/evaluation inputs"
        )
    recording = folder / "marker_positions_m.npy"
    if sha(recording) != receipt["trajectory_sha256"]:
        raise ValueError("Trajectory checksum differs from receipt")
    q = np.load(recording, allow_pickle=False)
    if len(q) != receipt["frames_completed"]:
        raise ValueError("Recorded frame count differs from receipt")
    if receipt.get("sample_hz") != 30 or receipt.get("physics_hz") != (
        900 if e["regime"] == "deformable" else 180
    ):
        raise ValueError("Recorded timing differs from protocol")
    if receipt["status"] == "completed" and (
        len(q) != e["rollout_frames"] or not np.isfinite(q).all()
    ):
        raise ValueError("Completed receipt has incomplete/nonfinite full trajectory")
    mapping = folder / "marker_mapping.json"
    if e["regime"] != "rigid":
        m = json.loads(mapping.read_text())
        if (
            not m["passed"]
            or m["max_error_m"] >= 0.01
            or len(set(m["native_ids"])) != len(m["native_ids"])
        ):
            raise ValueError("Native marker assignment is not unique or exceeds 1 cm")
    return receipt, q


def score(cases):
    audit_path = ROOT / "results/paper_reference_audit.json"
    reference_rows = {
        r["case_id"]: r for r in json.loads(audit_path.read_text())["rows"]
    }
    public = json.loads((SOURCE / "recipes/published_scores.json").read_text())
    lock = json.loads((SOURCE / "reproduction/runtime_lock.json").read_text())
    rows = []
    for case in cases:
        path, e = load_case(case)
        row = {
            **case,
            "engines": {},
            "baseline_audit": reference_rows.get(case["case_id"], {}),
        }
        for engine in ["isaac", "newton"]:
            folder = ROOT / "results/paper_versions" / engine / case["case_id"]
            result = {"status": "not_run", "scores": {}}
            if (folder / "status.json").exists():
                result = {
                    **json.loads((folder / "status.json").read_text()),
                    "scores": {},
                }
                # No failed or partial full-process attempt receives a score.
                if result["status"] == "completed":
                    try:
                        receipt, q = validate_recording(
                            folder, e, lock["engines"][engine]["version"]
                        )
                        if e["metric"] in ["cradle", "pendulum"]:
                            result["score_notice"] = (
                                "Special event selection unconfirmed; scores withheld"
                            )
                        else:
                            r = reference_rows[case["case_id"]]
                            if r["input_evaluation_sha256"] != sha(
                                path / "evaluation.json"
                            ):
                                raise ValueError(
                                    "Real reference audit belongs to different inputs"
                                )
                            with np.load(
                                ROOT
                                / "results/paper_references"
                                / (case["case_id"] + ".npz")
                            ) as references:
                                obs = observe(
                                    q[: e["frames"]] * 1000, e, "paper_formula"
                                )
                                for metric, fn in [("RMSE", rmse), ("DTW", dtw)]:
                                    raw = fn(obs, references["literal"])
                                    base = r["paper_formula_baseline"][metric]["mean"]
                                    pub = float(r["audit"][metric]["published_mean"])
                                    result["scores"][metric] = {
                                        "paper_formula_raw": raw,
                                        "paper_formula_normalized": raw / base
                                        if base > 0
                                        else None,
                                        "normalized_using_published_baseline": raw / pub
                                        if pub > 0
                                        else None,
                                        "published_engine_value": public[
                                            case["case_id"]
                                        ][metric]["engines"].get(
                                            {"isaac": "IsaacSim", "newton": "Newton"}[
                                                engine
                                            ]
                                        ),
                                        "real_baseline_mean_and_std_match": r["audit"][
                                            metric
                                        ]["mean_matches"]
                                        and r["audit"][metric]["std_matches"],
                                    }
                    except (ValueError, FileNotFoundError, KeyError) as ex:
                        result.update(
                            status="invalid_recording", score_error=str(ex), scores={}
                        )
            row["engines"][engine] = result
        row["engines"]["genesis"] = {
            "status": "await_author_version_or_commit",
            "required_version": "1.12.0",
        }
        rows.append(row)
        print("SCORED", case["case_id"], flush=True)
    report = {
        "scoring": "literal_paper_equations",
        "official_equivalence_verified": False,
        "comparison_notice": "Published-baseline scaling is diagnostic; no aggregate or certified rankings",
        "rows": rows,
    }
    (ROOT / "results/paper_version_score_report.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


def run(args, cases):
    output = "results/paper_smoke" if args.smoke else "results/paper_versions"
    env = {
        **os.environ,
        "GAUGE_WORKSPACE": str(ROOT),
        "GAUGE_NATIVE_PROFILE": "native_sdk_v1",
        "THIRDPARTY_OUTPUT_ROOT": output,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if args.engine == "isaac":
        if not args.isaac_root:
            raise ValueError("--isaac-root or GAUGE_ISAAC_RUNTIME is required")
        env["GAUGE_ISAAC_RUNTIME"] = str(Path(args.isaac_root).expanduser().resolve())
    unsuccessful = 0
    for case in cases:
        folder, e = load_case(case)
        out = ROOT / output / args.engine / case["case_id"]
        out.mkdir(parents=True, exist_ok=True)
        status = out / "status.json"
        if status.exists() and not args.retry:
            print("SKIP terminal attempt", args.engine, case["case_id"], flush=True)
            continue
        if args.engine == "genesis":
            status.write_text(
                json.dumps(
                    {
                        "engine": "genesis",
                        "case_id": case["case_id"],
                        "status": "await_author_version_or_commit",
                        "required_version": "1.12.0",
                        "error": "No substitute runtime or unverified adapter is used",
                    },
                    indent=2,
                )
            )
            unsuccessful += 1
            continue
        adapter = (
            "isaac_soft"
            if args.engine == "isaac" and case["regime"] != "rigid"
            else args.engine
            + "_"
            + {"rigid": "rigid", "textile": "cloth", "deformable": "foam"}[
                case["regime"]
            ]
        )
        script = SOURCE / "scripts/thirdparty" / (adapter + ".py")
        snapshot = {f.name: sha(f) for f in script.parent.glob("*.py")}
        # Preserve a previous failed attempt when --retry is explicit.
        if status.exists():
            history = out / ("previous_attempt_" + str(time.time_ns()))
            history.mkdir()
            for f in list(out.iterdir()):
                if f.is_file():
                    f.rename(history / f.name)
        if not args.no_gpu_wait:
            while True:
                active = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    text=True,
                ).strip()
                if not active:
                    break
                print("WAIT GPU occupied", flush=True)
                time.sleep(10)
        print("START", args.engine, case["case_id"], flush=True)
        timed_out = False
        with (out / "run.log").open("w") as log:
            child = subprocess.Popen(
                [args.python, str(script), case["case_id"]]
                + (["--smoke"] if args.smoke else []),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        result = (
            json.loads(status.read_text())
            if status.exists()
            else {
                "status": "process_error",
                "error": "No native terminal receipt; see run.log",
            }
        )
        if timed_out:
            result.update(status="timeout", limit_seconds=args.timeout)
        elif child.returncode and result["status"] in ["completed", "smoke_passed"]:
            result.update(
                status="process_error",
                error="Native process exited unsuccessfully after receipt",
            )
        result.update(
            engine=args.engine,
            case_id=case["case_id"],
            exit_code=child.returncode,
            adapter_snapshot_sha256=snapshot,
        )
        status.write_text(json.dumps(result, indent=2))
        print("END", args.engine, case["case_id"], result["status"], flush=True)
        unsuccessful += result["status"] not in ["completed", "smoke_passed"]
    return int(unsuccessful > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    q = commands.add_parser("import-results")
    q.add_argument("--archive", type=Path, required=True)
    q.add_argument("--case", action="append")
    for name in ["audit", "score", "verify"]:
        q = commands.add_parser(name)
        q.add_argument("--case", action="append")
    q = commands.add_parser("run")
    q.add_argument("--engine", choices=["isaac", "newton", "genesis"], required=True)
    q.add_argument("--python", default=sys.executable)
    q.add_argument("--isaac-root", default=os.environ.get("GAUGE_ISAAC_RUNTIME"))
    q.add_argument("--case", action="append")
    q.add_argument("--smoke", action="store_true")
    q.add_argument("--retry", action="store_true")
    q.add_argument("--timeout", type=int, default=7200)
    q.add_argument(
        "--no-gpu-wait",
        action="store_true",
        help="Dedicated machines without nvidia-smi; processes still run serially",
    )
    args = p.parse_args()
    cases = selections(args.case)
    if args.command == "import-results":
        import_results(args.archive, cases)
        return 0
    if args.command == "run":
        return run(args, cases)
    if args.command == "audit":
        audit(cases)
    elif args.command == "score":
        score(cases)
    else:
        inventory = json.loads((SOURCE / "recipes/dataset_files.json").read_text())
        verify_dataset_paths([f["path"] for f in inventory["files"]])
        lock = json.loads((SOURCE / "reproduction/runtime_lock.json").read_text())
        checked = 0
        for case in cases:
            _, e = load_case(case)
            for engine in ["isaac", "newton"]:
                folder = ROOT / "results/paper_versions" / engine / case["case_id"]
                if (folder / "status.json").exists():
                    validate_recording(folder, e, lock["engines"][engine]["version"])
                    checked += 1
        print(
            "VERIFIED",
            len(inventory["files"]),
            "pinned source files;",
            checked,
            "native recordings",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
