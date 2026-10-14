"""Reproduction recordings must remain bound to physical inputs and runtime."""

import json

import numpy as np
import pytest

import reproduce


def test_portable_fingerprint_accepts_only_storage_directory_changes():
    scene = {"path": "/old/configs/native_sdk_v1/a/mesh.msh", "dt": 1 / 180}
    e = {
        "scene_sha256": "old-path-sensitive-hash",
        "frames": 3,
        "drivers": [{"positions": [[0, 0, 0], [1, 0, 0]]}],
    }
    moved = {"path": "/new/configs/native_sdk_v1/a/mesh.msh", "dt": 1 / 180}
    key = reproduce.portable_fingerprint(scene, e, "/old")
    assert key == reproduce.portable_fingerprint(
        moved, {**e, "scene_sha256": "new"}, "/new"
    )
    for other in [{"frames": 2}, {"drivers": [{"positions": [[0, 0, 0], [2, 0, 0]]}]}]:
        assert key != reproduce.portable_fingerprint(moved, {**e, **other}, "/new")
    assert key != reproduce.portable_fingerprint({**moved, "dt": 1 / 900}, e, "/new")


def test_recording_rejects_wrong_runtime_and_corrupted_trajectory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    inputs = tmp_path / "configs/native_sdk_v1/case"
    inputs.mkdir(parents=True)
    e = {
        "case_id": "case",
        "regime": "rigid",
        "scene_sha256": "scene",
        "rollout_frames": 3,
    }
    (inputs / "evaluation.json").write_text(json.dumps(e))
    out = tmp_path / "results/paper_versions/newton/case"
    out.mkdir(parents=True)
    q = out / "marker_positions_m.npy"
    np.save(q, np.zeros((3, 1, 3)))
    receipt = {
        "version": "1.3.0",
        "status": "completed",
        "frames_completed": 3,
        "source_scene_sha256": "scene",
        "source_evaluation_sha256": reproduce.sha(inputs / "evaluation.json"),
        "trajectory_sha256": reproduce.sha(q),
        "sample_hz": 30,
        "physics_hz": 180,
    }
    (out / "status.json").write_text(json.dumps(receipt))
    reproduce.validate_recording(out, e, "1.3.0")
    with pytest.raises(ValueError, match="runtime version"):
        reproduce.validate_recording(out, e, "1.6.1")
    np.save(q, np.ones((3, 1, 3)))
    with pytest.raises(ValueError, match="checksum"):
        reproduce.validate_recording(out, e, "1.3.0")


def test_urdf_fingerprint_normalizes_only_display_location(tmp_path):
    a = tmp_path / "a.urdf"
    b = tmp_path / "b.urdf"
    xml = '<robot><link><inertial><mass value="0.033"/></inertial><collision><sphere radius="0.01"/></collision><visual><mesh filename="{root}/data/GAUGE-Dataset/assets/obj/ball.obj"/></visual></link></robot>'
    a.write_text(xml.format(root="/first"))
    b.write_text(xml.format(root="/second"))
    assert reproduce.asset_fingerprint(a) == reproduce.asset_fingerprint(b)
    b.write_text(xml.format(root="/second").replace("0.033", "0.034"))
    assert reproduce.asset_fingerprint(a) != reproduce.asset_fingerprint(b)
