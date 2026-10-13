# SDK-only GAUGE evaluation harness

This is a community evaluation harness in a personal fork of [InternRobotics/GAUGE](https://github.com/InternRobotics/GAUGE). It is **not the author's released evaluator** and does not certify official-equivalent scores.

The 55 cases expand the paper's 14 mechanisms across materials and subtasks: 29 rigid, 18 textile, and 8 volumetric deformable cases. The original XiePhysic profile uses IPC for all cases, with **RP enabled at native SDK defaults for rigid bodies only**. The optional 0.5.14 profile uses SAP with RP disabled. SoftPrimal is excluded. SDKs, engine source, dataset files, rendered videos, credentials, and workstation deployment scripts are not distributed here.

## Quick start

Use Python 3.11+ for preparation/scoring. Install the Xie GPU SDK independently in a compatible environment; this repository neither builds nor redistributes it.

```bash
cd evaluation
python -m pip install -r requirements.txt
# Optional: write data/configs/results outside the checkout.
export GAUGE_WORKSPACE="$PWD/_work"
python scripts/fetch_dataset.py
python scripts/prepare.py
python -m pytest tests -q

# The chosen Python must already have the appropriate SDK installed.
python scripts/run_suite.py --engine xiephysic --python /path/to/xie-sdk-env/bin/python
python scripts/run_suite.py --engine isaac --python /path/to/isaac-sim/python.sh
python scripts/run_suite.py --engine genesis --python /path/to/genesis-env/bin/python
python scripts/run_suite.py --engine newton --python /path/to/newton-env/bin/python
python scripts/score.py
```

The default workspace is `evaluation/`. For an existing dataset, symlink `$GAUGE_WORKSPACE/data/GAUGE-Dataset` to its root. Preparation needs `pxr` (`usd-core`) to read official tetrahedra. Dataset files are pinned to revision `9e0acb70fecc0d4161660264d9a4b08d8f56d45a`; download checks SHA-256 against the frozen inventory. Raw measured trajectories are reconstructed from that dataset and never embedded in the repository.

`--case CASE_ID` (repeatable) selects a subset for preparation/runs/scoring. `--smoke` runs three frames and writes to `results/smoke`, which the scorer never ranks. `run_suite.py` launches **one GPU process at a time**, defaults to 7200 seconds per case, retains errors and timeouts, and resumes completed receipts. `--retry` explicitly reruns them. For Xie SDK modules with another import name set `GAUGE_XIE_SDK_MODULE`. The default required version is 0.5.13; intentional version changes require `GAUGE_XIE_SDK_VERSION`. To select an unpacked release or override a stale editable installation, set `GAUGE_XIE_SDK_PATH` to the directory containing the SDK package. This selection removes stale SDK import caches inside the isolated process and verifies the actual package location. The receipt records actual version and native binary SHA-256; CUDA libraries remain configured by your SDK installation. `run_xie.py CASE_ID --record-state` additionally records full 30 Hz body state in `state_q.jsonl` for external renderers; it can be large.

Tested SDK installations: XiePhysic 0.5.13, Isaac Sim 6.0.1-rc.7, Genesis 1.4.3, Newton 1.6.1 / Warp 1.18. Use separate native environments (`genesis-world==1.4.3`, `newton==1.6.1` where available; use vendors' installation instructions). The Newton/Genesis adapters also require NumPy, SciPy, and trimesh. Isaac scripts must run under Isaac's `python.sh`, after `SimulationApp` initializes the physics schemas. These are tested local versions, not claims about which exact builds the paper authors used.

## What is frozen

`recipes/manifest.json` lists every case. Case JSON contains preselected material mappings, poses, marker IDs, boundary regions and conversion policies for Xie/native SDK inputs. Motion arrays and references are reconstructed from the external trials. `prepare.py` regenerates official planar textile OBJ meshes and official USD foam tetrahedra; resulting mesh bytes must match the frozen SHA-256.

Foam imports preserve all unique coordinates and cells: exact-coordinate welding and duplicate-cell removal, with positive tet orientation; **no simplification/remeshing**. Textile IPC uses the official planar OBJ with unchanged vertices/faces and mesh-dependent bending conversion. Native Isaac uses its official USD simulation mesh and native automatic attachments. They have different discretizations; the author's actual runtime mesh construction was not released.

Rigid boxes retain their exact surface through 12 triangles. Other rigid mesh assets use triangle contact for IPC; analytical spheres/URDF joints are retained where applicable. The user-approved **2 mm Newton's-cradle gap** is an explicitly declared Xie adaptation, not an author-equivalent initial configuration. Xie moving apparatus uses implicit PD (`kp=100000`, `kd=100`, gravity compensation) and coupled compliant attachments (`100000 N/m`). Native engines never borrow Xie's PD logic. All driver tracking deviations must be reported rather than assumed zero.

Physics runs at 180 Hz for rigid/textile and 900 Hz for volumetric deformables, recording at 30 Hz. Full motion is simulated; only the frozen official-table evaluation window is scored. Materials come from metadata, with explicit adapter conversions/clamps where native constitutive models require them; these conversions are visible in recipes and adapter code, not hidden per-case score tuning. `recipes/rp_defaults.json` records inspected 0.5.13 defaults as data; only `enabled: true` is passed for rigid RP, so later SDK defaults can change behavior and are not equivalent to that pinned build.

## Scoring and reproduction limits

The engine-independent implementation uses the declared `published_table_reproduction_v3` interpretation. It was frozen using real repeats, not simulator scores. Run `python scripts/score.py --references-only` to independently audit it. The complete pinned dataset reproduces **102/106 generic baseline mean and sample-standard-deviation pairs** at printed precision. Four generic rows remain unresolved; the two articulated cases' four special rows (PD/EL and LSD/MTE) are withheld from normalized rankings.

- RMSE is `sqrt(mean_t(sum_features(error**2)))`.
- DTW minimizes the **sum** of Euclidean local costs on an unconstrained, fixed-endpoint path, then divides by path length. Ties prefer diagonal, vertical, horizontal.
- Simulation reference is the mean of all real repeats. Rigid baseline uses the inclusive mean; textile/foam baseline uses leave-one-out means. Standard deviation uses `ddof=1`.
- Rigid observation is XYZ except bouncing-ball Z; textile is angle deficit except bending marker-chain circumcircle curvature (1/m); foam is marker-triangle area (mm²), with the documented second-face index correction.
- Scores are error ratios to a positive real baseline: **lower is better**, not distance to 1. No unsupported aggregate score is produced.

`score_report.json` contains unmodified printed author scores, local scores, per-case baseline checks, failures/blocked cases, and separate reference rankings against published peers and local peers. First/second counts and percentages are provided for all/rigid/textile/deformable, separately for RMSE/DTW. Ranking uses complete, eligible common cases with matching baseline mean/std. Scores are rounded half-up to the least precise published peer token in the row; ties use competition ranks (1,1,3). Coverage is always disclosed. Case flags exclude unresolved material-label/window/marker issues. A ranking against author-published scores is a **reference comparison**, not official certification.

The literal-paper observable option remains in `gauge_eval/observables.py` for diagnostics. It is not identical to the empirically table-reproducing profile. The paper and released data do not resolve all conventions. Matching real baselines does not prove the scene reconstruction, solver configuration, driving semantics or simulated scores reproduce the author runs.

Known native limitations:

- **Isaac textile material fix:** published local runs before this export omitted `OmniPhysicsSurfaceDeformableMaterialAPI` on a replacement material. This harness applies it using the native helper and asserts the API exists. Historical textile scores predate the fix and are not bundled/relabelled as fixed results. Backend effect and fresh complete runs still need validation.
- **Genesis:** moving-contact kinematic semantics were not verified in the tested build. Those cases return `protocol_blocked`, rather than using PD or teleporting a dynamic object to force a score. Volumetric mapping currently uses native MPM; that is not a claim that the paper authors used MPM/UIPC.
- **Newton:** cloth uses native VBD; volumetric soft uses implicit MPM with zero-mass prescribed boundary particles, beginning-of-step velocities/APIC rotation gradients, and explicit end-position boundary projection. This is disclosed native boundary mapping, not proof of the author's identical implementation. Nonfinite outputs are failures, never ranked.

## Layout and license

`evaluation/scripts/` contains preparation, SDK adapters, serial execution and shared scoring. `evaluation/recipes/` contains only portable configuration recipes, small generated apparatus descriptions, asset/trial hashes and printed reference numbers. `evaluation/tests/` checks metric semantics, geometry units, driver endpoints, and ranking ties. All writable caches/results are ignored by Git.

Harness contributions are offered under the repository's MIT license. External engine SDKs and the [GAUGE Dataset](https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset) retain their own licenses/terms; nothing here grants redistribution rights over those external dependencies.

## SAP 0.5.14 comparison

SAP requires the independently installed **0.5.14 prerelease** SDK. Select it explicitly; never relabel 0.5.13 trajectories as SAP. JSON scene defaults apply: `velocity_primal/frozen_sap`, BDF1, RP disabled, 10 inner iterations / 10 line searches, relative tolerance 0.001, velocity tolerance 0.05, swept point candidates enabled, beta 1 and sigma 0.0001. Linear selection is auto with fixed-budget convergence and Block Jacobi. The SDK's USD pure-rigid policy is different (80/30 with discrete candidates); `--sap-policy usd` is provided only for explicit diagnostics.

```bash
export GAUGE_XIE_SDK_PATH=/path/to/installed-0.5.14-sdk
export GAUGE_XIE_SDK_VERSION=0.5.14
# Optionally require the exact chosen release binary hash:
export GAUGE_XIE_SDK_SHA256=YOUR_NATIVE_BINARY_SHA256
python scripts/prepare_sap.py --profile xie_sap_native_0514_v3 --normalize-rigid-topology
python scripts/run_suite.py --engine xiephysic --profile xie_sap_native_0514_v3 \
  --result-key xie_sap_native_v3 --record-state
python scripts/score.py --xie-profile xie_sap_native_0514_v3 \
  --xie-result-key xie_sap_native_v3 --report-name sap_scores.json
python scripts/prepare_replay.py
/path/to/isaac-sim/python.sh scripts/render_isaac.py \
  --jobs "$GAUGE_WORKSPACE/results/sap_native_replay/jobs.json"
```

Assets, measured material parameters, time steps, PD adaptation, marker mapping and score windows remain frozen. The optional topology normalization separates disconnected rigid vertex fans **without changing any triangle's coordinates, order or winding**. It does not remove faces, weld points, use convex hulls, bake SDFs, or alter mass/inertia. For the official stretching frame this splits four shared vertex IDs (252 → 256); the native SAP admission check rejects its original nonmanifold vertex topology before solving. Per-asset source/output hashes and exact-coordinate checks are recorded. Nonmanifold edges remain explicit errors requiring asset review.

Each physical step records native backend, strict/best-effort convergence and original failure diagnostics in `diagnostics.jsonl`. Native step rejection is distinct from adapter exceptions. A completed rollout does not by itself certify correct contact behavior. Immutable recordings may be reused between profiles only when both physical scene hashes and every non-bookkeeping evaluation field match; their original profile and receipts are retained.

Replay uses recorded SDK forward kinematics and surface topology, with no physics stepping in Isaac. Analytical spheres may use the official equal-radius display mesh. Rendering follows physics serially on shared GPUs. Comparisons with previous-version IPC or previously recorded native peers must disclose those version/batch differences, including the historical Isaac textile material limitation above. Engine source and SDK binaries remain outside this repository.

The separate [SAP validation receipt](sap_0514_validation.json) records this prerelease's 55/55 completed full rollouts, 70 CPU checks, exact runtime identity and lossless normalization. It also discloses accepted best-effort steps and score-reproduction limits; completion is not a claim of official-equivalent physics or scores.

## Native SAP protocol with a frozen 32/12 budget

`prepare_sap_native.py` starts from the **native-peer recipes**, rather than the IPC adaptations. It maps static actors to stationary SDK kinematic bodies, drives moving actors with native kinematic poses and explicit twists, and retains passive dynamic actors with a null controller. It restores the official Newton's-cradle zero surface gap / 20 mm center spacing. No PD, RP, collision shrinking or gap enlargement is used.

All cases receive the same preselected SAP settings: nonlinear budget **32**, line search **12**, relative and velocity tolerances **1e-4**, swept start candidates and endpoint CCD **enabled**. Beta/sigma, linear policy, measured materials, density, mass/inertia, time steps, bending conversion and scoring remain unchanged. These are custom uniform settings, not the release defaults.

```bash
python scripts/prepare_sap_native.py --profile sap_native_protocol_0514_v4
python scripts/run_suite.py --engine xiephysic --profile sap_native_protocol_0514_v4 \
  --result-key sap_native_v4 --record-state
python scripts/score.py --xie-profile sap_native_protocol_0514_v4 \
  --xie-result-key sap_native_v4 --report-name sap_native_v4_scores.json
```

Convex admission uses original triangle support planes, surface area and volume. Only geometrically equivalent centered boxes become analytical boxes; other exact convex surfaces use the SDK's convex route; concave surfaces remain original meshes. Disconnected rigid vertex fans may be split losslessly. These rules never consult a trajectory or score. Authored USD contact exclusions use the previously audited native ownership mapping, with **zero attachment stiffness** where static hard vertex constraints provide the actual boundary condition. Filtering adds no hidden clamp force.

Native kinematic commands explicitly supply the beginning-of-step pose and body-frame twist used by SAP. The caller publishes the prescribed end pose after stepping; free/dynamic actors are never overwritten. Independent finite differences of a rotating plate's material point validate the velocity convention. Markers use direct Hungarian assignment to actual native initial positions, with a 1 cm admission threshold.

SDK 0.5.14 static `fixed_vertices` are exact stationary Dirichlet boundaries. Its moving soft attachment / position target APIs are compliant; exact moving soft-rigid attachments are explicitly unsupported. The strict protocol records 18 such configurations as `protocol_blocked`. A separate `--moving-grip compliant` profile can evaluate the SDK-native compliant attachment with the **previously declared uniform 100000 N/m** stiffness and a disclosure flag. Those reference scores must not be silently merged into strict-protocol rankings or counted as exact-boundary successes. Unsupported, failed and partial cases never receive zero scores.

Endpoint CCD is a **local soft/rod position filter**; it does not clip rigid/joint poses or generally certify EE/EF/SDF contacts, and velocity is committed separately. Enabling the option does not establish whole-scene zero penetration, momentum conservation or validity of the independently corrected mesh path. Rejection after clipping is retained with its original native diagnostics; no hidden switch-off or parameter tuning is performed.

The [native protocol validation receipt](sap_native_0514_validation.json) records 37 completed strict rollouts and 18 unsupported moving exact grips, plus a separate compliant reference with 7 completed and 11 native step rejections. All 73 CPU checks pass. Driver pose tracking and an independent point-velocity check validate the kinematic mapping. Ranking fractions disclose both eligible common cohorts and the full configuration scope; reduced coverage must not be interpreted as improved overall fidelity.

Release packaging checks are recorded in [validation.json](validation.json): 67 CPU tests, all 110 input roundtrips, dataset/mesh hashes, a full Xie 0.5.13 case with identical existing scores, and short Xie RP/Isaac textile SDK smoke runs. This is not a fresh complete four-engine benchmark.

Ranking follows the public website's two-decimal engine-error display for both local and published comparisons, with competition ties (1, 1, 3). Numeric exports may drop trailing zeroes (1.00 becomes 1); this never reduces an entire row to integer precision. Raw errors, normalization and baseline validation remain unchanged.

## Endpoint CCD off comparison

Pass `--no-endpoint-ccd` to `prepare_sap_native.py` to disable only the SAP endpoint point filter. The default remains enabled. Both strict and compliant-reference profiles retain 32/12, both 1e-4 tolerances, swept-start candidates, materials, native boundaries, drivers and scoring. Never mix the compliant reference into strict rankings.

```bash
python scripts/prepare_sap_native.py --profile sap_native_no_endpoint_0514_v5 --no-endpoint-ccd
python scripts/run_suite.py --engine xiephysic --profile sap_native_no_endpoint_0514_v5 \
  --result-key sap_native_v5 --record-state
python scripts/score.py --xie-profile sap_native_no_endpoint_0514_v5 \
  --xie-result-key sap_native_v5 --report-name sap_native_v5_scores.json
```

The experiment audits every scene/evaluation input against the enabled version. Asset paths may differ only when file bytes match. Physics batches and GPU rendering run serially. Every rollout is regenerated, with source hashes and actual runtime diagnostics retained; disabling CCD is not assumed to fix native grip support or establish valid contact behavior.

## SAP 0.5.17 rerun

Select the new SDK with `GAUGE_XIE_SDK_PATH`, `GAUGE_XIE_SDK_VERSION=0.5.17`, and the chosen release's `GAUGE_XIE_SDK_SHA256`. Preparation and rollout both verify the actual imported runtime identity. Keep the native protocol's explicit 32/12 budgets, relative and velocity tolerances of 1e-4, swept candidates enabled and endpoint CCD disabled. CUDA graph execution remains at the release default; disabling graphs is a separate diagnostic, not an unreported benchmark adjustment.

The version comparison retains byte-identical scene inputs from the previous 0.5.14 endpoint-off batch. SDK code and any unauthored release defaults may change; this is an engine-version comparison, not a single-fix ablation. For the current published benchmark, moving soft grips use the native compliant attachment implementation with uniform 100000 N/m stiffness. All completed cases are presented together; exact-grip availability is not an official GAUGE eligibility requirement. This supersedes the earlier strict-grip diagnostic classification above. The result selection is fixed by scene requirements (37 native static/rigid cases plus 18 native moving-grip cases), never by which score is lower. Keep the historical strict-grip interface probe as a diagnostic, not as a task failure. Validate full trajectories, actual backend, strict/best-effort steps, input hashes and marker admission before publishing new scores. Saved 30 Hz tetrahedral checks cannot certify every physical substep. Keep historical third-party results explicitly labeled rather than relabeling them as fresh peer runs.

To reproduce the current moving-grip mapping, explicitly prepare with `--moving-grip compliant --no-endpoint-ccd`. No PD controller is used. Native attachment implementation details remain available in configuration receipts.

The [0.5.17 validation receipt](sap_native_0517_validation.json) records all 55 complete native mappings, both comparison sources, trajectory hashes and convergence counts. This is an independent rerun, not official scoring certification.
