# GAUGE community reproduction

The current author-facing entry point is [the paper-version reproduction](reproduction/README.zh-CN.md). It provides Isaac Sim **6.0.0** and Newton **1.3.0** adapters, a pinned dataset inventory, literal-paper scoring, recorded results and a list of discrepancies. The paper's Genesis **1.12.0** version remains unconfirmed; no other version is substituted.

This is an independent implementation, **not the authors' evaluator**. Matching versions and measured materials does not establish identical scene construction or scores. The [current discrepancies](reproduction/DISCREPANCIES.md) distinguish observed differences from hypotheses.

## Current native reproduction

```bash
cd evaluation
python -m pip install -r requirements.txt
export GAUGE_WORKSPACE="$PWD/_work"
python scripts/fetch_dataset.py
python scripts/prepare.py --mode native
python scripts/reproduce.py audit
python scripts/reproduce.py run --engine newton --python /path/to/newton-1.3.0/bin/python
python scripts/reproduce.py run --engine isaac --python /path/to/isaac-6.0.0/python.sh --isaac-root /path/to/isaac-6.0.0
python scripts/reproduce.py verify
python scripts/reproduce.py score
```

Install SDKs independently. Run GPU simulations serially. See [reproduction instructions](reproduction/README.zh-CN.md) for exact builds, dependencies, single-case commands, reference-window semantics and result archives. Generated data/configs/results, SDK binaries, engine code and credentials are ignored by Git.

## Optional XiePhysic SDK experiments

The existing XiePhysic integration is kept separately. It is **not used to construct the Isaac or Newton reproduction**. Preparation without `--mode` materializes both profiles. Xie moving apparatus uses its declared PD adaptation, while native peers use their own boundary/control interfaces. The Xie IPC cradle's 2 mm gap is an explicit Xie experiment only.

```bash
python scripts/prepare.py
export GAUGE_XIE_SDK_PATH=/path/to/installed-sdk
export GAUGE_XIE_SDK_VERSION=0.5.17
python scripts/prepare_sap_native.py --profile sap_native_0517 --moving-grip compliant --no-endpoint-ccd
python scripts/run_suite.py --engine xiephysic --profile sap_native_0517 --result-key sap_native_0517 --record-state
```

The current SAP preset uses the declared uniform 32 nonlinear iterations / 12 line searches, relative and velocity tolerances of 1e-4, swept-start candidates enabled and endpoint CCD disabled. These are experimental settings, not SDK defaults. Moving soft grips use native compliant attachments with uniform 100000 N/m stiffness; exact-grip availability is not a GAUGE eligibility rule. [The existing 0.5.17 receipt](sap_native_0517_validation.json) belongs to that experiment, not to the paper-version native run.

`scripts/score.py` retains the earlier `published_table_reproduction_v3` interpretation for existing Xie experiments. It uses leave-one-out textile/foam baselines and a bending-chain observable; **do not use it to score the author-facing reproduction or merge its ranks with literal-paper results**. `scripts/reproduce.py score` is the current reproduction scorer. No aggregate or author-certified rank is produced there.

## Checks and license

```bash
python -m pytest tests -q
```

Only the current third-party adapters are distributed. Superseded SDK adapters and obsolete validation receipts were removed; earlier experiments remain available in Git history. Shared metric and geometry utilities are tested without an engine SDK. Contributions use the repository MIT license. External SDKs and the [official dataset](https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset) retain their own terms; the repository does not redistribute them.
