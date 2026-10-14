"""Select the independently installed Xie SDK; native peers never import this."""

import hashlib
import importlib
import os
import sys
from pathlib import Path


def load_sdk():
    path = Path(os.environ["GAUGE_XIE_SDK_PATH"]).expanduser().resolve()
    sys.meta_path[:] = [
        f for f in sys.meta_path if not type(f).__module__.startswith("_editable")
    ]
    for name in list(sys.modules):
        if name == "xiephy_gpu" or name.startswith("xiephy_gpu."):
            del sys.modules[name]
    sys.path.insert(0, str(path))
    importlib.invalidate_caches()
    sdk = importlib.import_module("xiephy_gpu")
    expected = os.environ.get("GAUGE_XIE_SDK_VERSION", "0.5.17")
    if sdk.__version__ != expected or not Path(sdk.__file__).resolve().is_relative_to(
        path
    ):
        raise ValueError(f"Preparation requires the explicitly selected {expected} SDK")
    expected_sha = os.environ.get("GAUGE_XIE_SDK_SHA256")
    if (
        expected_sha
        and hashlib.sha256(Path(sdk._C.__file__).read_bytes()).hexdigest()
        != expected_sha
    ):
        raise ValueError(
            "Preparation SDK binary SHA256 differs from the selected release"
        )
    return sdk
