"""
Pytest bootstrap.

The unit suite exercises only pure-Python plumbing in `train.py`, `score.py`,
and `evaluate_on_holdout.py`. To keep these tests runnable in the dev container
(no GPU, no large model wheels), we install lightweight stubs in `sys.modules`
for the heavy ML deps **before any test imports the production modules**.

Two classes are subclassed at module load time — `transformers.Trainer`
(`WeightedTrainer`) and `transformers.integrations.MLflowCallback`
(`AzureMLSafeCallback`). Subclassing a `MagicMock` instance raises at class
creation, so those two are replaced by real (empty) Python classes; everything
else is a `MagicMock`.

If the real packages happen to be installed (e.g. when the suite is run inside
the AML training container), `setdefault` leaves them alone — tests still work.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make the repo root importable so `import train`, `import score` resolve.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_stubs() -> None:
    class _StubTrainer:
        """Stand-in for transformers.Trainer — must be subclassable."""

        def __init__(self, *args, **kwargs) -> None:
            self._args = args
            self._kwargs = kwargs

        def remove_callback(self, *a, **k) -> None: ...
        def add_callback(self, *a, **k) -> None: ...

    class _StubMLflowCallback:
        """Stand-in for transformers.integrations.MLflowCallback."""

        def setup(self, *a, **k) -> None: ...

    torch_stub = MagicMock(name="torch")
    transformers_stub = MagicMock(name="transformers")
    transformers_stub.Trainer = _StubTrainer

    integrations_stub = MagicMock(name="transformers.integrations")
    integrations_stub.MLflowCallback = _StubMLflowCallback

    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("transformers", transformers_stub)
    sys.modules.setdefault("transformers.integrations", integrations_stub)

    for mod in (
        "datasets",
        "sklearn",
        "sklearn.metrics",
        "sklearn.model_selection",
        "mlflow",
        "mlflow.transformers",
        "pandas",
        "numpy",
    ):
        sys.modules.setdefault(mod, MagicMock(name=mod))


_install_stubs()
