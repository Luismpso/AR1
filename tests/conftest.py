"""
pytest configuration for the AR1 portfolio test-suite.

Makes the repository root importable so that ``from AR1.xxx import …`` works
when running ``pytest`` from inside ``AR1/`` or from the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
