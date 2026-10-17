"""Smoke test: the mock_pipeline tutorial must keep running end-to-end.

Guards against silent drift. A library API change (e.g. ``write_ski``'s
signature) once broke the demo without any test noticing, because the demo
is documentation and is never executed by the test suite. Running it here
keeps it honest: any API change that breaks the tutorial now fails CI.

The run exercises the offline path (no TNG key, no particle data), so the
heavy sections (selection / particle conversion / SKIRT) must skip cleanly
while the pure-local sections (config, .ski generation, fake HST, measures)
must execute.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO = REPO_ROOT / "notebooks" / "mock_pipeline_demo.py"


def test_mock_pipeline_demo_runs():
    assert DEMO.is_file(), f"demo not found: {DEMO}"
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env.pop("TNG_API_KEY", None)   # force the offline skip path
        env["HOME"] = tmp              # ensure ~/.tng_api_key is not found
        env["GALAXYLENS_DATA_ROOT"] = tmp
        env["GALAXYLENS_OUTPUT_ROOT"] = tmp
        r = subprocess.run(
            [sys.executable, str(DEMO)],
            env=env, capture_output=True, text=True, timeout=300,
        )
        assert r.returncode == 0, (
            f"mock_pipeline_demo.py exited {r.returncode}\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
