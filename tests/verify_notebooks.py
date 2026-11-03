"""Smoke test: every `# %%` tutorial notebook runs as a script to exit 0.

Guards the tutorials against silent breakage when the API changes (e.g. a
signature change breaks a demo and no test notices). Each notebook runs with
`GALAXYLENS_DATA_ROOT` pointed at a temp dir so the heavy/optional sections
take their skip path; offline sections must complete. The mock demo is skipped
when its optional deps are absent.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"


def _run(name: str):
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["GALAXYLENS_DATA_ROOT"] = tempfile.mkdtemp()   # force offline skip paths
    env["GALAXYLENS_OUTPUT_ROOT"] = tempfile.mkdtemp()
    r = subprocess.run(
        [sys.executable, str(NOTEBOOKS / name)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return r


def main():
    failures = []
    for name in ["identify_demo.py", "fit_demo.py", "mock_pipeline_demo.py"]:
        r = _run(name)
        if r.returncode != 0:
            failures.append(name)
            print(f"[FAIL] {name} (exit {r.returncode})")
            print(r.stdout[-1500:])
            print(r.stderr[-1500:])
        else:
            print(f"[ok]   {name}")

    if failures:
        raise SystemExit(f"tutorial notebooks failed: {failures}")
    print("ALL notebook tutorials ran to exit 0")


if __name__ == "__main__":
    main()
