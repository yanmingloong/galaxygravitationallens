"""Regression: gl.mock (mock-closure Pipeline) assembles + each stage guards its inputs.

Pure-python (no TNG/SKIRT/autolens): verifies the orchestration layer and the
per-stage pre-flight checks. The heavy stages (truth/render/arc/fit/closure)
need external data, so this only exercises the guard paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import galaxylens as gl


def test_mock_pipeline():
    p = gl.mock("j2141", "/tmp/data", "/tmp/out")
    assert p.name == "mock:j2141"
    assert p.stage_names == ["select", "pick", "truth", "render", "make_arc",
                             "mock_fit", "closure"], p.stage_names

    # pick guards: missing select
    try:
        p.run(only=["pick"], verbose=False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "select" in str(e)

    # pick works with a seeded catalog
    ctx = {"select": [{"sim": "TNG50", "snap": 90, "subhalo_id": 348489}]}
    p.run(only=["pick"], ctx=ctx, verbose=False)
    assert ctx["pick"]["subhalo_id"] == 348489

    # truth guards: missing particle paths
    try:
        p.run(only=["truth"], ctx=ctx, verbose=False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "star_h5" in str(e)

    # mock_fit guards: missing recipe (also depends on render/make_arc)
    try:
        p.run(only=["mock_fit"], ctx={}, verbose=False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "render" in str(e) or "mock_cfg_path" in str(e)

    print("[mock] pipeline assembly + stage guards passed")


if __name__ == "__main__":
    test_mock_pipeline()
    print("ALL verify_mock_orchestrate checks passed")
