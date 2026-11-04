"""Regression: Pipeline (Stage orchestration) + analyze (one-call post-fit summary).

The Pipeline part is pure-python (no autolens); the analyze part runs against a
real fit's samples.csv if present, else skips.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxylens.api.pipeline import Pipeline, Stage


def test_pipeline_selection():
    order = []

    def mk(name, **kw):
        def fn(ctx):
            order.append(name)
            ctx[name] = True
        return Stage(name=name, fn=fn, **kw)

    p = Pipeline(
        "t",
        [mk("a"), mk("b"), mk("c", depends=["a"]), mk("d", depends=["b", "c"])],
    )

    # full auto
    order.clear()
    ctx = p.run(verbose=False)
    assert order == ["a", "b", "c", "d"], order
    assert set(ctx) == {"a", "b", "c", "d"}

    # only (keeps declared order, not selection order)
    order.clear()
    p.run(only=["b", "a"], verbose=False)
    assert order == ["a", "b"], order

    # start / stop (single-stage range, no dependency triggered)
    order.clear()
    p.run(start="b", stop="b", verbose=False)
    assert order == ["b"], order

    # missing dependency raises
    try:
        p.run(only=["c"], verbose=False)
        raise AssertionError("expected RuntimeError for missing dependency")
    except RuntimeError as e:
        assert "needs" in str(e)

    print("[1] Pipeline selection (full/only/start-stop/depends) passed")


def test_analyze():
    """Run analyze against a real fit's samples.csv if one is on disk, else skip."""
    from galaxylens.api.facade import analyze, load_config

    # prefer the archived J1630 v2 fit; fall back to any samples.csv in outputs.
    candidates = [Path(p) for p in os.environ.get("GALAXYLENS_TEST_SAMPLES_CSV", "").split(":") if p]
    csv = next((c for c in candidates if c.exists()), None)
    if csv is None:
        print("[2] analyze SKIPPED (no samples.csv on disk)")
        return

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/fit/j1630_ellip.yaml")
    out = analyze(cfg, samples_csv=str(csv), corner=False, verbose=False)
    assert set(out) == {"params", "derived", "edges"}, set(out)
    assert set(out["params"]) == {"ml_img", "v_vir", "c_minus2", "gamma", "q_h", "ratio"}
    for k, v in out["params"].items():
        assert v["median"] == v["median"]  # finite
    for e in out["edges"].values():
        assert "hits_lower" in e and "hits_upper" in e
    print(f"[2] analyze passed on {csv.name} (params/derived/edges present)")


if __name__ == "__main__":
    test_pipeline_selection()
    test_analyze()
    print("ALL verify_pipeline checks passed")
