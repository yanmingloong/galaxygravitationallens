"""CLI runner: ``python -m galaxylens.api.runner --config <path>``.

This is what the slurm backend submits; it loads the recipe and runs ``gl.fit``.
Kept separate from the facade so a cluster job has a minimal, self-contained
entry point (no interactive shell).
"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a galaxylens fit recipe")
    parser.add_argument("--config", required=True, help="path to the fit-recipe YAML")
    parser.add_argument(
        "--smoke", action="store_true", help="reduce n_live for a quick smoke run"
    )
    args = parser.parse_args(argv)

    from galaxylens.api.facade import fit, load_config

    cfg = load_config(args.config)
    # The runner executes the actual fit on the cluster; a recipe YAML that says
    # backend: slurm (so a user's local gl.fit submits it) must be run as local
    # here, otherwise the runner would try to re-submit itself.
    cfg.search.backend = "local"
    if args.smoke:
        cfg.search.n_live = min(cfg.search.n_live, 50)

    result = fit(cfg)

    print("\n=== fit results ===")
    print(
        f"beta_z = {result.beta_z[1]:.3f} "
        f"[-{result.beta_z[1] - result.beta_z[0]:.3f} +{result.beta_z[2] - result.beta_z[1]:.3f}]"
    )
    print(f"c_-2 = {result.c_minus2.median:.2f}, gamma = {result.gamma.median:.3f}")
    print(f"v_vir = {result.v_vir.median:.1f} km/s, q_h = {result.q_h.median:.3f}")
    print(f"M* = {result.m_star[1]:.3e} Msun")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
