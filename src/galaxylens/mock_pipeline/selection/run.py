"""End-to-end selection runner: one command per target galaxy.

``python -m galaxylens.mock_pipeline.selection.run --target j2141`` runs the
scan -> L1 -> L3 -> catalog chain for the configured target. L2 (merger
history) is intentionally not in this chain — it matches the census flow
(scan -> L1 -> L3) and is available as a standalone ``merger.check_merger``
call per candidate.

Data is rooted at ``data_root`` (default ``GALAXYLENS_DATA_ROOT``); scan
catalogs land in ``<data_root>/catalog`` and L3 star cutouts in
``<data_root>/stars_l3``.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .catalog import build_catalog
from .config import get_target
from .filter import filter_l1
from .scan import scan_snapshot
from .verify import verify_one


def run_selection(target_name, data_root, outdir, workers=1, l3_workers=4,
                  max_detail=None):
    """Run the full selection chain and return the catalog rows."""
    target = get_target(target_name)
    criteria = target.criteria
    cat_dir = os.path.join(data_root, "catalog")
    stars_dir = os.path.join(data_root, "stars_l3")

    # 1) scan all configured sims x snapshots
    l1_rows = []
    for sim in criteria.sims:
        for snap in criteria.snapshots:
            json_path, _ = scan_snapshot(sim, snap, criteria.mass_min, cat_dir,
                                         workers=workers, max_detail=max_detail)
            if json_path and os.path.exists(json_path):
                with open(json_path) as f:
                    rows = json.load(f)
                for r in rows:
                    r = dict(r)
                    r.setdefault("snap", snap)
                    r.setdefault("sim", sim)
                l1_rows.extend(rows)

    # 2) L1 filter
    cands = filter_l1(l1_rows, criteria)
    print(f"L1 candidates: {len(cands)}", flush=True)

    # 3) L3 verify (download 3-field cutout + disk kinematics)
    results = []
    if l3_workers <= 1:
        for c in cands:
            results.append(verify_one(c["sim"], c["snap"], c["subhalo_id"],
                                      stars_dir))
    else:
        with ThreadPoolExecutor(max_workers=l3_workers) as pool:
            futs = {pool.submit(verify_one, c["sim"], c["snap"],
                                c["subhalo_id"], stars_dir): c for c in cands}
            for fut in as_completed(futs):
                results.append(fut.result())

    # 4) assemble the SKIRT-consumable catalog (disk-dominant subset)
    os.makedirs(outdir, exist_ok=True)
    catalog_path = os.path.join(outdir, "catalog.csv")
    rows = build_catalog(results, catalog_path,
                         criteria.f_disk_min, criteria.f_disk_disk)
    print(f"catalog: {len(rows)} rows -> {catalog_path}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="TNG disk-lens analog selection")
    ap.add_argument("--target", default="j2141")
    ap.add_argument("--data-root",
                    default=os.environ.get("GALAXYLENS_DATA_ROOT", "."))
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--l3-workers", type=int, default=4)
    ap.add_argument("--max-detail", type=int, default=None,
                    help="smoke test: cap the number of subhalo details per snapshot")
    args = ap.parse_args()
    run_selection(args.target, args.data_root, args.outdir,
                  workers=args.workers, l3_workers=args.l3_workers,
                  max_detail=args.max_detail)


if __name__ == "__main__":
    main()
