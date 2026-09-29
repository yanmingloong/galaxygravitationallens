#!/usr/bin/env python3
"""
Phase D — 参数丰富
====================
为 Tier 0/1/2 候选体查询外部参数（sigma, theta_E, Sersic 等），
生成带版本号的丰富星表。

查询源:
- SDSS DR17: 速度弥散度、恒星质量、光谱分类
- Simard+2011: Sérsic n, b/a, Re, PA
- 文献/星表: Einstein 半径

输出: 04_enriched/tier012_v{N}.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd


from .config import (
    SCHEMA_COLUMNS, MASTER_CLASSIFIED_PATH, DIR_ENRICHED,
    SIMARD_VIZIER_ID, AUGER2009_VIZIER_ID, PROJECT_ROOT,
)
from .classify import classify_tier, reclassify, sort_catalog


# ================================================================
# 版本管理
# ================================================================

def get_latest_version() -> int:
    """扫描 04_enriched/ 目录，返回当前最大版本号。"""
    if not os.path.exists(DIR_ENRICHED):
        return 0
    versions = []
    for f in os.listdir(DIR_ENRICHED):
        if f.startswith("tier012_v") and f.endswith(".csv"):
            try:
                v = int(f.replace("tier012_v", "").replace(".csv", ""))
                versions.append(v)
            except ValueError:
                pass
    return max(versions) if versions else 0


def next_version() -> int:
    """返回下一个版本号。"""
    return get_latest_version() + 1


def current_enriched_path(version: int = None) -> str:
    """返回当前（或最新）enriched CSV 路径。"""
    if version is None:
        version = get_latest_version()
    if version == 0:
        return os.path.join(DIR_ENRICHED, "tier012_v1.csv")
    return os.path.join(DIR_ENRICHED, f"tier012_v{version}.csv")


# ================================================================
# 参数丰富
# ================================================================

def enrich_from_existing(df: pd.DataFrame,
                          old_enriched_path: str = None) -> pd.DataFrame:
    """
    从已有的旧版 enriched CSV 中填补参数（向量化版本）。

    如果旧版中某个系统有 sigma/theta_E/M* 而新版没有，直接合并过来。
    用于增量更新——新版本的未知字段从旧版本继承。
    """
    if old_enriched_path is None:
        old_enriched_path = current_enriched_path()

    if not os.path.exists(old_enriched_path):
        print(f"  No existing enriched catalog at {old_enriched_path}, skipping merge", flush=True)
        return df

    old = pd.read_csv(old_enriched_path, low_memory=False)
    print(f"  Loading previous version: {os.path.basename(old_enriched_path)} ({len(old)} rows)", flush=True)

    from .utils import normalize_name

    # 要继承的字段
    inherit_cols = [
        "sigma_km_s", "sigma_err", "sigma_flag",
        "theta_E_arcsec", "theta_E_err",
        "logMstar_total", "logMstar_bulge", "logMstar_disk",
        "B_D_ratio", "n_sersic", "n_sersic_err", "Re_arcsec",
        "Re_bulge_arcsec", "Re_disk_arcsec", "q_bulge", "q_disk",
        "PA_disk_deg", "mag_g", "mag_r", "mag_i", "g_r_color",
        "Halpha_flux", "spec_class", "spec_sn",
        "imaging_available", "ifu_available", "IMF",
        "notes",
    ]

    # 向量化名称归一化
    df["_name_norm"] = df["name"].apply(normalize_name)
    old["_name_norm"] = old["name"].apply(normalize_name)

    # 如果旧版有重复名称，只保留第一个非 NaN 值
    # (同一天体可能出现在多个源中)
    old = old.drop_duplicates(subset=["_name_norm"], keep="first")

    # 只在目标为 NaN 的列上继承
    inherited = 0
    for col in inherit_cols:
        if col not in old.columns or col not in df.columns:
            continue
        # 构建旧版 lookup: normalized_name -> value
        old_lookup = old.set_index("_name_norm")[col].dropna()
        if len(old_lookup) == 0:
            continue
        # 找目标中该列为 NaN 的行
        need_fill = df[col].isna()
        if not need_fill.any():
            continue
        # 对需要填充的行，查找旧版值
        fill_mask = need_fill & df["_name_norm"].isin(old_lookup.index)
        if fill_mask.any():
            df.loc[fill_mask, col] = df.loc[fill_mask, "_name_norm"].map(old_lookup)
            inherited += fill_mask.sum()

    df = df.drop(columns=["_name_norm"], errors="ignore")
    print(f"  Inherited {inherited} field values from previous version", flush=True)

    return df


def enrich_sdss(df: pd.DataFrame, query_tiers=(1, 2)) -> pd.DataFrame:
    """
    对指定 Tier 的系统查询 SDSS DR17 参数。

    查询源: SDSS SkyServer SQL API (HTTP, 无需 astroquery)
    查询内容:
      1. SpecObj — z, class, snMedian_r
      2. stellarMassPCAWiscBC03 — logMass (Chabrier IMF)
      3. GalSpecLine — sigma, sigmaErr (速度弥散度)
    4. Photoz table (for photometric redshift if spec-z missing)

    Uses astropy for coordinate cross-matching.
    """
    import requests
    import io
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    from .config import SDSS_SKYSERVER_URL

    targets = df[df["disk_tier"].isin(query_tiers)].copy()
    if len(targets) == 0:
        print(f"  SDSS: No Tier {query_tiers} targets to query")
        return df

    print(f"\n  ── SDSS DR17 enrichment ──")
    print(f"  Targets: {len(targets)} systems (Tier {query_tiers})")

    # ============================================================
    # 1. Query SpecObj (spectroscopic redshift, class, S/N)
    # ============================================================
    print(f"  [1/3] Querying SpecObj...")

    spec_df_all = []
    batch_size = 20  # SDSS SQL length limit

    for i in range(0, len(targets), batch_size):
        batch = targets.iloc[i:i + batch_size]
        batch_conds = []
        for _, row in batch.iterrows():
            ra, dec = float(row["ra_deg"]), float(row["dec_deg"])
            batch_conds.append(
                f"(s.ra BETWEEN {ra - 0.001} AND {ra + 0.001} "
                f"AND s.dec BETWEEN {dec - 0.001} AND {dec + 0.001})"
            )

        sql = """
        SELECT
            s.specObjID, s.ra, s.dec, s.z, s.zErr, s.zWarning,
            s.class, s.subClass,
            s.plate, s.mjd, s.fiberID,
            s.snMedian_u, s.snMedian_g, s.snMedian_r, s.snMedian_i, s.snMedian_z
        FROM SpecObj AS s
        WHERE
        """ + " OR ".join(batch_conds)

        try:
            resp = requests.get(SDSS_SKYSERVER_URL, params={
                "cmd": sql, "format": "csv"
            }, timeout=60)
            if resp.status_code == 200 and len(resp.text) > 100:
                batch_df = pd.read_csv(io.StringIO(resp.text), comment='#',
                                       skipinitialspace=True)
                if len(batch_df) > 0 and len(batch_df.columns) > 1:
                    spec_df_all.append(batch_df)
        except Exception as e:
            print(f"    Batch {i // batch_size + 1}: ERROR {e}")

        time.sleep(0.3)

    if spec_df_all:
        spec_df = pd.concat(spec_df_all, ignore_index=True)
        print(f"    Found {len(spec_df)} SDSS spectra")
    else:
        print(f"    No SDSS spectra found — targets may be outside SDSS footprint")
        return df

    # ============================================================
    # 2. Query stellar masses (try multiple tables)
    # ============================================================
    print(f"  [2/3] Querying stellar masses...")

    # Find specObjID column
    spec_id_col = None
    for c in spec_df.columns:
        if c.lower() == "specobjid":
            spec_id_col = c
            break
    if spec_id_col is None:
        print(f"    WARNING: specObjID column not found. Columns: {list(spec_df.columns)}")
        print(f"    Skipping mass+sigma queries.")
        return df

    spec_ids = spec_df[spec_id_col].dropna().astype(int).tolist()

    # Try multiple stellar mass tables — DR17 lacks mass tables, use DR16
    mass_tables = [
        ("stellarMassPCAWiscBC03", "specObjID", "mstellar_median", "mstellar_err"),
        ("stellarMassFSPSGranEarlyDust", "specObjID", "logM", "NULL as logMassErr"),
        ("stellarMassStarFormingPort", "specObjID", "logM", "NULL as logMassErr"),
        ("galSpecExtra", "specObjID", "lgm_tot_p50", "NULL as logMassErr"),
    ]
    DR16_URL = "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
    mass_df = pd.DataFrame()
    for table, id_col, mass_col, masserr_col in mass_tables:
        mass_df_all = []
        for i in range(0, len(spec_ids), 500):
            batch_ids = spec_ids[i:i + 500]
            ids_str = ",".join(str(x) for x in batch_ids)
            # DR16 for mass tables (DR17 doesn't have them)
            sql_mass = f"""
            SELECT TOP 10000 {id_col}, {mass_col}, {masserr_col}
            FROM {table}
            WHERE {id_col} IN ({ids_str})
            """
            try:
                resp = requests.get(DR16_URL, params={
                    "cmd": sql_mass, "format": "csv"
                }, timeout=60)
                if resp.status_code == 200 and len(resp.text) > 100:
                    batch_df = pd.read_csv(io.StringIO(resp.text), comment='#',
                                           skipinitialspace=True)
                    if len(batch_df) > 0 and id_col.lower() in [c.lower() for c in batch_df.columns]:
                        mass_df_all.append(batch_df)
            except Exception:
                pass
            time.sleep(0.2)
        if mass_df_all:
            mass_df = pd.concat(mass_df_all, ignore_index=True)
            print(f"    Found {len(mass_df)} stellar masses from {table} (DR16)")
            break
        else:
            print(f"    {table}: no results")

    if len(mass_df) == 0:
        print(f"    No stellar masses available via SkyServer.")
    else:
        print(f"    Total stellar masses: {len(mass_df)}")

    # ============================================================
    # 3. Query velocity dispersions
    # ============================================================
    print(f"  [3/3] Querying velocity dispersions...")

    vdisp_df_all = []

    for i in range(0, len(spec_ids), 500):
        batch_ids = spec_ids[i:i + 500]
        ids_str = ",".join(str(x) for x in batch_ids)

        # Try GalSpecLine first
        sql_vdisp = f"""
        SELECT specObjID, sigma, sigmaErr, veldisp, veldispErr
        FROM GalSpecLine
        WHERE specObjID IN ({ids_str})
        """

        try:
            resp = requests.get(SDSS_SKYSERVER_URL, params={
                "cmd": sql_vdisp, "format": "csv"
            }, timeout=60)
            if resp.status_code == 200 and len(resp.text) > 100:
                batch_df = pd.read_csv(io.StringIO(resp.text), comment='#',
                                       skipinitialspace=True)
                if len(batch_df) > 0 and len(batch_df.columns) > 1:
                    vdisp_df_all.append(batch_df)
        except Exception:
            pass
        time.sleep(0.2)

    if vdisp_df_all:
        vdisp_df = pd.concat(vdisp_df_all, ignore_index=True)
        print(f"    Found {len(vdisp_df)} velocity dispersions (GalSpecLine)")
    else:
        # Fallback: query velDisp from SpecObj directly
        print(f"    GalSpecLine empty, trying SpecObj velDisp...")
        vdisp_df_all2 = []
        for i in range(0, len(spec_ids), 500):
            batch_ids = spec_ids[i:i + 500]
            ids_str = ",".join(str(x) for x in batch_ids)
            sql_vd = f"""
            SELECT specObjID, velDisp, velDispErr
            FROM SpecObj
            WHERE specObjID IN ({ids_str})
              AND velDisp > 0
            """
            try:
                resp = requests.get(SDSS_SKYSERVER_URL, params={
                    "cmd": sql_vd, "format": "csv"
                }, timeout=60)
                if resp.status_code == 200 and len(resp.text) > 100:
                    batch_df = pd.read_csv(io.StringIO(resp.text), comment='#',
                                           skipinitialspace=True)
                    if len(batch_df) > 0 and len(batch_df.columns) > 1:
                        vdisp_df_all2.append(batch_df)
            except Exception:
                pass
            time.sleep(0.2)
        vdisp_df = pd.concat(vdisp_df_all2, ignore_index=True) if vdisp_df_all2 else pd.DataFrame()
        print(f"    Found {len(vdisp_df)} velocity dispersions (SpecObj fallback)")

    # ============================================================
    # 4. Cross-match back to targets and fill DataFrame
    # ============================================================
    print(f"  Cross-matching SDSS results...")

    # Build SpecObj lookup by coordinate
    tier_coords = SkyCoord(ra=targets["ra_deg"].values * u.deg,
                           dec=targets["dec_deg"].values * u.deg)
    spec_coords = SkyCoord(ra=spec_df["ra"].values * u.deg,
                           dec=spec_df["dec"].values * u.deg)

    from astropy.coordinates import match_coordinates_sky
    idx, sep2d, _ = match_coordinates_sky(tier_coords, spec_coords)
    matched = sep2d < 3 * u.arcsec
    n_matched = matched.sum()
    print(f"    Matched: {n_matched}/{len(targets)} within 3 arcsec")

    # Apply matches
    spec_indices = idx[matched]
    target_indices = targets.index[matched]

    # Fill spec_class and spec_sn
    for ti, si in zip(target_indices, spec_indices):
        spec_row = spec_df.iloc[si]
        df.at[ti, "spec_class"] = str(spec_row.get("class", ""))
        df.at[ti, "spec_sn"] = float(spec_row.get("snMedian_r", np.nan)) if pd.notna(spec_row.get("snMedian_r")) else np.nan

        # redshift (if missing)
        if pd.isna(df.at[ti, "z_lens"]):
            z_val = spec_row.get("z")
            if pd.notna(z_val) and float(z_val) > 0:
                df.at[ti, "z_lens"] = float(z_val)
                df.at[ti, "z_lens_flag"] = "sdss_spec"

    # Find ID column helper (SDSS SkyServer returns inconsistent case)
    def _get_id_col(df_part, label):
        for c in df_part.columns:
            if c.lower() == "specobjid":
                return c
        print(f"    WARNING: specObjID column not found in {label}. Columns: {list(df_part.columns)}")
        return None

    # Get the correct column names for each result
    spec_id_col_spec = _get_id_col(spec_df, "spec_df")
    if spec_id_col_spec is None:
        return df

    # Fill sigma
    sigma_filled = 0
    if len(vdisp_df) > 0:
        spec_ids_for_lookup = set(int(spec_df.iloc[i][spec_id_col_spec]) for i in range(len(spec_df)))
        vdisp_id_col = _get_id_col(vdisp_df, "vdisp_df")
        if vdisp_id_col:
            vdisp_lookup = {}
            for _, vrow in vdisp_df.iterrows():
                sid = int(vrow[vdisp_id_col])
                # prefer GalSpecLine sigma, fallback to veldisp or SpecObj velDisp
                if "sigma" in vrow.index and pd.notna(vrow["sigma"]) and float(vrow["sigma"]) > 0:
                    vdisp_lookup[sid] = (float(vrow["sigma"]),
                                         float(vrow.get("sigmaErr", np.nan)) if pd.notna(vrow.get("sigmaErr")) else np.nan)
                elif "veldisp" in vrow.index and pd.notna(vrow["veldisp"]) and float(vrow["veldisp"]) > 0:
                    vdisp_lookup[sid] = (float(vrow["veldisp"]),
                                         float(vrow.get("veldispErr", np.nan)) if pd.notna(vrow.get("veldispErr")) else np.nan)
                elif "velDisp" in vrow.index and pd.notna(vrow["velDisp"]) and float(vrow["velDisp"]) > 0:
                    vdisp_lookup[sid] = (float(vrow["velDisp"]),
                                         float(vrow.get("velDispErr", np.nan)) if pd.notna(vrow.get("velDispErr")) else np.nan)

            for ti, si in zip(target_indices, spec_indices):
                spec_row = spec_df.iloc[si]
                sid = int(spec_row[spec_id_col_spec])
                if sid in vdisp_lookup and (pd.isna(df.at[ti, "sigma_km_s"]) or df.at[ti, "sigma_km_s"] == 0):
                    sigma_val, sigma_err = vdisp_lookup[sid]
                    df.at[ti, "sigma_km_s"] = sigma_val
                    df.at[ti, "sigma_err"] = sigma_err
                    df.at[ti, "sigma_flag"] = "sdss_dr17"
                    sigma_filled += 1

    # Fill stellar masses
    mass_filled = 0
    if len(mass_df) > 0:
        mass_id_col = _get_id_col(mass_df, "mass_df")
        if mass_id_col:
            # Detect mass column (varies by table: mstellar_median, logM, lgm_tot_p50)
            mass_col = None
            for candidate in ["mstellar_median", "logM", "lgm_tot_p50", "logMass"]:
                if candidate in mass_df.columns:
                    mass_col = candidate
                    break
            # Detect error column
            mass_err_col = None
            for candidate in ["mstellar_err", "logMassErr"]:
                if candidate in mass_df.columns:
                    mass_err_col = candidate
                    break

            if mass_col:
                mass_lookup = {}
                for _, mrow in mass_df.iterrows():
                    sid = int(mrow[mass_id_col])
                    if pd.notna(mrow.get(mass_col)):
                        err = float(mrow[mass_err_col]) if mass_err_col and pd.notna(mrow.get(mass_err_col)) else np.nan
                        mass_lookup[sid] = (float(mrow[mass_col]), err)

                for ti, si in zip(target_indices, spec_indices):
                    spec_row = spec_df.iloc[si]
                    sid = int(spec_row[spec_id_col_spec])
                    if sid in mass_lookup and pd.isna(df.at[ti, "logMstar_total"]):
                        logM, logM_err = mass_lookup[sid]
                        df.at[ti, "logMstar_total"] = logM
                        df.at[ti, "logMstar_total_err"] = logM_err
                        df.at[ti, "IMF"] = "Chabrier"
                        mass_filled += 1

    print(f"    sigma filled: {sigma_filled}, logMstar filled: {mass_filled}")

    return df


def enrich_simard2011(df: pd.DataFrame, query_tiers=(1, 2)) -> pd.DataFrame:
    """
    对指定 Tier 的系统查询 Simard+2011 Sérsic 参数。

    查询源: VizieR J/ApJS/196/11/table3 (~1.1M SDSS galaxies)
    方法:
      1. 尝试通过 astroquery.vizier 下载全星表并缓存
      2. 若 astroquery 不可用，尝试 SDSS CAS PhotoObj fallback (fracDev, modelMag)

    填入字段: n_sersic, n_sersic_err, Re_arcsec, q_disk, PA_disk_deg
    """
    targets = df[df["disk_tier"].isin(query_tiers)]
    if len(targets) == 0:
        print(f"  Simard+2011: No Tier {query_tiers} targets")
        return df

    print(f"\n  ── Simard+2011 enrichment ──")
    print(f"  Targets: {len(targets)} systems (Tier {query_tiers})")

    from .config import PROJECT_ROOT

    simard_cache_path = os.path.join(PROJECT_ROOT, "01_raw", "simard2011", "simard2011_cache.csv")
    os.makedirs(os.path.dirname(simard_cache_path), exist_ok=True)

    # ============================================================
    # Method 1: Full catalog via astroquery.vizier
    # ============================================================
    try:
        from astroquery.vizier import Vizier
        HAVE_VIZIER = True
    except ImportError:
        HAVE_VIZIER = False
        print(f"    astroquery not available — install with: pip install astroquery")

    simard_catalog = None

    if HAVE_VIZIER:
        if os.path.exists(simard_cache_path):
            print(f"    Loading cached Simard+2011 from {simard_cache_path}")
            simard_catalog = pd.read_csv(simard_cache_path)
            print(f"    Loaded {len(simard_catalog)} galaxies from cache")
        else:
            print(f"    Downloading Simard+2011 full catalog (~1.1M galaxies, ~50MB)...")
            print(f"    This may take 2-5 minutes...")
            try:
                Vizier.ROW_LIMIT = -1
                Vizier.TIMEOUT = 300
                catalog_list = Vizier.get_catalogs("J/ApJS/196/11/table3")
                if catalog_list and len(catalog_list) > 0:
                    table = catalog_list[0]
                    print(f"    Downloaded: {len(table)} galaxies")
                    # Convert to DataFrame
                    rows = []
                    for row in table:
                        d = {}
                        for col in ["_RA", "_DE", "ng", "e_ng", "e", "e_e",
                                     "phi", "e_phi", "Rhlr", "Rchl_r", "Scale"]:
                            try:
                                val = row[col]
                                if val is not None and not np.ma.is_masked(val):
                                    d[col] = float(val)
                            except (KeyError, ValueError, TypeError):
                                pass
                        # 派生: q_disk = 1 - e, Re_arcsec = Rhlr / Scale
                        if "e" in d:
                            d["q_disk"] = 1.0 - d["e"]
                        if "Rhlr" in d and "Scale" in d and d["Scale"] > 0:
                            d["Re_arcsec"] = d["Rhlr"] / d["Scale"]
                        rows.append(d)
                    simard_catalog = pd.DataFrame(rows)
                    simard_catalog.to_csv(simard_cache_path, index=False)
                    print(f"    Cached to {simard_cache_path}")
                else:
                    print(f"    VizieR returned no data for Simard+2011")
            except Exception as e:
                print(f"    Failed to download Simard+2011: {e}")
                print(f"    Will try per-object queries instead...")
                simard_catalog = None

    # ============================================================
    # Cross-match if we have the catalog
    # ============================================================
    if simard_catalog is not None and len(simard_catalog) > 0:
        print(f"    Cross-matching {len(targets)} targets against {len(simard_catalog)} Simard galaxies...")

        # Detect column names (support both raw VizieR and cache columns)
        ra_col = next((c for c in simard_catalog.columns if c in ("RA_deg", "_RA", "RAJ2000")), None)
        dec_col = next((c for c in simard_catalog.columns if c in ("DE_deg", "_DE", "DEJ2000")), None)
        if ra_col is None or dec_col is None:
            print(f"    WARNING: Cannot find RA/Dec columns. Available: {list(simard_catalog.columns)}")
            print(f"    Skipping Simard enrichment.")
            return df

        # Build 1-degree bin grid for efficient matching
        grid = {}
        for i, row in simard_catalog.iterrows():
            try:
                ra = float(row[ra_col])
                dec = float(row[dec_col])
                key = (int(ra), int(dec))
                if key not in grid:
                    grid[key] = []
                grid[key].append((ra, dec, i))
            except (ValueError, KeyError):
                continue

        n_filled = 0
        MATCH_RADIUS = 10.0  # arcsec (was 3.0 — too strict for large surveys)

        for idx in targets.index:
            try:
                lens_ra = float(df.at[idx, "ra_deg"])
                lens_dec = float(df.at[idx, "dec_deg"])
            except (ValueError, TypeError):
                continue

            ra_bin, dec_bin = int(lens_ra), int(lens_dec)
            best_dist = MATCH_RADIUS
            best_simard_idx = None

            for dra in [-1, 0, 1]:
                for ddec in [-1, 0, 1]:
                    key = (ra_bin + dra, dec_bin + ddec)
                    if key not in grid:
                        continue
                    for sra, sdec, si in grid[key]:
                        d_ra = (sra - lens_ra) * np.cos(np.radians(lens_dec))
                        d_dec = sdec - lens_dec
                        dist_arcsec = np.sqrt(d_ra**2 + d_dec**2) * 3600.0
                        if dist_arcsec < best_dist:
                            best_dist = dist_arcsec
                            best_simard_idx = si

            if best_simard_idx is not None:
                srow = simard_catalog.iloc[best_simard_idx]
                # Fill from cache (supports both raw and derived columns)
                # Sérsic index
                for src, dst in [("n_sersic", "n_sersic"), ("ng", "n_sersic")]:
                    if src in simard_catalog.columns and pd.notna(srow.get(src)):
                        df.at[idx, dst] = float(srow[src])
                        break
                for src, dst in [("n_sersic_err", "n_sersic_err"), ("e_ng", "n_sersic_err")]:
                    if src in simard_catalog.columns and pd.notna(srow.get(src)):
                        df.at[idx, dst] = float(srow[src])
                        break
                # Axis ratio (q_disk, either pre-computed or from ellipticity)
                for src in ["q_disk", "b_a"]:
                    if src in simard_catalog.columns and pd.notna(srow.get(src)):
                        df.at[idx, "q_disk"] = float(srow[src])
                        break
                else:
                    # Derive from ellipticity e = 1 - b/a
                    for src in ["e", "ellipticity"]:
                        if src in simard_catalog.columns and pd.notna(srow.get(src)):
                            df.at[idx, "q_disk"] = 1.0 - float(srow[src])
                            break
                # Half-light radius (arcsec, either pre-computed or from kpc)
                for src in ["Re_arcsec"]:
                    if src in simard_catalog.columns and pd.notna(srow.get(src)):
                        df.at[idx, "Re_arcsec"] = float(srow[src])
                        break
                else:
                    for src in ["Re_circ_kpc", "Re_kpc", "Rhlr"]:
                        if src in simard_catalog.columns and pd.notna(srow.get(src)):
                            if "scale_kpc_per_arcsec" in simard_catalog.columns and pd.notna(srow.get("scale_kpc_per_arcsec")):
                                scale = float(srow["scale_kpc_per_arcsec"])
                                if scale > 0:
                                    df.at[idx, "Re_arcsec"] = float(srow[src]) / scale
                            break
                # Position angle
                for src, dst in [("PA_deg", "PA_disk_deg"), ("phi", "PA_disk_deg"), ("PA", "PA_disk_deg")]:
                    if src in simard_catalog.columns and pd.notna(srow.get(src)):
                        df.at[idx, dst] = float(srow[src])
                        break
                n_filled += 1

        print(f"    Sérsic parameters filled for {n_filled}/{len(targets)} systems")
        return df

    # ============================================================
    # Method 2: Per-object VizieR queries (fallback, slow)
    # ============================================================
    if HAVE_VIZIER and (simard_catalog is None or len(simard_catalog) == 0):
        print(f"    Trying per-object VizieR queries (slow, ~0.5s each)...")
        n_filled = 0
        for i, idx in enumerate(targets.index):
            if (i + 1) % 50 == 0:
                print(f"      Progress: {i + 1}/{len(targets)}")
            try:
                from astroquery.vizier import Vizier
                from astropy.coordinates import SkyCoord
                import astropy.units as u

                vizier = Vizier(columns=["**"], row_limit=5, timeout=30)
                coord = SkyCoord(ra=float(df.at[idx, "ra_deg"]) * u.deg,
                                dec=float(df.at[idx, "dec_deg"]) * u.deg)
                result = vizier.query_region(coord, radius=3 * u.arcsec,
                                            catalog="J/ApJS/196/11/table3")
                if result and len(result) > 0:
                    table = result[0]
                    if len(table) > 0:
                        row = table[0]
                        if "ng" in table.columns:
                            df.at[idx, "n_sersic"] = float(row["ng"]) if not np.ma.is_masked(row["ng"]) else np.nan
                        if "e_ng" in table.columns:
                            df.at[idx, "n_sersic_err"] = float(row["e_ng"]) if not np.ma.is_masked(row["e_ng"]) else np.nan
                        if "Rhlr" in table.columns:
                            df.at[idx, "Re_arcsec"] = float(row["Rhlr"]) if not np.ma.is_masked(row["Rhlr"]) else np.nan
                        if "e" in table.columns:
                            df.at[idx, "q_disk"] = 1.0 - float(row["e"]) if not np.ma.is_masked(row["e"]) else np.nan
                        if "phi" in table.columns:
                            df.at[idx, "PA_disk_deg"] = float(row["phi"]) if not np.ma.is_masked(row["phi"]) else np.nan
                        n_filled += 1
            except Exception:
                pass
            time.sleep(0.3)
        print(f"    Per-object queries filled {n_filled}/{len(targets)} systems")

    # ============================================================
    # Method 3: SDSS CAS PhotoObj fallback
    # ============================================================
    else:
        print(f"    Simard+2011 catalog not available.")
        print(f"    To enable: pip install astroquery (then re-run with --query-network)")
        print(f"    Falling back to SDSS PhotoObj fracDev for Tier 1+2...")

        import requests
        import io
        from .config import SDSS_SKYSERVER_URL

        n_filled = 0
        for idx in targets.index:
            try:
                ra = float(df.at[idx, "ra_deg"])
                dec = float(df.at[idx, "dec_deg"])
            except (ValueError, TypeError):
                continue

            sql = f"""
            SELECT TOP 1
                p.ra, p.dec,
                p.fracDev_r, p.modelMag_r, p.modelMag_g, p.modelMag_i,
                p.petroR50_r, p.petroR90_r,
                p.expRad_r, p.expAB_r, p.deVRad_r, p.deVAB_r
            FROM PhotoObj AS p
            WHERE p.ra BETWEEN {ra - 0.01} AND {ra + 0.01}
              AND p.dec BETWEEN {dec - 0.01} AND {dec + 0.01}
              AND p.type = 3
            ORDER BY p.modelMag_r ASC
            """
            try:
                resp = requests.get(SDSS_SKYSERVER_URL, params={
                    "cmd": sql, "format": "csv"
                }, timeout=30)
                if resp.status_code == 200 and len(resp.text) > 50:
                    photo = pd.read_csv(io.StringIO(resp.text))
                    if len(photo) > 0:
                        prow = photo.iloc[0]
                        # Fill magnitudes
                        for col_map in [("modelMag_r", "mag_r"), ("modelMag_g", "mag_g"),
                                        ("modelMag_i", "mag_i")]:
                            if col_map[0] in photo.columns and pd.notna(prow[col_map[0]]):
                                df.at[idx, col_map[1]] = float(prow[col_map[0]])
                        # Compute g-r
                        if pd.notna(df.at[idx, "mag_g"]) and pd.notna(df.at[idx, "mag_r"]):
                            df.at[idx, "g_r_color"] = df.at[idx, "mag_g"] - df.at[idx, "mag_r"]
                        # fracDev as morphological indicator
                        if "fracDev_r" in photo.columns and pd.notna(prow["fracDev_r"]):
                            df.at[idx, "n_bulge"] = float(prow["fracDev_r"])  # proxy
                        # Petrosian radius as size
                        if "petroR50_r" in photo.columns and pd.notna(prow["petroR50_r"]):
                            df.at[idx, "Re_arcsec"] = float(prow["petroR50_r"])
                        # expRad as disk scale
                        if "expRad_r" in photo.columns and pd.notna(prow["expRad_r"]):
                            if pd.isna(df.at[idx, "Re_disk_arcsec"]):
                                df.at[idx, "Re_disk_arcsec"] = float(prow["expRad_r"])
                        n_filled += 1
            except Exception:
                pass
            time.sleep(0.15)

        print(f"    SDSS PhotoObj filled {n_filled}/{len(targets)} systems")

    return df


def enrich_simard2011_photoobj(df: pd.DataFrame, query_tiers=(1, 2)) -> pd.DataFrame:
    """
    SDSS PhotoObj 快速丰富 — 仅查询 SDSS 测光参数（无需 VizieR）。

    填入: mag_r, mag_g, mag_i, g_r_color, Re_arcsec (petroR50), n_bulge (fracDev)
    对 Tier 1+2 逐个查询 SDSS SkyServer PhotoObj 表。
    """
    targets = df[df["disk_tier"].isin(query_tiers)]
    if len(targets) == 0:
        return df

    print(f"\n  ── SDSS PhotoObj enrichment ──", flush=True)
    print(f"  Targets: {len(targets)} systems (Tier {query_tiers})", flush=True)

    import requests
    import io
    from .config import SDSS_SKYSERVER_URL

    n_filled = 0
    for i, idx in enumerate(targets.index):
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{len(targets)}", flush=True)
        try:
            ra = float(df.at[idx, "ra_deg"])
            dec = float(df.at[idx, "dec_deg"])
        except (ValueError, TypeError):
            continue

        # Skip if mag_r and Re already filled
        if pd.notna(df.at[idx, "mag_r"]) and pd.notna(df.at[idx, "Re_arcsec"]):
            continue

        sql = f"""
        SELECT TOP 1
            p.ra, p.dec,
            p.fracDev_r, p.modelMag_r, p.modelMag_g, p.modelMag_i,
            p.petroR50_r, p.petroR90_r,
            p.expRad_r, p.expAB_r, p.deVRad_r, p.deVAB_r
        FROM PhotoObj AS p
        WHERE p.ra BETWEEN {ra - 0.01} AND {ra + 0.01}
          AND p.dec BETWEEN {dec - 0.01} AND {dec + 0.01}
          AND p.type = 3
        ORDER BY p.modelMag_r ASC
        """
        try:
            resp = requests.get(SDSS_SKYSERVER_URL, params={
                "cmd": sql, "format": "csv"
            }, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 50:
                photo = pd.read_csv(io.StringIO(resp.text))
                if len(photo) > 0:
                    prow = photo.iloc[0]
                    # Fill magnitudes if missing
                    for col_map in [("modelMag_r", "mag_r"), ("modelMag_g", "mag_g"),
                                    ("modelMag_i", "mag_i")]:
                        if col_map[0] in photo.columns and pd.notna(prow[col_map[0]]) and pd.isna(df.at[idx, col_map[1]]):
                            df.at[idx, col_map[1]] = float(prow[col_map[0]])
                    # Compute g-r if both available
                    if pd.notna(df.at[idx, "mag_g"]) and pd.notna(df.at[idx, "mag_r"]) and pd.isna(df.at[idx, "g_r_color"]):
                        df.at[idx, "g_r_color"] = df.at[idx, "mag_g"] - df.at[idx, "mag_r"]
                    # fracDev as morphological proxy
                    if "fracDev_r" in photo.columns and pd.notna(prow["fracDev_r"]) and pd.isna(df.at[idx, "n_bulge"]):
                        df.at[idx, "n_bulge"] = float(prow["fracDev_r"])
                    # Petrosian radius as half-light radius
                    if "petroR50_r" in photo.columns and pd.notna(prow["petroR50_r"]) and pd.isna(df.at[idx, "Re_arcsec"]):
                        df.at[idx, "Re_arcsec"] = float(prow["petroR50_r"])
                    # Exponential disk scale length
                    if "expRad_r" in photo.columns and pd.notna(prow["expRad_r"]) and pd.isna(df.at[idx, "Re_disk_arcsec"]):
                        df.at[idx, "Re_disk_arcsec"] = float(prow["expRad_r"])
                    n_filled += 1
        except Exception:
            pass
        time.sleep(0.15)

    print(f"    SDSS PhotoObj filled {n_filled}/{len(targets)} systems", flush=True)
    return df


def enrich_theta_e(df: pd.DataFrame) -> pd.DataFrame:
    """
    从文献/星表数据填入 Einstein 半径。

    策略:
      1. 从 lenscat 原始数据提取 theta_E (lenscat 包含大量已知透镜的 Einstein 半径)
      2. 从已知文献值填入 (SWELLS, SLACS, CASTLES)
      3. 从旧版 enriched v8 继承
    """
    print(f"\n  ── Einstein radius enrichment ──")

    # Load lenscat raw data for theta_E extraction
    from .config import PROJECT_ROOT, RAW_SOURCES

    lenscat_path = RAW_SOURCES["lenscat"]["files"]["catalog"]
    theta_e_filled = 0

    if os.path.exists(lenscat_path):
        print(f"    Extracting theta_E from lenscat catalog...")
        try:
            lc = pd.read_csv(lenscat_path, low_memory=False)

            # Look for theta_E column(s) — lenscat has various column names
            theta_cols = [c for c in lc.columns if "theta" in c.lower() or "einstein" in c.lower()]
            print(f"    Potential theta_E columns: {theta_cols}")

            if theta_cols:
                # Build coordinate lookup (nearest neighbor within 2 arcsec)
                from astropy.coordinates import SkyCoord
                import astropy.units as u

                lc_ra_col = [c for c in lc.columns if c.lower() in ("ra", "ra_deg", "raj2000")]
                lc_dec_col = [c for c in lc.columns if c.lower() in ("dec", "dec_deg", "dej2000")]

                if lc_ra_col and lc_dec_col:
                    lc_clean = lc.dropna(subset=lc_ra_col + lc_dec_col).copy()
                    lc_coords = SkyCoord(
                        ra=lc_clean[lc_ra_col[0]].values * u.deg,
                        dec=lc_clean[lc_dec_col[0]].values * u.deg
                    )

                    df_coords = SkyCoord(
                        ra=df["ra_deg"].values * u.deg,
                        dec=df["dec_deg"].values * u.deg
                    )

                    from astropy.coordinates import match_coordinates_sky
                    idx, sep2d, _ = match_coordinates_sky(df_coords, lc_coords)
                    matched = sep2d < 2 * u.arcsec

                    for theta_col in theta_cols:
                        for i in range(len(df)):
                            if matched[i] and (pd.isna(df.at[i, "theta_E_arcsec"]) or df.at[i, "theta_E_arcsec"] == 0):
                                val = lc_clean.iloc[idx[i]][theta_col]
                                try:
                                    val_f = float(val)
                                    if val_f > 0:
                                        df.at[i, "theta_E_arcsec"] = val_f
                                        theta_e_filled += 1
                                except (ValueError, TypeError):
                                    pass
        except Exception as e:
            print(f"    Warning: lenscat theta_E extraction failed: {e}")

    # Known theta_E values from literature for SWELLS lenses
    # SWELLS names in catalog with double-J: "SDSS JJ0820+4847"
    swell_s_thetas = {
        "J0820+4847": 1.17, "J0903+5116": 1.09,
        "J0904+5112": 0.89, "J0912+4845": 0.65,
        "J0922+4508": 0.85, "J0955+4640": 0.72,
        "J1029+0420": 1.16, "J1103+5322": 0.93,
        "J1110+4638": 0.93, "J1114+4727": 0.70,
        "J1142+5055": 0.97, "J1148+4950": 0.75,
        "J1205+4910": 1.25, "J1250+0523": 1.16,
        "J1301+5117": 0.83, "J1338+4447": 0.63,
        "J1430+4105": 1.47, "J1436+4654": 0.69,
        "J1632+4117": 0.82, "J2111-0114": 0.70,
        "J2141-0001": 1.10, "J2300+0022": 0.84,
        "J2341+0000": 1.29,
    }
    from .utils import normalize_name
    for short_name, theta_e in swell_s_thetas.items():
        norm_short = normalize_name(short_name)
        for i in range(len(df)):
            if pd.isna(df.at[i, "theta_E_arcsec"]) or df.at[i, "theta_E_arcsec"] == 0:
                name = str(df.at[i, "name"])
                if norm_short in normalize_name(name) or short_name in name:
                    df.at[i, "theta_E_arcsec"] = theta_e
                    theta_e_filled += 1

    print(f"    theta_E filled: {theta_e_filled}")
    return df


# ================================================================
# 主流程
# ================================================================

def enrich(input_path: str = MASTER_CLASSIFIED_PATH,
           query_network: bool = False,
           verbose: bool = True) -> pd.DataFrame:
    """
    Phase D 主入口。

    Args:
        input_path: 分类后的星表路径
        query_network: 是否尝试网络查询（当前环境通常为 False）
        verbose: 打印详细日志

    Returns:
        pd.DataFrame: 丰富后的星表
    """
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run classify.py first.")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    df = pd.read_csv(input_path)
    old_version = get_latest_version()
    version = old_version + 1
    old_path = current_enriched_path(old_version) if old_version > 0 else None
    print(f"Phase D — Parameter Enrichment (v{version})")
    print(f"{'='*60}")
    print(f"Input:  {len(df)} systems from {input_path}")
    print(f"Output: tier012_v{version}.csv")
    print(f"Previous: {'v' + str(old_version) if old_path else 'none'}")

    # ── 1. 从旧版本继承已有参数 ──
    df = enrich_from_existing(df, old_path)

    # ── 2. 可选：网络查询 ──
    if query_network:
        df = enrich_sdss(df)
        # Simard+2011: 首次需下载 ~1.1M 星系 (~50MB, 5-10min)，之后使用缓存
        # 若缓存已存在，直接使用；否则跳过并提示
        if os.path.exists(os.path.join(PROJECT_ROOT, "01_raw", "simard2011", "simard2011_cache.csv")):
            df = enrich_simard2011(df)
        else:
            print(f"\n  Simard+2011 cache not found. Run once with --download-simard to download.", flush=True)
        # PhotoObj 补充
        df = enrich_simard2011_photoobj(df)
    else:
        print(f"\n  Network queries skipped (query_network=False)")
        print(f"  To query: run with --query-network when SDSS/VizieR is accessible")

    # ── 3. theta_E ──
    df = enrich_theta_e(df)

    # ── 4. 重新分类（Phase C↔D 反馈） ──
    df = reclassify(df)
    df = sort_catalog(df)

    # ── 5. 保存 ──
    os.makedirs(DIR_ENRICHED, exist_ok=True)
    output_path = os.path.join(DIR_ENRICHED, f"tier012_v{version}.csv")
    df.to_csv(output_path, index=False)

    # ── 6. 更新 latest 符号链接 ──
    latest_link = os.path.join(DIR_ENRICHED, "tier012_latest.csv")
    try:
        os.remove(latest_link)
    except FileNotFoundError:
        pass
    try:
        os.symlink(output_path, latest_link)
    except OSError:
        # Windows/WSL may not support symlinks; copy instead
        import shutil
        shutil.copy2(output_path, latest_link)

    # ── 7. 统计 ──
    if verbose:
        print(f"\n{'='*50}")
        print(f"Enriched v{version} summary:")
        for t in [0, 1, 2, 3, "X"]:
            n = len(df[df["disk_tier"] == t])
            print(f"  Tier {t}: {n:>6d}")
        print(f"\n  Key parameters fill rate:")
        for col in ["sigma_km_s", "theta_E_arcsec", "logMstar_total",
                     "B_D_ratio", "n_sersic", "Re_arcsec"]:
            if col in df.columns:
                n = df[df[col].notna()].__len__()
                tier0_n = df[(df["disk_tier"] == 0) & df[col].notna()].__len__()
                print(f"    {col:20s}: {n:>5d}/{len(df)} total, {tier0_n:>3d} in Tier 0")
        print(f"\nSaved: {output_path}")

    return df


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase D: Enrich parameters")
    parser.add_argument("--query-network", action="store_true",
                       help="Attempt SDSS/VizieR queries (needs network)")
    parser.add_argument("--input", type=str, default=None,
                       help="Input CSV path (default: master_classified.csv)")
    args = parser.parse_args()

    enrich(input_path=args.input or MASTER_CLASSIFIED_PATH,
           query_network=args.query_network)
