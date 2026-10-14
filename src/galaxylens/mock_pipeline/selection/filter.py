"""L1 selection filter (API catalog -> field-center disk-lens candidates).

Criteria come from a :class:`~galaxylens.mock_pipeline.selection.config.SelectionCriteria`
dataclass (never hard-coded), so the same filter serves any target galaxy.
Field-center means ``primary_flag == 1`` and FoF group ``logM_grp < 13``.

This mirrors the project's ``filter_l1_census.py`` and the L1 stage of
``l3_verify_cluster.py``, with the thresholds lifted into the criteria.
"""


def is_field(row):
    """Field center: primary (group-center) flag and non-cluster group."""
    return (row.get("primary_flag") == 1
            and (row.get("group_mass_log") is None
                 or row["group_mass_log"] < 13))


def pass_l1(row, criteria):
    """True if ``row`` satisfies the L1 criteria."""
    if criteria.field_only and not is_field(row):
        return False
    mstar = row.get("mstar_msun")
    if mstar is None or not (criteria.mstar_range[0] <= mstar <= criteria.mstar_range[1]):
        return False
    sigma = row.get("sigma_km_s")
    if sigma is None or sigma < criteria.sigma_min:
        return False
    return True


def filter_l1(rows, criteria):
    """Return the list of rows passing L1 (each with a ``snap`` key added)."""
    out = []
    for r in rows:
        if pass_l1(r, criteria):
            out.append(dict(r))
    return out
