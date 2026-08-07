"""
Shared, cached "enriched" dataset builder.

WHY THIS FILE EXISTS
---------------------
The Gender & Specialization Deep Dive page and the Yearly Deep Dive page each
used to define their OWN separate load_and_enrich() function - identical
logic, copy-pasted. Streamlit's @st.cache_data caches by function identity,
so those were two entirely separate caches: visiting one page and waiting for
it to finish did nothing for the other page's cache. This module centralizes
that work into ONE cached function so whichever page you open first "warms"
the cache for both.

PERFORMANCE NOTE (2026-07-29)
------------------------------
The old flatten_qualifications() ran a FULL pass over every row, 7 separate
times (once per possible qualification slot), using pandas .apply() +
pd.json_normalize() each time - regardless of whether a given doctor actually
had 1, 2, or 7 qualifications on file. Since most doctors only have 1-2
qualification entries, this meant ~5-6 of those 7 passes were pure wasted
work for the vast majority of rows.

The version below does ONE single pass over the rows (not 7), and for each
row only processes however many qualification entries that doctor actually
has - so the total work is proportional to the real amount of data, not to
max_quals x row_count. It also avoids pd.json_normalize() (which carries
overhead for handling arbitrarily nested structures we don't need here) in
favor of direct dict.get() calls on plain Python lists, which is
substantially faster than pandas .apply() for this kind of row-by-row work.

Timing is printed (visible in the Space's Logs tab, same pattern as
utils/data_loader.py's per-collection fetch timing) so any future slowness
can be diagnosed precisely instead of guessed at.
"""
import logging
import time

import pandas as pd
import streamlit as st

from utils.gender_province import (
    infer_gender_series,
    infer_province_series,
    is_specialist_from_series,
)

logger = logging.getLogger(__name__)

SPEC_COLS = [f"Qualification_{i}_Speciality" for i in range(1, 8)]
# Generic "this is just your base degree" markers - NOT real specializations.
# BUG FIX: only "BASIC MEDICAL QUALIFICATION" (MBBS's generic marker) was
# excluded here; "BASIC DENTAL QUALIFICATION" (the exact same kind of
# generic marker, just for BDS) was missing. That meant ~26,238 dentists
# whose only qualification is a base BDS degree were wrongly counted as
# "Is_Specialist" and their BDS showed up as a bar on the "Specialization
# Degree Types" chart - BDS is a base degree, not a specialization.
_NON_SPECIALTY_VALUES = {"", ".", "BASIC MEDICAL QUALIFICATION", "BASIC DENTAL QUALIFICATION", "NAN"}

BASE_DEGREE_MAP = {
    "mbbs": "MBBS", "bds": "BDS",
}


def _normalize_token(s: str) -> str:
    import re
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def canonicalize_base_degree(raw) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown Base Degree"
    token = _normalize_token(raw)
    return BASE_DEGREE_MAP.get(token, raw.strip().title())


def real_specialty_mask(col: pd.Series) -> pd.Series:
    cleaned = col.astype(str).str.strip().str.upper()
    return col.notna() & ~cleaned.isin(_NON_SPECIALTY_VALUES)


def flatten_qualifications(df: pd.DataFrame, max_quals: int = 7) -> pd.DataFrame:
    """Fast, single-pass version. Builds all Qualification_i_* columns in one
    sweep over the rows, doing only as much work as each row's own
    Qualifications list actually contains (not always 7 slots' worth)."""
    if "Qualifications" not in df.columns:
        return df

    n = len(df)
    field_names = ("University", "Speciality", "Degree", "PassingYear")
    # cols[i][field] -> plain Python list of length n, filled in as we go
    cols = {
        i: {field: [None] * n for field in field_names}
        for i in range(max_quals)
    }

    quals_list = df["Qualifications"].tolist()
    for row_idx, quals in enumerate(quals_list):
        if not isinstance(quals, list) or not quals:
            continue
        limit = min(len(quals), max_quals)
        for i in range(limit):
            entry = quals[i]
            if isinstance(entry, dict):
                slot = cols[i]
                slot["University"][row_idx] = entry.get("University")
                slot["Speciality"][row_idx] = entry.get("Speciality")
                slot["Degree"][row_idx] = entry.get("Degree")
                slot["PassingYear"][row_idx] = entry.get("PassingYear")

    for i in range(max_quals):
        qi = i + 1
        df[f"Qualification_{qi}_University"] = cols[i]["University"]
        df[f"Qualification_{qi}_Speciality"] = cols[i]["Speciality"]
        df[f"Qualification_{qi}_Degree"] = cols[i]["Degree"]
        df[f"Qualification_{qi}_PassingYear"] = pd.to_numeric(pd.Series(cols[i]["PassingYear"]), errors="coerce")

    return df


@st.cache_data(ttl=3600)
def load_and_enrich_all() -> pd.DataFrame:
    """The single, shared, expensive computation. Cached once here - every
    page that needs Gender/Province/Specialist/Field data should call THIS
    function instead of defining its own copy, so the work only ever
    happens once per cache ttl, no matter which page triggers it first."""
    from utils.data_loader import load_all_data

    t_start = time.time()
    df = load_all_data()
    t_loaded = time.time()
    print(f"[enriched_data] load_all_data: {t_loaded - t_start:.2f}s  rows={0 if df is None else len(df)}")
    if df is None or df.empty:
        return pd.DataFrame()

    df = flatten_qualifications(df)
    t_flat = time.time()
    print(f"[enriched_data] flatten_qualifications: {t_flat - t_loaded:.2f}s")

    df["Gender"] = infer_gender_series(df["Name"], df.get("Qualification_1_University"))
    df["Province"] = infer_province_series(
        df.get("Qualification_1_University", pd.Series(dtype=str, index=df.index)),
        df.get("source_table"),
    )
    df["RegYear"] = pd.to_datetime(df["RegistrationDate"], errors="coerce", dayfirst=True).dt.year
    t_gender_prov = time.time()
    print(f"[enriched_data] gender/province/regyear: {t_gender_prov - t_flat:.2f}s")

    if "source_table" in df.columns:
        df["Is_Specialist_Series"] = is_specialist_from_series(df["source_table"])
    else:
        df["Is_Specialist_Series"] = False

    is_specialist_text = pd.Series(False, index=df.index)
    for c in SPEC_COLS:
        if c in df.columns:
            is_specialist_text |= real_specialty_mask(df[c])
    df["Is_Specialist"] = is_specialist_text
    t_specialist_flag = time.time()
    print(f"[enriched_data] specialist flags: {t_specialist_flag - t_gender_prov:.2f}s")

    primary_specialty = pd.Series(pd.NA, index=df.index, dtype=object)
    specialty_year = pd.Series(pd.NA, index=df.index, dtype=object)
    specialty_degree_raw = pd.Series(pd.NA, index=df.index, dtype=object)
    for i in range(1, 8):
        sc, yc, dc = f"Qualification_{i}_Speciality", f"Qualification_{i}_PassingYear", f"Qualification_{i}_Degree"
        if sc not in df.columns:
            continue
        still_empty = primary_specialty.isna()
        if not still_empty.any():
            break
        fillable = still_empty & real_specialty_mask(df[sc])
        primary_specialty.loc[fillable] = df.loc[fillable, sc].astype(str).str.strip().str.title()
        if yc in df.columns:
            specialty_year.loc[fillable] = df.loc[fillable, yc]
        if dc in df.columns:
            specialty_degree_raw.loc[fillable] = df.loc[fillable, dc]

    df["Primary_Specialty"] = primary_specialty
    df["Specialty_Year"] = pd.to_numeric(specialty_year, errors="coerce")
    df["Specialist_Degree"] = specialty_degree_raw.apply(canonicalize_base_degree)
    df.loc[~df["Is_Specialist"], "Specialist_Degree"] = pd.NA
    t_primary_specialty = time.time()
    print(f"[enriched_data] primary specialty scan: {t_primary_specialty - t_specialist_flag:.2f}s")

    # Base degree (MBBS/BDS/etc.) from the doctor's FIRST qualification -
    # used by the Yearly Deep Dive page's "include MBBS/BDS as a field" toggle.
    if "Qualification_1_Degree" in df.columns:
        df["Base_Degree"] = df["Qualification_1_Degree"].apply(canonicalize_base_degree)
    else:
        df["Base_Degree"] = "Unknown Base Degree"
    if "Qualification_1_PassingYear" in df.columns:
        df["Base_Degree_Year"] = pd.to_numeric(df["Qualification_1_PassingYear"], errors="coerce")
    else:
        df["Base_Degree_Year"] = pd.NA

    df["Field_All"] = df["Primary_Specialty"]
    df.loc[~df["Is_Specialist"], "Field_All"] = df.loc[~df["Is_Specialist"], "Base_Degree"]
    df["Field_All_Year"] = df["Specialty_Year"]
    df.loc[~df["Is_Specialist"], "Field_All_Year"] = df.loc[~df["Is_Specialist"], "Base_Degree_Year"]

    t_end = time.time()
    print(f"[enriched_data] base degree/field_all: {t_end - t_primary_specialty:.2f}s")
    print(f"[enriched_data] TOTAL load_and_enrich_all: {t_end - t_start:.2f}s  rows={len(df)}")

    return df