"""
Data loading layer.

HISTORY / WHY THIS FILE CHANGED (2026-08-05)
-----------------------------------------------
This used to pull data live from MongoDB Atlas (see utils/mongodb.py, still
kept around for reference but no longer imported here). The dashboard is now
backed by a static CSV export of the same data instead, produced with
`mongoexport` + a small flattening script. Every OTHER file in this project
(utils/enriched_data.py, all utils/*.py, every page in pages/) calls
load_all_data() or load_collection() exactly like before and does not need
to change - this file is the only seam that had to be rewritten, because it
was the only place that knew HOW the data got fetched.

WHAT THE CSV LOOKS LIKE vs WHAT THIS FILE RETURNS
---------------------------------------------------
The CSV (data/doctors_combined_full_all_qualifications.csv) is in "long"
format: one row PER QUALIFICATION, not one row per doctor. A doctor with 3
postgraduate qualifications on file appears as 3 rows, all sharing the same
RegistrationNo/Name/etc. but with different Speciality/Degree/University/
PassingYear.

Every downstream file in this project (utils/enriched_data.py's
flatten_qualifications(), and any page that does
`doc.get("Qualifications")`) was written against the OLD Mongo shape: one
row PER DOCTOR, with a "Qualifications" column holding a Python list of
dicts (exactly what pymongo used to hand back for a Mongo array field).

So load_collection() below re-groups the long CSV rows back into that
one-row-per-doctor / Qualifications-list-of-dicts shape before handing the
DataFrame off - nothing downstream had to know the storage format changed.
"""
import time
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "doctors_combined_full_all_qualifications.csv"

COLLECTIONS = ["doctors", "doctors_b_series", "doctors_s_series", "doctors_d_series", "doctors_f_series",
               "doctors_n_series", "doctors_ajk_series"]  # your actual collection/series names

BASE_FIELDS = ["numeric_id", "Name", "FatherName", "RegistrationType",
               "RegistrationDate", "ValidUpto", "Status"]
QUAL_FIELDS = ("Speciality", "Degree", "University", "PassingYear")


@st.cache_data(ttl=3600)
def _load_raw_csv() -> pd.DataFrame:
    """Read the exported CSV from disk exactly once per cache TTL, no matter
    how many collections/pages ask for data afterward."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data export not found at {DATA_FILE}. Make sure "
            f"data/doctors_combined_full_all_qualifications.csv is committed to the repo."
        )
    return pd.read_csv(DATA_FILE, low_memory=False)


@st.cache_data(ttl=3600)
def load_collection(collection_name):
    """Rebuild one 'collection' (source_collection slice of the CSV) back
    into one-row-per-doctor shape, with a Qualifications column holding a
    list of dicts - matching what the old Mongo-backed version returned."""
    t0 = time.time()
    raw = _load_raw_csv()
    subset = raw[raw["source_collection"] == collection_name]

    quals_by_reg = {}
    base_by_reg = {}
    order = []
    for row in subset.itertuples(index=False):
        reg = row.RegistrationNo
        if reg not in quals_by_reg:
            quals_by_reg[reg] = []
            base_by_reg[reg] = {f: getattr(row, f) for f in BASE_FIELDS}
            order.append(reg)
        speciality, degree, university, passing_year = row.Speciality, row.Degree, row.University, row.PassingYear
        # Skip an all-empty qualification slot (doctor with no qualification
        # on file still gets one CSV row, with these fields blank).
        if pd.isna(speciality) and pd.isna(degree):
            continue
        quals_by_reg[reg].append({
            "Speciality": speciality,
            "Degree": degree,
            "University": university,
            "PassingYear": passing_year,
        })

    records = []
    for reg in order:
        record = dict(base_by_reg[reg])
        record["RegistrationNo"] = reg
        record["Qualifications"] = quals_by_reg[reg]
        records.append(record)

    df = pd.DataFrame(records)
    df["source_table"] = collection_name
    t1 = time.time()

    print(f"[{collection_name}] rebuild={t1 - t0:.2f}s  rows={len(df)}")

    return df


@st.cache_data(ttl=3600)
def load_all_data():
    """Fetch and concatenate all collections into one DataFrame."""
    dfs = [load_collection(name) for name in COLLECTIONS]
    return pd.concat(dfs, ignore_index=True)
