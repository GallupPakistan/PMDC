import time
import pandas as pd
import streamlit as st

from utils.mongodb import get_mongo_collection

COLLECTIONS = ["doctors", "doctors_b_series", "doctors_s_series", "doctors_d_series", "doctors_f_series", 
               "doctors_n_series", "doctors_ajk_series"]  # your actual collection names


@st.cache_data(ttl=3600)
def load_collection(collection_name):
    collection = get_mongo_collection(collection_name)

    t0 = time.time()
    cursor = collection.find({}, {"_id": 0}, batch_size=10000)  # exclude Mongo _id field, use batch_size for large collections
    docs = list(cursor)  # materialize cursor to list for DataFrame conversion
    t1 = time.time()

    df = pd.DataFrame(docs)
    df["source_table"] = collection_name
    t2 = time.time()

    print(f"[{collection_name}] fetch={t1-t0:.2f}s  df_build={t2-t1:.2f}s  rows={len(docs)}")

    return df


@st.cache_data(ttl=3600)
def load_all_data():
    """Fetch and concatenate all collections into one DataFrame."""
    dfs = [load_collection(name) for name in COLLECTIONS]
    return pd.concat(dfs, ignore_index=True)