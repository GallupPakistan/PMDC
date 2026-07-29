import streamlit as st
import pandas as pd
import plotly.express as px
import re
import logging

from utils.gender_province import (
    infer_gender,
    infer_province,
    SOURCE_LABEL_MAP,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# REFERENCE MAPPINGS (gender/province now come from utils.gender_province)
# ============================================================================

def normalize_status(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"
    s = raw.strip()
    if re.search(r"in[\s\-]?active", s, re.I):
        return "In-Active"
    if re.search(r"^active$", s, re.I):
        return "Active"
    return "Other"


def normalize_reg_type(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"
    v = raw.strip().lower()
    if v == "permanent":
        return "Permanent"
    if v == "provisional":
        return "Provisional"
    return "Other"


def get_first_qual_field(quals, key):
    if isinstance(quals, list) and len(quals) > 0 and isinstance(quals[0], dict):
        return quals[0].get(key)
    return None


DEGREE_CANONICAL_MAP = {
    "mbbs": "MBBS",
    "bds": "BDS",
    "md": "MD",
    "ms": "MS",
    "fcps": "FCPS",
    "mcps": "MCPS",
    "dpt": "DPT",
    "phd": "Ph.D",
}


def normalize_token(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def canonicalize_degree(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"
    token = normalize_token(raw)
    return DEGREE_CANONICAL_MAP.get(token, raw.strip())


# ============================================================================
# DATA LOADING & TRANSFORM
# ============================================================================
@st.cache_data(ttl=3600)
def load_search_data(use_real_db: bool):
    if not use_real_db:
        return pd.DataFrame()
    try:
        from utils.data_loader import load_all_data
        df = load_all_data()
        if df is None or df.empty:
            logger.warning("DB returned empty dataframe.")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"DB load failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def transform_search_data(use_real_db: bool) -> pd.DataFrame:
    # NOTE: this used to take the raw DataFrame as its argument. Hashing a
    # large nested-object DataFrame on every rerun is itself slow enough that
    # it can feel like the cache isn't helping at all. A plain bool hashes
    # almost instantly.
    df = load_search_data(use_real_db)
    if df.empty:
        return df

    df = df.copy()
    df["RegistrationDate_parsed"] = pd.to_datetime(df.get("RegistrationDate"), format="%d/%m/%Y", errors="coerce")
    df["ValidUpto_parsed"] = pd.to_datetime(df.get("ValidUpto"), format="%d/%m/%Y", errors="coerce")
    df["RegYear"] = df["RegistrationDate_parsed"].dt.year

    df["Status_Norm"] = df.get("Status", pd.Series(dtype=str)).apply(normalize_status)
    df["RegType_Norm"] = df.get("RegistrationType", pd.Series(dtype=str)).apply(normalize_reg_type)

    df["Primary_Degree"] = df.get("Qualifications", pd.Series(dtype=object)).apply(lambda q: get_first_qual_field(q, "Degree")).apply(canonicalize_degree)
    df["Primary_University"] = df.get("Qualifications", pd.Series(dtype=object)).apply(lambda q: get_first_qual_field(q, "University"))
    df["Primary_Speciality"] = df.get("Qualifications", pd.Series(dtype=object)).apply(lambda q: get_first_qual_field(q, "Speciality"))

    # Gender: first-name-only matching (shared, fixed heuristic).
    df["Gender"] = df.get("Name", pd.Series(dtype=str)).apply(infer_gender)
    # Province: university-name guess, except AJK-series which is ground truth.
    df["Province"] = df["Primary_University"].apply(infer_province)
    if "source_table" in df.columns:
        df.loc[df["source_table"].astype(str).str.strip() == "doctors_ajk_series", "Province"] = "AJK"
    df["source_label"] = df.get("source_table", pd.Series(dtype=str)).map(SOURCE_LABEL_MAP).fillna(df.get("source_table"))

    return df


@st.cache_data(ttl=3600)
def build_qualifications_detail(df: pd.DataFrame) -> pd.DataFrame:
    """Exploded qualification rows for the expandable detail view."""
    if df.empty or "Qualifications" not in df.columns:
        return pd.DataFrame()

    base = df[["RegistrationNo", "Qualifications"]].copy()
    base["Qualifications"] = base["Qualifications"].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else [{}])
    exploded = base.explode("Qualifications", ignore_index=True)
    qual_df = pd.json_normalize(exploded["Qualifications"])
    qual_df.index = exploded.index
    result = pd.concat([exploded.drop(columns=["Qualifications"]), qual_df], axis=1)
    result["QualSeq"] = result.groupby("RegistrationNo").cumcount() + 1
    return result


# ============================================================================
# CSS
# ============================================================================
def inject_custom_css():
    st.markdown("""
        <style>
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid rgba(184, 134, 11, 0.3); border-radius: 12px;
            padding: 20px; text-align: center; transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .kpi-card:hover {
            transform: translateY(-5px); box-shadow: 0 8px 24px rgba(184, 134, 11, 0.15);
            border: 1px solid rgba(184, 134, 11, 0.6);
        }
        .kpi-title {
            color: #6B7280; font-size: 0.85rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;
        }
        .kpi-value { color: #9A6B00; font-size: 2rem; font-weight: 700; margin: 0; }
        .search-box {
            background: rgba(0, 0, 0, 0.02);
            border: 1px solid rgba(184, 134, 11, 0.4);
            border-radius: 8px; padding: 2px; margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            height: 50px; white-space: pre-wrap; background-color: transparent;
            border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
            color: #6B7280; font-weight: 600;
        }
        .stTabs [aria-selected="true"] { color: #9A6B00 !important; border-bottom: 2px solid #B8860B !important; }
        .caveat { color: #6B7280; font-size: 0.78rem; font-style: italic; margin-top: -8px; margin-bottom: 8px; }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN PAGE LOGIC
# ============================================================================
def render_doctor_search():
    inject_custom_css()

    st.markdown("<h1 style='color: #9A6B00;'>Practitioner Search & Cohort Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4B5563; margin-bottom: 1.5rem;'>Search the national registry and generate analytics on the filtered cohort.</p>", unsafe_allow_html=True)

    use_real_db = st.session_state.get("db_connected", False)
    raw_df = load_search_data(use_real_db=use_real_db)

    if raw_df.empty:
        st.warning("⚠️ No database connection detected. Please check your DB connection to view live registry data.")
        return

    df = transform_search_data(use_real_db)

    # --- SEARCH ---
    st.markdown("<div class='search-box'>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_query = st.text_input("🔍 Search by Doctor Name, Father's Name, or Registration No.",
                                      placeholder="e.g. Naseer, Chaudhry, 99999-P...")
    with col_s2:
        search_type = st.selectbox("Search Field Priority", ["All Fields", "Name Only", "Registration No. Only"])
    st.markdown("</div>", unsafe_allow_html=True)

    # --- SIDEBAR FILTERS ---
    st.sidebar.markdown("<h3 style='color:#9A6B00;'>⚙️ Refine Cohort</h3>", unsafe_allow_html=True)
    prov_filter = st.sidebar.multiselect("Province", options=sorted([p for p in df["Province"].dropna().unique() if p != "Unknown"]))
    status_filter = st.sidebar.multiselect("Status", options=sorted(df["Status_Norm"].dropna().unique()))
    deg_filter = st.sidebar.multiselect("Primary Degree", options=sorted(df["Primary_Degree"].dropna().unique()))
    source_opts = sorted(df["source_label"].dropna().unique())
    selected_sources = st.sidebar.selectbox("Series / Collection", ["All"] + source_opts)

    # --- APPLY SEARCH & FILTERS ---
    filtered_df = df.copy()

    # Registration Year slicer
    if "RegYear" in df.columns:
        valid_years = df["RegYear"].dropna()
        if not valid_years.empty:
            year_range = st.sidebar.slider(
                "Registration Year",
                int(valid_years.min()),
                2018,
                (int(valid_years.min()), 2018)
            )
            df = df[df["RegYear"].between(year_range[0], year_range[1]) | df["RegYear"].isna()]

    if search_query:
        query = search_query.lower()
        name_match = filtered_df.get("Name", pd.Series(dtype=str)).fillna("").str.lower().str.contains(query, regex=False)
        father_match = filtered_df.get("FatherName", pd.Series(dtype=str)).fillna("").str.lower().str.contains(query, regex=False)
        regno_match = filtered_df.get("RegistrationNo", pd.Series(dtype=str)).fillna("").str.lower().str.contains(query, regex=False)

        if search_type == "All Fields":
            filtered_df = filtered_df[name_match | father_match | regno_match]
        elif search_type == "Name Only":
            filtered_df = filtered_df[name_match]
        elif search_type == "Registration No. Only":
            filtered_df = filtered_df[regno_match]

    if prov_filter:
        filtered_df = filtered_df[filtered_df["Province"].isin(prov_filter)]
    if status_filter:
        filtered_df = filtered_df[filtered_df["Status_Norm"].isin(status_filter)]
    if deg_filter:
        filtered_df = filtered_df[filtered_df["Primary_Degree"].isin(deg_filter)]
    if selected_sources != "All":
        filtered_df = filtered_df[filtered_df["source_label"] == selected_sources]

    # --- KPIs ---
    total_found = len(filtered_df)
    active_found = (filtered_df["Status_Norm"] == "Active").sum()
    top_prov = filtered_df["Province"].mode()[0] if not filtered_df.empty and filtered_df["Province"].notna().any() else "N/A"
    spec_pool = filtered_df[filtered_df["Primary_Speciality"].notna() & (filtered_df["Primary_Speciality"] != "")]
    top_spec = spec_pool["Primary_Speciality"].mode()[0] if not spec_pool.empty else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Records Found</div><div class="kpi-value">{total_found:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Active Status</div><div class="kpi-value" style="color:#1E7A6D;">{active_found:,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Dominant Region</div><div class="kpi-value" style="font-size:1.5rem; line-height:2.8rem;">{top_prov}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Top Speciality</div><div class="kpi-value" style="font-size:1.3rem; line-height:3rem; color:#9A6B00;">{top_spec}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("⚠️ No records match your current search and filter criteria. Please adjust your query.")
        return

    layout_theme = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#374151"), 
        title_font=dict(size=20, color="#9A6B00", family="Cinzel"),
        margin=dict(t=50, b=30, l=30, r=30),
    )
    color_seq = ["#B8860B", "#2A9D8F", "#E8C547", "#EF4444", "#374151", "#64748B"]

    tab1, tab2, tab3 = st.tabs(["📋 Data Grid", "📊 Cohort Demographics", "🔬 Advanced Cross-Analytics"])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.markdown("<h3 style='color:#9A6B00;'>Search Results</h3>", unsafe_allow_html=True)
        display_cols = [c for c in [
            "RegistrationNo", "Name", "FatherName", "Primary_Degree", "Primary_University",
            "Primary_Speciality", "Status_Norm", "RegType_Norm", "Province", "ValidUpto", "source_label"
        ] if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True, height=400)

        st.markdown("<h4 style='color:#9A6B00; margin-top:1.5rem;'>Full Qualification History</h4>", unsafe_allow_html=True)
        if total_found > 200:
            st.info("Refine your search to 200 or fewer results to view per-doctor qualification detail.")
        else:
            qual_detail = build_qualifications_detail(filtered_df)
            for reg_no in filtered_df["RegistrationNo"].head(50):
                doc_row = filtered_df[filtered_df["RegistrationNo"] == reg_no].iloc[0]
                with st.expander(f"{doc_row.get('Name', 'Unknown')} — {reg_no}"):
                    doc_quals = qual_detail[qual_detail["RegistrationNo"] == reg_no]
                    qual_cols = [c for c in ["QualSeq", "Degree", "Speciality", "University", "PassingYear"] if c in doc_quals.columns]
                    st.dataframe(doc_quals[qual_cols], use_container_width=True, hide_index=True)
            if total_found > 50:
                st.caption(f"Showing detail for first 50 of {total_found} results.")

    # ---------------- TAB 2 ----------------
    with tab2:
        st.markdown(f"<h3 style='color:#9A6B00;'>Demographics for '{search_query or 'All'}' Cohort</h3>", unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            fig1 = px.pie(filtered_df, names="Status_Norm", hole=0.6, title="Cohort Legal Status",
                         color_discrete_sequence=["#2A9D8F", "#EF4444", "#94A3B8"])
            fig1.update_traces(textinfo="percent+label")
            fig1.update_layout(**layout_theme, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with r1c2:
            gender_pool = filtered_df[filtered_df["Gender"] != "Unknown"]
            fig2 = px.pie(gender_pool, names="Gender", title="Gender Distribution (Inferred)",
                         color_discrete_sequence=["#B8860B", "#2A9D8F", "#94A3B8"])
            fig2.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#FFFFFF", width=2)))
            fig2.update_layout(**layout_theme)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<p class="caveat">Gender is inferred from the doctor\'s first name only (heuristic — names outside our reference list show as Unknown rather than a risky guess). Province is ground-truth for AJK, and university-based for everyone else — treat both as approximate outside AJK.</p>', unsafe_allow_html=True)

        r2c1, r2c2 = st.columns([1.5, 1])
        with r2c1:
            prov_df = filtered_df[filtered_df["Province"] != "Unknown"]
            if prov_df.empty:
                st.info("No province-mappable records in current cohort.")
            else:
                fig3 = px.treemap(prov_df, path=["Province", "RegType_Norm"], title="Regional Registration Types",
                                  color_discrete_sequence=color_seq)
                fig3.update_layout(**layout_theme)
                st.plotly_chart(fig3, use_container_width=True)

        with r2c2:
            deg_counts = filtered_df["Primary_Degree"].value_counts().reset_index().head(10)
            deg_counts.columns = ["Primary_Degree", "count"]
            fig4 = px.bar(deg_counts, x="count", y="Primary_Degree", orientation="h", title="Degree Dominance",
                         color_discrete_sequence=["#B8860B"])
            fig4.update_layout(**layout_theme, yaxis_title="", xaxis_title="")
            st.plotly_chart(fig4, use_container_width=True)

        timeline = filtered_df.dropna(subset=["RegYear"]).groupby("RegYear").size().reset_index(name="Count")
        timeline = timeline[timeline["RegYear"].between(1950, 2026)]
        fig5 = px.area(timeline, x="RegYear", y="Count", title="Cohort Registration Timeline",
                       color_discrete_sequence=["#2A9D8F"])
        fig5.update_layout(**layout_theme)
        st.plotly_chart(fig5, use_container_width=True)

    # ---------------- TAB 3 ----------------
    with tab3:
        st.markdown("<h3 style='color:#9A6B00;'>Deep-Dive Logic Matrix</h3>", unsafe_allow_html=True)
        r3c1, r3c2 = st.columns(2)

        with r3c1:
            spec_pool2 = filtered_df[
                filtered_df["Primary_Speciality"].notna() & 
                (filtered_df["Primary_Speciality"] != "") &
                (filtered_df["Primary_Speciality"] != ".")
            ]
            spec_counts = spec_pool2["Primary_Speciality"].value_counts().reset_index().head(10)
            spec_counts.columns = ["Speciality", "count"]
            if spec_counts.empty:
                st.info("No speciality data in current cohort.")
            else:
                fig6 = px.funnel(spec_counts, x="count", y="Speciality", title="Specialization Pipeline (Top 10)",
                                 color_discrete_sequence=["#2A9D8F", "#B8860B", "#0D3B2A"])
                fig6.update_layout(**layout_theme)
                st.plotly_chart(fig6, use_container_width=True)

        with r3c2:
            prov_df2 = filtered_df[filtered_df["Province"] != "Unknown"]
            if prov_df2.empty:
                st.info("No province-mappable records in current cohort.")
            else:
                prov_stat = prov_df2.groupby(["Province", "Status_Norm"]).size().reset_index(name="Volume")
                fig7 = px.bar(prov_stat, x="Province", y="Volume", color="Status_Norm", title="Compliance by Region",
                             color_discrete_sequence=["#2A9D8F", "#EF4444", "#94A3B8"])
                fig7.update_layout(**layout_theme)
                st.plotly_chart(fig7, use_container_width=True)

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            sun_df = filtered_df[
                filtered_df["RegType_Norm"].notna() & (filtered_df["RegType_Norm"] != "") &
                filtered_df["Primary_Degree"].notna() & (filtered_df["Primary_Degree"] != "Unknown")
            ]
            if sun_df.empty:
                st.info("Insufficient RegistrationType/Degree data for classification chart in current cohort.")
            else:
                fig8 = px.sunburst(sun_df, path=["RegType_Norm", "Primary_Degree"], title="Practitioner Classification",
                                   color_discrete_sequence=["#E8C547", "#2A9D8F"])
                fig8.update_layout(**layout_theme)
                st.plotly_chart(fig8, use_container_width=True)

        with r4c2:
            hist_df = filtered_df.dropna(subset=["RegYear"])
            hist_df = hist_df[hist_df["RegYear"].between(1950, 2026)]
            hist_df = hist_df[hist_df["Gender"] != "Unknown"]
            fig10 = px.histogram(hist_df, x="RegYear", color="Gender", nbins=20, title="Experience by Gender (Inferred)",
                                barmode="group", color_discrete_sequence=["#B8860B", "#2A9D8F", "#94A3B8"])
            fig10.update_layout(**layout_theme)
            st.plotly_chart(fig10, use_container_width=True)


if __name__ == "__main__":
    render_doctor_search()
