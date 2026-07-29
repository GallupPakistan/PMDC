import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# REFERENCE MAPPINGS
# ============================================================================

DEGREE_CANONICAL_MAP = {
    "mbbs": "MBBS",
    "mbbsb": "MBBS",
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

def canonicalize_degree(raw: str):
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown Degree", False
    token = normalize_token(raw)
    if token in DEGREE_CANONICAL_MAP:
        canon = DEGREE_CANONICAL_MAP[token]
        return canon, (raw.strip() == canon)
    return "Unmapped Outliers", False

SOURCE_LABEL_MAP = {
    "doctors": "P-Series",
    "doctors_s_series": "S-Series",
    "doctors_d_series": "D-Series",
    "doctors_f_series": "F-Series",
    "doctors_n_series": "N-Series",
    "doctors_b_series": "B-Series",
    "doctors_ajk_series": "AJK-Series",
}

PASSING_YEAR_MIN, PASSING_YEAR_MAX = 1940, 2026

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(ttl=3600)
def load_raw_data(use_real_db: bool):
    if not use_real_db:
        return pd.DataFrame()
    try:
        from utils.data_loader import load_all_data
        df = load_all_data()
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"DB load failed: {e}")
        return pd.DataFrame()

# ============================================================================
# QUALITY PROFILING
# ============================================================================
@st.cache_data(ttl=3600)
def profile_doctor_level(use_real_db: bool) -> pd.DataFrame:
    # NOTE: this used to take the raw DataFrame as its argument, and
    # profile_qualification_level() below took THIS function's already-large
    # output DataFrame as ITS argument too - two expensive hashes of huge
    # nested-object DataFrames on every single rerun. Chaining bool-keyed
    # cached functions instead makes both cache checks nearly instant.
    df = load_raw_data(use_real_db)
    if df.empty:
        return df

    df = df.copy()
    df["source_label"] = df.get("source_table", pd.Series(dtype=str)).map(SOURCE_LABEL_MAP).fillna(df.get("source_table")).fillna("Unknown Series")

    df["RegistrationDate_parsed"] = pd.to_datetime(df.get("RegistrationDate"), format="%d/%m/%Y", errors="coerce")
    df["ValidUpto_parsed"] = pd.to_datetime(df.get("ValidUpto"), format="%d/%m/%Y", errors="coerce")

    df["Is_Duplicate"] = df.duplicated(subset=["source_table", "RegistrationNo"], keep=False)

    both_parsed = df["RegistrationDate_parsed"].notna() & df["ValidUpto_parsed"].notna()
    df["Date_Anomaly"] = both_parsed & (df["ValidUpto_parsed"] < df["RegistrationDate_parsed"])
    df["Date_Unparseable"] = df["RegistrationDate_parsed"].isna() | df["ValidUpto_parsed"].isna()

    df["Has_Qualification"] = df["Qualifications"].apply(lambda x: isinstance(x, list) and len(x) > 0) if "Qualifications" in df.columns else False

    core_fields = [c for c in ["Name", "FatherName", "RegistrationDate", "ValidUpto", "Status", "RegistrationType"] if c in df.columns]
    df["Missing_Core_Fields"] = df[core_fields].isnull().sum(axis=1) if core_fields else 0
    df["Missing_Core_Fields"] = df["Missing_Core_Fields"] + (~df["Has_Qualification"]).astype(int)

    df["RegYear"] = df["RegistrationDate_parsed"].dt.year

    return df

@st.cache_data(ttl=3600)
def profile_qualification_level(use_real_db: bool) -> pd.DataFrame:
    df = profile_doctor_level(use_real_db)
    if df.empty or "Qualifications" not in df.columns:
        return pd.DataFrame()

    keep_cols = [c for c in ["RegistrationNo", "source_table", "source_label", "Qualifications"] if c in df.columns]
    base = df[keep_cols].copy()
    base["Qualifications"] = base["Qualifications"].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else [{"Degree": np.nan, "PassingYear": np.nan}])
    exploded = base.explode("Qualifications", ignore_index=True)
    
    qual_df = pd.json_normalize(exploded["Qualifications"])
    qual_df.index = exploded.index
    result = pd.concat([exploded.drop(columns=["Qualifications"]), qual_df], axis=1)

    result["PassingYear_raw"] = result.get("PassingYear").astype(str).str.strip().replace("nan", np.nan)
    result["PassingYear_num"] = pd.to_numeric(result["PassingYear_raw"], errors="coerce")
    result["PassingYear_Missing"] = result["PassingYear_num"].isna()
    result["PassingYear_Outlier"] = result["PassingYear_num"].notna() & (
        ~result["PassingYear_num"].between(PASSING_YEAR_MIN, PASSING_YEAR_MAX)
    )

    result["Degree"] = result.get("Degree").astype(str).str.strip().replace("nan", np.nan)
    canon_results = result["Degree"].apply(canonicalize_degree)
    
    result["Degree_Canonical"] = [t[0] for t in canon_results]
    result["Degree_Standardized"] = [t[1] for t in canon_results]
    result["Degree_Unmapped"] = result["Degree_Canonical"] == "Unmapped Outliers"

    return result

# ============================================================================
# CSS & RENDERING HELPERS
# ============================================================================
def inject_custom_css():
    st.markdown("""
        <style>
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid rgba(184, 134, 11, 0.35); border-radius: 12px;
            padding: 20px; text-align: center; margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .kpi-title { color: #4B5563; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
        .kpi-value { color: #9A6B00; font-size: 2.1rem; font-weight: 700; margin: 0; }
        
        .stSelectbox label, .stRadio label, .stSidebar h3, div[data-testid="stWidgetLabel"] p {
            color: #374151 !important; font-weight: 600 !important;
        }
        
        .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            height: 50px; white-space: pre-wrap; background-color: transparent;
            color: #6B7280; font-weight: 600; font-size: 0.95rem;
        }
        .stTabs [aria-selected="true"] { color: #9A6B00 !important; border-bottom: 2px solid #B8860B !important; }
        .caveat { color: #6B7280; font-size: 0.78rem; font-style: italic; margin-top: -4px; margin-bottom: 12px; }
        </style>
    """, unsafe_allow_html=True)

def safe_render_chart(fig):
    layout_theme = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#374151", size=11),
        margin=dict(t=50, b=40, l=40, r=40), height=320,
        title_font=dict(size=13, color="#9A6B00", family="Cinzel"),
        legend=dict(font=dict(color="#374151", size=10), bgcolor="rgba(255,255,255,0.6)")
    )
    fig.update_layout(**layout_theme)
    if hasattr(fig, "update_xaxes"):
        fig.update_xaxes(showgrid=False, zeroline=False, title_font=dict(color="#9A6B00"), tickfont=dict(color="#374151"))
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)", zeroline=False, title_font=dict(color="#9A6B00"), tickfont=dict(color="#374151"))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# MAIN INTERFACE CONTROL
# ============================================================================
def render_data_quality():
    inject_custom_css()

    st.markdown("<h1 style='color: #9A6B00;'>Data Quality & Governance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4B5563; margin-bottom: 2rem;'>Audit registry health, missing values, logic anomalies, and nomenclature standardization.</p>", unsafe_allow_html=True)

    use_real_db = st.session_state.get("db_connected", False)
    raw_df = load_raw_data(use_real_db=use_real_db)

    if raw_df.empty:
        st.warning("⚠️ No database connection detected. Please check your DB connection to view live registry data.")
        return

    df = profile_doctor_level(use_real_db)
    qual_df = profile_qualification_level(use_real_db)

    total_records = len(df)
    duplicate_count = df["Is_Duplicate"].sum()
    date_anomaly_count = df["Date_Anomaly"].sum()
    date_unparseable_count = df["Date_Unparseable"].sum()
    perfect_records = (df["Missing_Core_Fields"] == 0).sum()

    missing_penalty = (df["Missing_Core_Fields"].sum() / (total_records * max(1, len(df.columns)))) * 100 if total_records else 0
    anomaly_penalty = (date_anomaly_count / total_records) * 100 if total_records else 0
    duplicate_penalty = (duplicate_count / total_records) * 100 if total_records else 0
    health_score = max(0, 100 - (missing_penalty * 1.5) - (anomaly_penalty * 2) - (duplicate_penalty * 1.5))

    # --- SIDEBAR FILTERS ---
    st.sidebar.markdown("<h3 style='color:#9A6B00;'>⚙️ Audit Filters</h3>", unsafe_allow_html=True)
    source_opts = sorted(df["source_label"].dropna().unique())
    selected_sources = st.sidebar.selectbox("Series / Collection", ["All"] + source_opts)
    health_view = st.sidebar.radio("View Mode", ["All Records", "Clean Data Only", "Anomalies Only"])

    if selected_sources != "All":
        df = df[df["source_label"] == selected_sources]
        if not qual_df.empty:
            qual_df = qual_df[qual_df["source_label"] == selected_sources]

    if health_view == "Clean Data Only":
        df = df[(df["Missing_Core_Fields"] == 0) & (~df["Date_Anomaly"]) & (~df["Is_Duplicate"])]
    elif health_view == "Anomalies Only":
        df = df[(df["Missing_Core_Fields"] > 0) | (df["Date_Anomaly"]) | (df["Is_Duplicate"])]

    # Registration Year slicer
    if "RegYear" in df.columns:
        valid_years = df["RegYear"].dropna()
        if not valid_years.empty:
            min_year = int(valid_years.min())
            max_year = min(2018, int(valid_years.max()))
            if min_year < max_year:
                year_range = st.sidebar.slider(
                    "Registration Year",
                    min_year,
                    max_year,
                    (min_year, max_year)
                )
                df = df[df["RegYear"].between(year_range[0], year_range[1]) | df["RegYear"].isna()]
            else:
                st.sidebar.caption(f"Registration Year: {min_year}")

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        return

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Records</div><div class="kpi-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with col2:
        score_color = "#1E7A6D" if health_score > 85 else "#9A6B00"
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Database Health Score</div><div class="kpi-value" style="color: {score_color};">{health_score:.1f}/100</div></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Duplicate Entries</div><div class="kpi-value" style="color: #DC2626;">{duplicate_count:,}</div></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Perfect Records</div><div class="kpi-value" style="color: #1E7A6D;">{perfect_records:,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    palette_sequence = ["#B8860B", "#2A9D8F", "#EF4444", "#E8C547", "#577590", "#F9C74F", "#90BE6D"]
    tab1, tab2, tab3 = st.tabs(["🧩 Completeness & Missingness", "🚨 Anomalies & Duplicates", "🧹 Standardization Diagnostics"])

    # ========================================================================
    # TAB 1: COMPLETENESS & MISSINGNESS
    # ========================================================================
    with tab1:
        st.markdown("<h3 style='color:#9A6B00;'>Data Nullity Profiling</h3>", unsafe_allow_html=True)
        
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig1 = go.Figure(go.Indicator(mode="gauge+number", value=health_score, title={"text": "Data Health Index", "font": {"color": "#374151"}},
                gauge={"axis": {"range": [None, 100], "tickcolor": "#374151"}, "bar": {"color": "#2A9D8F"}, "bgcolor": "rgba(0,0,0,0.03)",
                       "steps": [{"range": [0, 60], "color": "rgba(239, 68, 68, 0.15)"}, {"range": [60, 100], "color": "rgba(42, 157, 143, 0.15)"}]}))
            safe_render_chart(fig1)
        with r1c2:
            core_fields = [c for c in ["Name", "FatherName", "RegistrationDate", "ValidUpto", "Status", "RegistrationType"] if c in df.columns]
            missing_df = df[core_fields].isnull().sum().reset_index()
            missing_df.columns = ["Field", "Missing Count"]
            missing_df = pd.concat([missing_df, pd.DataFrame({"Field": ["Qualifications (Empty)"], "Missing Count": [int((~df["Has_Qualification"]).sum())]})], ignore_index=True)
            safe_render_chart(px.bar(missing_df, x="Missing Count", y="Field", orientation="h", title="2. Missing Data Count by Attribute Field", color_discrete_sequence=["#EF4444"]))

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            comp_counts = pd.DataFrame({"Status": ["Perfect Rows", "Minor (1 Missing)", "Critical (2+ Missing)"],
                "Count": [(df["Missing_Core_Fields"] == 0).sum(), (df["Missing_Core_Fields"] == 1).sum(), (df["Missing_Core_Fields"] >= 2).sum()]})
            safe_render_chart(px.pie(comp_counts, names="Status", values="Count", hole=0.5, title="3. Record Completeness Level Allocation", color_discrete_sequence=["#2A9D8F", "#E8C547", "#EF4444"]))
        with r2c2:
            if len(core_fields) > 1:
                safe_render_chart(px.imshow(df[core_fields].isnull().corr().fillna(0), title="4. Missing Data Attribute Correlation Topology", color_continuous_scale="Oranges"))

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            series_comp = df.groupby("source_label")["Missing_Core_Fields"].mean().reset_index()
            safe_render_chart(px.bar(series_comp, x="source_label", y="Missing_Core_Fields", title="5. Mean Absolute Missing Weights by Collection Batch", color_discrete_sequence=["#B8860B"]))
        with r3c2:
            safe_render_chart(px.histogram(df, x="Missing_Core_Fields", title="6. Row-Level Null Fields Dispersion Volume Density", color_discrete_sequence=["#577590"]))

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            safe_render_chart(px.box(df, x="source_label", y="Missing_Core_Fields", title="7. Outlier Deficient Deviations across Registries", color_discrete_sequence=["#E8C547"]))
        with r4c2:
            qual_nulls = df.groupby("source_label")["Has_Qualification"].apply(lambda x: (~x).sum()).reset_index(name="Missing Quals")
            safe_render_chart(px.pie(qual_nulls, names="source_label", values="Missing Quals", title="8. Empty Qualifications Subsets Ratio Spread", color_discrete_sequence=palette_sequence))

        r5c1, r5c2 = st.columns(2)
        with r5c1:
            sun_df = df[["source_label", "Missing_Core_Fields"]].copy()
            sun_df["Nullity_Status"] = sun_df["Missing_Core_Fields"].apply(lambda x: "Pristine" if x==0 else "Deficient")
            sun_df["source_label"] = sun_df["source_label"].fillna("Unknown Series")
            safe_render_chart(px.sunburst(sun_df, path=["source_label", "Nullity_Status"], title="9. Hierarchical Registry Integrity Distribution Hierarchy"))
        with r5c2:
            tree_df = df.groupby(["source_label", "Status"]).size().reset_index(name="Count")
            tree_df["source_label"] = tree_df["source_label"].fillna("Unknown Series")
            tree_df["Status"] = tree_df["Status"].fillna("Unknown Status")
            safe_render_chart(px.treemap(tree_df, path=["source_label", "Status"], values="Count", title="10. Volume Area Layout Matrix of Registry Status Batches", color_discrete_sequence=palette_sequence))

    # ========================================================================
    # TAB 2: ANOMALIES & DUPLICATES
    # ========================================================================
    with tab2:
        st.markdown("<h3 style='color:#9A6B00;'>Logic Conflicts & Duplicates</h3>", unsafe_allow_html=True)
        
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            outlier_years = qual_df["PassingYear_Outlier"].sum() if not qual_df.empty else 0
            anom_counts = pd.DataFrame({"Anomaly Type": ["Duplicate Reg. Nos", "ValidUpto Inversions", "Unparseable Dates", "Outlier Grad Years"],
                                        "Volume": [duplicate_count, date_anomaly_count, date_unparseable_count, outlier_years]})
            safe_render_chart(px.bar(anom_counts, x="Anomaly Type", y="Volume", title="1. Total Volumetric Logic Conflicts Detected", color_discrete_sequence=["#EF4444"]))
        with r1c2:
            safe_df = df.dropna(subset=["RegistrationDate_parsed", "ValidUpto_parsed"])
            if not safe_df.empty:
                safe_render_chart(px.scatter(safe_df, x="RegistrationDate_parsed", y="ValidUpto_parsed", color="Date_Anomaly", title="2. Chronological Registry Timeline Matrix", color_discrete_map={True: "#EF4444", False: "#2A9D8F"}))

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            if not qual_df.empty and qual_df["PassingYear_num"].notna().sum() > 0:
                safe_render_chart(px.box(qual_df.dropna(subset=["PassingYear_num"]), x="PassingYear_num", title="3. Graduate Year Boundary Dispersion", color_discrete_sequence=["#B8860B"]))
        with r2c2:
            dup_df = df[df["Is_Duplicate"]]
            if not dup_df.empty:
                safe_render_chart(px.pie(dup_df["source_label"].value_counts().reset_index(name="Count"), names="source_label", values="Count", title="4. Profile Redundancies Share by Series Batch", color_discrete_sequence=palette_sequence))
            else:
                st.info("4. No active duplication profile metrics under current filters.")

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            date_anom_src = df.groupby("source_label")["Date_Anomaly"].sum().reset_index(name="Inversions")
            safe_render_chart(px.bar(date_anom_src, x="source_label", y="Inversions", title="5. Chronological Sequence Logic Failures by Batch", color_discrete_sequence=["#E8C547"]))
        with r3c2:
            unparse_src = df.groupby("source_label")["Date_Unparseable"].sum().reset_index(name="Corrupted")
            safe_render_chart(px.pie(unparse_src, names="source_label", values="Corrupted", title="6. Corrupted/Unparseable Date Formats Weight Share", color_discrete_sequence=palette_sequence))

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            if not qual_df.empty:
                outlier_src = qual_df.groupby("source_label")["PassingYear_Outlier"].sum().reset_index(name="Outliers")
                safe_render_chart(px.line(outlier_src, x="source_label", y="Outliers", title="7. Timeline Boundary Anomaly Volumes per Batch Stream", markers=True, color_discrete_sequence=["#EF4444"]))
        with r4c2:
            df["Combined_Risk_Score"] = df["Is_Duplicate"].astype(int) + df["Date_Anomaly"].astype(int) + df["Date_Unparseable"].astype(int)
            safe_render_chart(px.histogram(df, x="Combined_Risk_Score", title="8. Compound Record Integrity Failure Risk Distribution", color_discrete_sequence=["#577590"]))

        r5c1, r5c2 = st.columns(2)
        with r5c1:
            sun_dup = df[["RegistrationType", "Is_Duplicate"]].copy()
            sun_dup["RegistrationType"] = sun_dup["RegistrationType"].fillna("Unknown Type")
            sun_dup["Is_Duplicate"] = sun_dup["Is_Duplicate"].map({True: "Duplicate", False: "Unique"})
            safe_render_chart(px.sunburst(sun_dup, path=["RegistrationType", "Is_Duplicate"], title="9. Registration Type Duplication Risk Correlation Hierarchy"))
        with r5c2:
            clean_vs_flagged = pd.DataFrame({"Classification": ["Fully Compliant", "Flagged Logic Flaws"], "Count": [(df["Combined_Risk_Score"] == 0).sum(), (df["Combined_Risk_Score"] > 0).sum()]})
            safe_render_chart(px.bar(clean_vs_flagged, x="Classification", y="Count", title="10. Global Target Health Inversion Overview", color_discrete_sequence=["#2A9D8F"]))

    # ========================================================================
    # TAB 3: STANDARDIZATION DIAGNOSTICS
    # ========================================================================
    with tab3:
        st.markdown("<h3 style='color:#9A6B00;'>Degree Nomenclature Standardization</h3>", unsafe_allow_html=True)
        
        if qual_df.empty:
            st.info("No active qualification records available for standardization tracking.")
        else:
            valid_degrees = qual_df[qual_df["Degree"].notna() & (qual_df["Degree"].astype(str).str.strip() != "")]
            
            if valid_degrees.empty:
                st.info("No valid raw degree strings text available to standardize.")
            else:
                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    std_counts = pd.DataFrame({
                        "Category": ["Canonical Exact Form", "Recognized Fuzzy Form", "Unmapped Outliers"],
                        "Count": [
                            int(valid_degrees["Degree_Standardized"].sum()),
                            int((valid_degrees["Degree_Canonical"].notna() & ~valid_degrees["Degree_Standardized"] & (valid_degrees["Degree_Canonical"] != "Unmapped Outliers")).sum()),
                            int((valid_degrees["Degree_Canonical"] == "Unmapped Outliers").sum())
                        ]
                    })
                    safe_render_chart(px.pie(std_counts, names="Category", values="Count", hole=0.5, title="1. Degree Ingestion Cleanliness Status Mix", color_discrete_sequence=["#2A9D8F", "#E8C547", "#EF4444"]))
                with r1c2:
                    unmapped = valid_degrees[valid_degrees["Degree_Canonical"] == "Unmapped Outliers"]
                    if not unmapped.empty:
                        unmapped_counts = unmapped["Degree"].value_counts().reset_index().head(10)
                        unmapped_counts.columns = ["Raw Input String", "Frequency"]
                        safe_render_chart(px.bar(unmapped_counts, x="Frequency", y="Raw Input String", orientation="h", title="2. Top 10 High-Frequency Unmapped Degree Outliers", color_discrete_sequence=["#B8860B"]))
                    else:
                        st.info("2. No completely unmapped string outliers found in current data.")

                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    batch_std = valid_degrees.groupby(["source_label", "Degree_Standardized"]).size().reset_index(name="Count")
                    batch_std["source_label"] = batch_std["source_label"].fillna("Unknown Series")
                    batch_std["Degree_Standardized"] = batch_std["Degree_Standardized"].map({True: "Standardized", False: "Non-Standardized"})
                    safe_render_chart(px.bar(batch_std, x="source_label", y="Count", color="Degree_Standardized", title="3. Ingestion Quality Profiles Matrix Across Batches", barmode="stack", color_discrete_sequence=["#2A9D8F", "#EF4444"]))
                with r2c2:
                    unmapped_batch = valid_degrees.groupby("source_label")["Degree_Unmapped"].sum().reset_index(name="Total Unmapped")
                    unmapped_batch["source_label"] = unmapped_batch["source_label"].fillna("Unknown Series")
                    safe_render_chart(px.pie(unmapped_batch, names="source_label", values="Total Unmapped", title="4. Lexical Dictionary Defect Volume Share by Source", color_discrete_sequence=palette_sequence))

                r3c1, r3c2 = st.columns(2)
                with r3c1:
                    safe_render_chart(px.scatter(batch_std, x="source_label", y="Count", color="Degree_Standardized", title="5. Standardization Matrix Density Grouping Plot", color_discrete_sequence=["#E8C547", "#577590"]))
                with r3c2:
                    safe_render_chart(px.line(unmapped_batch, x="source_label", y="Total Unmapped", title="6. Structural Lexical Cleansing Error Vectors per Batch", markers=True, color_discrete_sequence=["#EF4444"]))

                r4c1, r4c2 = st.columns(2)
                with r4c1:
                    funnel_data = pd.DataFrame({
                        "Stage": ["Raw Rows Ingested", "Deduped Repositories", "Core Elements Verified", "Fully Clean Profiles"],
                        "Count": [total_records, total_records - duplicate_count, total_records - duplicate_count - (df["Missing_Core_Fields"] > 0).sum(), perfect_records]
                    })
                    safe_render_chart(px.funnel(funnel_data, x="Count", y="Stage", title="7. Data Refinement Pipeline Transformation Funnel", color_discrete_sequence=["#2A9D8F"]))
                with r4c2:
                    trend_df = df.dropna(subset=["RegistrationDate_parsed"]).copy()
                    trend_df["Reg_Year"] = trend_df["RegistrationDate_parsed"].dt.year
                    trend_group = trend_df.groupby("Reg_Year").agg(Total=("RegistrationNo", "count"), Clean=("Missing_Core_Fields", lambda x: (x == 0).sum())).reset_index()
                    trend_group["Quality_Rate"] = (trend_group["Clean"] / trend_group["Total"]) * 100
                    trend_group = trend_group[(trend_group["Reg_Year"] >= 2010) & (trend_group["Reg_Year"] <= 2026)]
                    if not trend_group.empty:
                        safe_render_chart(px.line(trend_group, x="Reg_Year", y="Quality_Rate", title="8. Ingestion Integrity Performance Over Time", markers=True, color_discrete_sequence=["#E8C547"]))
                    else:
                        st.info("8. Trend timeline metrics empty for active dates filter.")

                r5c1, r5c2 = st.columns(2)
                with r5c1:
                    sun_canon = valid_degrees[["source_label", "Degree_Canonical"]].copy()
                    sun_canon["source_label"] = sun_canon["source_label"].fillna("Unknown Series")
                    sun_canon["Degree_Canonical"] = sun_canon["Degree_Canonical"].fillna("Unrecognized Label")
                    safe_render_chart(px.sunburst(sun_canon, path=["source_label", "Degree_Canonical"], title="9. Hierarchy Matrix of Mapped Degree Classes"))
                with r5c2:
                    canon_counts = valid_degrees.groupby("Degree_Canonical").size().reset_index(name="Volume Size")
                    canon_counts["Degree_Canonical"] = canon_counts["Degree_Canonical"].fillna("Unrecognized Label")
                    safe_render_chart(px.treemap(canon_counts, path=["Degree_Canonical"], values="Volume Size", title="10. Relative Volume Density Map of Cleaned Registry Labels", color_discrete_sequence=palette_sequence))

        st.markdown('<p class="caveat">Canonicalization map covers MBBS/BDS/MD/MS/FCPS/MCPS/DPT/Ph.D variants — extend DEGREE_CANONICAL_MAP as new variants surface.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="PMDC Governance Engine")
    render_data_quality()
