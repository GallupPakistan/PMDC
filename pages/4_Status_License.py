import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import logging

from utils.gender_province import (
    infer_gender_series,
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


def get_first_university(quals):
    if isinstance(quals, list) and len(quals) > 0 and isinstance(quals[0], dict):
        return quals[0].get("University")
    return None


def exclude_unknown(data: pd.DataFrame, *cols) -> pd.DataFrame:
    """Drop rows where any of the given columns is 'Unknown', so charts never
    show an 'Unknown' bar/slice/line for that field."""
    mask = pd.Series(True, index=data.index)
    for c in cols:
        if c in data.columns:
            mask &= (data[c].astype(str).str.strip() != "Unknown")
    return data[mask]


# ============================================================================
# DATA LOADING & TRANSFORM
# ============================================================================
@st.cache_data(ttl=3600)
def load_license_data(use_real_db: bool):
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


@st.cache_data(ttl=3600)
def transform_license_data(use_real_db: bool) -> pd.DataFrame:
    # NOTE: this used to take the raw DataFrame as its argument. Hashing a
    # 200k+ row DataFrame with nested list/dict columns (Qualifications) on
    # every rerun is itself slow enough to feel like "nothing is cached" even
    # though the result was. Taking a plain bool instead is nearly instant to
    # hash, so the cache actually pays off.
    df = load_license_data(use_real_db)
    if df.empty:
        return df

    df = df.copy()

    df["RegistrationDate"] = pd.to_datetime(df.get("RegistrationDate"), format="%d/%m/%Y", errors="coerce")
    df["ValidUpto"] = pd.to_datetime(df.get("ValidUpto"), format="%d/%m/%Y", errors="coerce")

    df["ExpiryYear"] = df["ValidUpto"].dt.year
    df["RegYear"] = df["RegistrationDate"].dt.year

    df["Status_Norm"] = df.get("Status", pd.Series(dtype=str)).apply(normalize_status)
    df["RegType_Norm"] = df.get("RegistrationType", pd.Series(dtype=str)).apply(normalize_reg_type)

    if "Qualifications" in df.columns:
        df["Qualification_1_University"] = df["Qualifications"].apply(get_first_university)
        df["Province"] = df["Qualification_1_University"].apply(infer_province)
        # AJK-series records get Province="AJK" with full confidence — ground
        # truth from the collection they came from, not a university guess.
        if "source_table" in df.columns:
            df.loc[df["source_table"].astype(str).str.strip() == "doctors_ajk_series", "Province"] = "AJK"
    else:
        df["Province"] = "Unknown"

    df["source_label"] = df.get("source_table", pd.Series(dtype=str)).map(SOURCE_LABEL_MAP).fillna(df.get("source_table"))

    # Gender: use the real field if present; where it's genuinely missing, infer
    # from the doctor's Name using the shared first-name-only matcher. Real
    # values already present are never touched.
    if "Gender" not in df.columns:
        df["Gender"] = "Unknown"
    df["Gender"] = df["Gender"].fillna("Unknown")
    if "Name" in df.columns:
        inferred_gender = infer_gender_series(df["Name"], df.get("Qualification_1_University"))
        unknown_mask = df["Gender"].astype(str).str.strip().isin(["", "Unknown", "nan"])
        df.loc[unknown_mask, "Gender"] = inferred_gender[unknown_mask]

    return df


# ============================================================================
# CSS
# ============================================================================
def inject_custom_css():
    st.markdown("""
        <style>
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid rgba(184, 134, 11, 0.35); border-radius: 12px;
            padding: 20px; text-align: center; transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .kpi-card:hover {
            transform: translateY(-5px); box-shadow: 0 8px 24px rgba(184, 134, 11, 0.2);
            border: 1px solid rgba(184, 134, 11, 0.7);
        }
        .kpi-title {
            color: #4B5563; font-size: 0.85rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;
        }
        .kpi-value { color: #9A6B00; font-size: 2rem; font-weight: 700; margin: 0; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            height: 50px; white-space: pre-wrap; background-color: transparent;
            border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
            color: #374151; font-weight: 600;
        }
        .stTabs [aria-selected="true"] { color: #9A6B00 !important; border-bottom: 2px solid #B8860B !important; }
        
        .stMultiSelect, .stSelectbox, label {
            color: #374151 !important;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)


def safe_plotly_render(fig_object):
    layout_theme = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#374151", size=11),
        margin=dict(t=45, b=35, l=35, r=35), height=320,
        title_font=dict(size=13, color="#9A6B00", family="Arial Black"),
        legend=dict(font=dict(color="#374151", size=11), bgcolor="rgba(255,255,255,0.6)")
    )
    fig_object.update_layout(**layout_theme)
    if hasattr(fig_object, "update_xaxes"):
        fig_object.update_xaxes(showgrid=False, zeroline=False, title_font=dict(color="#9A6B00"), tickfont=dict(color="#374151"))
        fig_object.update_yaxes(gridcolor="rgba(0,0,0,0.08)", zeroline=False, title_font=dict(color="#9A6B00"), tickfont=dict(color="#374151"))
    st.plotly_chart(fig_object, use_container_width=True, config={'displayModeBar': False})


# ============================================================================
# MAIN PAGE LOGIC
# ============================================================================
def render_status_license():
    inject_custom_css()

    st.markdown("<h1 style='color: #9A6B00;'>License Status & Registration Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4B5563; margin-bottom: 2rem;'>Active/In-Active status, registration type, and temporal trends across PMDC series.</p>", unsafe_allow_html=True)

    use_real_db = st.session_state.get("db_connected", False)
    raw_df = load_license_data(use_real_db=use_real_db)

    if raw_df.empty:
        st.warning("⚠️ No database connection detected. Please check your DB connection to view live registry data.")
        return

    df = transform_license_data(use_real_db)

    # --- SIDEBAR FILTERS ---
    st.sidebar.markdown("<h3 style='color:#9A6B00;'>⚙️ Compliance Filters</h3>", unsafe_allow_html=True)

    status_filter = st.sidebar.multiselect("Status", options=sorted(df["Status_Norm"].dropna().unique()))
    regtype_filter = st.sidebar.multiselect("Registration Type", options=sorted(df["RegType_Norm"].dropna().unique()))
    prov_filter = st.sidebar.multiselect("Province", options=sorted([p for p in df["Province"].dropna().unique() if p != "Unknown"]))

    source_opts = sorted(df["source_label"].dropna().unique())
    selected_sources = st.sidebar.selectbox("Series / Collection", ["All"] + source_opts)

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

    if status_filter:
        df = df[df["Status_Norm"].isin(status_filter)]
    if regtype_filter:
        df = df[df["RegType_Norm"].isin(regtype_filter)]
    if prov_filter:
        df = df[df["Province"].isin(prov_filter)]
    if selected_sources != "All":
        df = df[df["source_label"] == selected_sources]

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        return

    # --- KPI CALCULATIONS ---
    total_records = len(df)
    active_count = (df["Status_Norm"] == "Active").sum()
    inactive_count = (df["Status_Norm"] == "In-Active").sum()
    other_status_count = (~df["Status_Norm"].isin(["Active", "In-Active"])).sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Filtered</div><div class="kpi-value">{total_records:,}</div></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Active</div><div class="kpi-value" style="color: #1E7A6D;">{active_count:,}</div></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">In-Active</div><div class="kpi-value" style="color: #DC2626;">{inactive_count:,}</div></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Other Status</div><div class="kpi-value" style="color: #9A6B00;">{other_status_count:,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    color_map_status = {"Active": "#2A9D8F", "In-Active": "#EF4444", "Other": "#E8C547", "Unknown": "#94A3B8"}
    palette_list = ["#B8860B", "#2A9D8F", "#EF4444", "#E8C547", "#64748B"]

    tab1, tab2, tab3 = st.tabs(["📊 Macro Status View", "⏳ Temporal Registration Trends", "📍 Regional Status (Inferred)"])

    # ========================================================================
    # TAB 1: MACRO STATUS VIEW
    # ========================================================================
    with tab1:
        st.markdown("<h3 style='color:#9A6B00;'>Global Status Breakdown</h3>", unsafe_allow_html=True)
        
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            stat_counts = df["Status_Norm"].value_counts().reset_index()
            stat_counts.columns = ["Status", "Count"]
            safe_plotly_render(px.pie(stat_counts, names="Status", values="Count", hole=0.6, title="1. Overall Audited License Status", color="Status", color_discrete_map=color_map_status))
        with r1c2:
            regtype_counts = df["RegType_Norm"].value_counts().reset_index()
            regtype_counts.columns = ["Registration Type", "Count"]
            safe_plotly_render(px.pie(regtype_counts, names="Registration Type", values="Count", hole=0.6, title="2. Permanent vs. Provisional Macro Mix", color_discrete_sequence=palette_list))

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            type_stat = df.groupby(["RegType_Norm", "Status_Norm"]).size().reset_index(name="Volume")
            safe_plotly_render(px.bar(type_stat, x="RegType_Norm", y="Volume", color="Status_Norm", barmode="group", title="3. Compliance Status by Registration Type", color_discrete_map=color_map_status))
        with r2c2:
            src_stat = df.groupby(["source_label", "Status_Norm"]).size().reset_index(name="Volume")
            safe_plotly_render(px.bar(src_stat, x="source_label", y="Volume", color="Status_Norm", barmode="stack", title="4. Registration Integrity Across Collection Batches", color_discrete_map=color_map_status))

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            gen_stat = exclude_unknown(df, "Gender").groupby(["Gender", "Status_Norm"]).size().reset_index(name="Count")
            safe_plotly_render(px.bar(gen_stat, x="Count", y="Gender", color="Status_Norm", orientation="h", barmode="group", title="5. Gender-Stratified Status Inflow Breakdown", color_discrete_map=color_map_status))
        with r3c2:
            gen_type = exclude_unknown(df, "Gender").groupby(["Gender", "RegType_Norm"]).size().reset_index(name="Count")
            safe_plotly_render(px.bar(gen_type, x="RegType_Norm", y="Count", color="Gender", barmode="stack", title="6. Registration Modality Spread by Gender", color_discrete_sequence=palette_list))

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            vol_tree = df["source_label"].value_counts().reset_index()
            vol_tree.columns = ["Series", "Count"]
            safe_plotly_render(px.treemap(vol_tree, path=["Series"], values="Count", title="7. Series Data Volume Capacity Matrix", color_discrete_sequence=palette_list))
        with r4c2:
            fig_sun1 = px.sunburst(df, path=["Status_Norm", "RegType_Norm"], title="8. Multi-Layer Registration Lifecycle Mapping")
            fig_sun1.update_traces(textinfo="label+percent entry")
            safe_plotly_render(fig_sun1)

        r5c1, r5c2 = st.columns(2)
        with r5c1:
            scat_df = df.groupby(["source_label", "RegType_Norm"]).size().reset_index(name="Density")
            safe_plotly_render(px.scatter(scat_df, x="source_label", y="RegType_Norm", size="Density", title="9. Class-to-Batch Volume Cross Density Matrix", color_discrete_sequence=["#2A9D8F"]))
        with r5c2:
            other_types = df.groupby(["Status_Norm"]).size().reset_index(name="Count")
            safe_plotly_render(px.bar(other_types, x="Status_Norm", y="Count", title="10. Total Category Audit Status Profile", color_discrete_sequence=["#EF4444"]))

    # ========================================================================
    # TAB 2: TEMPORAL REGISTRATION TRENDS
    # ========================================================================
    with tab2:
        st.markdown("<h3 style='color:#9A6B00;'>Registration Volume Over Time</h3>", unsafe_allow_html=True)
        
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            reg_trend = df.dropna(subset=["RegYear"]).groupby("RegYear").size().reset_index(name="Registrations")
            reg_trend = reg_trend[reg_trend["RegYear"].between(1950, 2026)]
            safe_plotly_render(px.area(reg_trend, x="RegYear", y="Registrations", title="1. National Aggregate Inflow Vector Over Time", color_discrete_sequence=["#B8860B"]))
        with r1c2:
            reg_by_status = df.dropna(subset=["RegYear"]).groupby(["RegYear", "Status_Norm"]).size().reset_index(name="Volume")
            reg_by_status = reg_by_status[reg_by_status["RegYear"].between(1950, 2026)]
            safe_plotly_render(px.line(reg_by_status, x="RegYear", y="Volume", color="Status_Norm", title="2. Registration Volume Trajectory by Status Over Time", color_discrete_map=color_map_status))

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            reg_by_type = df.dropna(subset=["RegYear"]).groupby(["RegYear", "RegType_Norm"]).size().reset_index(name="Volume")
            reg_by_type = reg_by_type[reg_by_type["RegYear"].between(1950, 2026)]
            safe_plotly_render(px.line(reg_by_type, x="RegYear", y="Volume", color="RegType_Norm", title="3. Historical Trend of Permanent vs Provisional Classes", color_discrete_sequence=palette_list))
        with r2c2:
            cum_trend = reg_trend.copy()
            cum_trend["Cumulative"] = cum_trend["Registrations"].cumsum()
            safe_plotly_render(px.line(cum_trend, x="RegYear", y="Cumulative", title="4. Cumulative Total System Registered Capacity Growth", color_discrete_sequence=["#2A9D8F"]))

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            reg_gen = exclude_unknown(df.dropna(subset=["RegYear"]), "Gender").groupby(["RegYear", "Gender"]).size().reset_index(name="Volume")
            reg_gen = reg_gen[reg_gen["RegYear"].between(1950, 2026)]
            safe_plotly_render(px.line(reg_gen, x="RegYear", y="Volume", color="Gender", title="5. Chronological Registration Influx by Gender Mix", color_discrete_sequence=palette_list))
        with r3c2:
            safe_plotly_render(px.box(df, x="Status_Norm", y="RegYear", title="6. Historical Registration Cohort Density Dispersion Spread", color="Status_Norm", color_discrete_map=color_map_status))

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            reg_src = df.dropna(subset=["RegYear"]).groupby(["RegYear", "source_label"]).size().reset_index(name="Volume")
            reg_src = reg_src[reg_src["RegYear"].between(1950, 2026)]
            safe_plotly_render(px.bar(reg_src, x="RegYear", y="Volume", color="source_label", title="7. Historical Influx Share per Data Series Batch", barmode="stack", color_discrete_sequence=palette_list))
        with r4c2:
            safe_plotly_render(px.scatter(reg_by_status, x="RegYear", y="Volume", size="Volume", color="Status_Norm", title="8. Temporal Status Inflow Clustering & Outliers Map", color_discrete_map=color_map_status))

        r5c1, r5c2 = st.columns(2)
        with r5c1:
            safe_plotly_render(px.box(df, x="RegType_Norm", y="RegYear", title="9. Allocation Age Cohort Matrix by License Class", color="RegType_Norm", color_discrete_sequence=palette_list))
        with r5c2:
            exp_trend = df.dropna(subset=["ExpiryYear"]).groupby("ExpiryYear").size().reset_index(name="Count")
            exp_trend = exp_trend[exp_trend["ExpiryYear"].between(2020, 2040)]
            safe_plotly_render(px.bar(exp_trend, x="ExpiryYear", y="Count", title="10. Forward-Looking Systemic License Expiry Vector Projections", color_discrete_sequence=["#EF4444"]))

    # ========================================================================
    # TAB 3: REGIONAL STATUS
    # ========================================================================
    with tab3:
        st.markdown("<h3 style='color:#9A6B00;'>Regional Breakdown (AJK ground-truth; other provinces inferred)</h3>", unsafe_allow_html=True)
        prov_df = df[df["Province"] != "Unknown"]

        if prov_df.empty:
            st.info("No province-mappable records in current filter.")
        else:
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                prov_stat = prov_df.groupby(["Province", "Status_Norm"]).size().reset_index(name="Count")
                fig9 = px.bar(prov_stat, x="Count", y="Province", color="Status_Norm", orientation="h", barmode="stack", title="1. Status Load & Distribution by Province", color_discrete_map=color_map_status)
                fig9.update_layout(yaxis={"categoryorder": "total ascending"})
                safe_plotly_render(fig9)
            with r1c2:
                fig_sun2 = px.sunburst(prov_df, path=["Province", "RegType_Norm", "Status_Norm"], title="2. Province → Reg. Type → Status Routing", color="Status_Norm", color_discrete_map=color_map_status)
                fig_sun2.update_traces(textinfo="label+percent entry")
                safe_plotly_render(fig_sun2)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                inactive_df = prov_df[prov_df["Status_Norm"] == "In-Active"]
                if not inactive_df.empty:
                    bad_counts = inactive_df["Province"].value_counts().reset_index()
                    bad_counts.columns = ["Province", "Count"]
                    safe_plotly_render(px.pie(bad_counts, names="Province", values="Count", hole=0.5, title="3. In-Active Records Load by Province", color_discrete_sequence=px.colors.sequential.OrRd[2:]))
                else:
                    st.info("No In-Active records found.")
            with r2c2:
                prov_type = prov_df.groupby(["Province", "RegType_Norm"]).size().reset_index(name="Volume")
                safe_plotly_render(px.bar(prov_type, x="Province", y="Volume", color="RegType_Norm", barmode="group", title="4. Registration Class Variance across Provinces", color_discrete_sequence=palette_list))

            r3c1, r3c2 = st.columns(2)
            with r3c1:
                prov_gen = exclude_unknown(prov_df, "Gender").groupby(["Province", "Gender"]).size().reset_index(name="Volume")
                safe_plotly_render(px.bar(prov_gen, x="Volume", y="Province", color="Gender", orientation="h", barmode="stack", title="5. Territorial Workforce Balance", color_discrete_sequence=palette_list))
            with r3c2:
                prov_src = prov_df.groupby(["Province", "source_label"]).size().reset_index(name="Volume")
                safe_plotly_render(px.scatter(prov_src, x="Province", y="source_label", size="Volume", title="6. Data Series Structural Distribution over Provinces", color_discrete_sequence=["#E8C547"]))

            r4c1, r4c2 = st.columns(2)
            with r4c1:
                safe_plotly_render(px.box(prov_df, x="Province", y="RegYear", title="7. Regional Registration Entry Timeline Footprint Spread", color="Province", color_discrete_sequence=palette_list))
            with r4c2:
                prov_yr_scat = prov_df.groupby(["Province", "RegYear"]).size().reset_index(name="Count")
                safe_plotly_render(px.scatter(prov_yr_scat, x="RegYear", y="Province", size="Count", title="8. Inflow Concentration Density Clusters by Territory", color_discrete_sequence=["#2A9D8F"]))

            st.markdown("<br>", unsafe_allow_html=True)
            active_prov = prov_df[prov_df["Status_Norm"] == "Active"].groupby("Province").size().reset_index(name="Count")
            safe_plotly_render(px.bar(active_prov, x="Province", y="Count", title="9. Net Active Compliance Capacity Strength per Province", color_discrete_sequence=["#2A9D8F"]))


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="PMDC Intelligence Engine")
    render_status_license()
