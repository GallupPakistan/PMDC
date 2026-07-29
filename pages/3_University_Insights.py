import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import logging

from utils.gender_province import (
    infer_gender_series,
    infer_province_series,
    SOURCE_LABEL_MAP,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# REFERENCE MAPPINGS (gender/province now come from utils.gender_province)
# ============================================================================

FOREIGN_UNIVERSITY_KEYWORDS = [
    "china", "chinese", "xinjiang", "fujian", "guangxi", "guangzhou", "shandong",
    "wuhan", "nanjing", "tianjin", "anhui", "harbin", "zhengzhou", "qingdao",
    "kyrgyz", "osh", "bishkek", "kazan", "kazakhstan", "russia", "russian",
    "ukraine", "ukrainian", "georgia", "georgian", "tbilisi", "armenia",
    "azerbaijan", "uzbekistan", "tajik", "nepal", "bangladesh", "sri lanka",
    "philippine", "manila", "cebu", "caribbean", "grenada", "saint james",
    "egypt", "cairo", "al-azhar", "sudan", "khartoum", "iran", "tehran",
    "iraq", "baghdad", "saudi", "riyadh", "jeddah", "uk ", "united kingdom",
    "usa", "united states", "australia",
]

STATUS_RULES = [
    ("In-Active", re.compile(r"in[\s\-]?active", re.I)),
    ("Suspended", re.compile(r"suspend", re.I)),
    ("Cancelled", re.compile(r"cancel", re.I)),
    ("Active", re.compile(r"^active$", re.I)),
]

def normalize_status(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"
    s = raw.strip()
    for label, pattern in STATUS_RULES:
        if pattern.search(s):
            return label
    if "ACTIVE" in s.upper():
        return "Active"
    return "Other"

def is_foreign_university(university: str) -> str:
    if not isinstance(university, str) or not university.strip():
        return "Unknown"
    u = university.lower()
    for kw in FOREIGN_UNIVERSITY_KEYWORDS:
        if kw in u:
            return "Foreign"
    return "Domestic"

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

def exclude_unknown(data: pd.DataFrame, *cols) -> pd.DataFrame:
    """Drop rows where any of the given columns is 'Unknown', so charts never
    show an 'Unknown' bar/slice/line for that field."""
    mask = pd.Series(True, index=data.index)
    for c in cols:
        if c in data.columns:
            mask &= (data[c].astype(str).str.strip() != "Unknown")
    return data[mask]

# ============================================================================
# DATA LOADING + TRANSFORM
# ============================================================================
def flatten_qualifications(df: pd.DataFrame) -> pd.DataFrame:
    def get_first_qual(quals, key):
        if isinstance(quals, list) and len(quals) > 0 and isinstance(quals[0], dict):
            return quals[0].get(key)
        return None

    df = df.copy()
    df["Qualification_1_Speciality"] = df["Qualifications"].apply(lambda q: get_first_qual(q, "Speciality"))
    df["Qualification_1_Degree"] = df["Qualifications"].apply(lambda q: get_first_qual(q, "Degree"))
    df["Qualification_1_University"] = df["Qualifications"].apply(lambda q: get_first_qual(q, "University"))
    df["Qualification_1_PassingYear"] = df["Qualifications"].apply(lambda q: get_first_qual(q, "PassingYear"))

    df["Qualification_1_PassingYear"] = pd.to_numeric(df["Qualification_1_PassingYear"], errors="coerce")
    n_before = len(df)
    df = df[df["Qualification_1_PassingYear"].between(1940, 2018, inclusive="both")]
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"Dropped {n_dropped} rows with missing/invalid primary PassingYear "
                       f"({n_dropped/n_before*100:.1f}% of loaded data).")
    df["Qualification_1_PassingYear"] = df["Qualification_1_PassingYear"].astype(int)

    df["Qualification_1_Degree"] = df["Qualification_1_Degree"].apply(canonicalize_degree)
    df["Qualification_1_University"] = df["Qualification_1_University"].fillna("Unknown").str.strip()

    df["Location_Type"] = df["Qualification_1_University"].apply(is_foreign_university)
    df["Status_Norm"] = df.get("Status", pd.Series(dtype=str)).apply(normalize_status)
    df["source_label"] = df.get("source_table", pd.Series(dtype=str)).map(SOURCE_LABEL_MAP).fillna(df.get("source_table"))
    
    # Gender: use the real field if present; where it's genuinely missing, infer
    # from the doctor's Name using the shared first-name-only matcher (fixes
    # the old substring bug, e.g. "Abiha Arif Butt" no longer becomes "Male"
    # just because "Arif" appeared somewhere in the full name).
    if "Gender" not in df.columns:
        df["Gender"] = "Unknown"
    df["Gender"] = df["Gender"].fillna("Unknown")
    if "Name" in df.columns:
        inferred_gender = infer_gender_series(df["Name"], df["Qualification_1_University"])
        unknown_mask = df["Gender"].astype(str).str.strip().isin(["", "Unknown", "nan"])
        df.loc[unknown_mask, "Gender"] = inferred_gender[unknown_mask]

    # Province: use the real field if present; where it's genuinely missing,
    # infer from the university name — AJK-series records get Province="AJK"
    # with full confidence (ground truth from the collection they came from).
    if "Province" not in df.columns:
        df["Province"] = "Unknown"
    df["Province"] = df["Province"].fillna("Unknown")
    inferred_province = infer_province_series(df["Qualification_1_University"], df.get("source_table"))
    prov_unknown_mask = df["Province"].astype(str).str.strip().isin(["", "Unknown", "nan"])
    df.loc[prov_unknown_mask, "Province"] = inferred_province[prov_unknown_mask]

    return df

@st.cache_data(ttl=3600)
def load_university_data(use_real_db: bool):
    if not use_real_db:
        return pd.DataFrame()
    try:
        from utils.data_loader import load_all_data
        df = load_all_data()
        if df is None or df.empty:
            return pd.DataFrame()
        return flatten_qualifications(df)
    except Exception as e:
        logger.error(f"DB load failed: {e}")
        return pd.DataFrame()

# ============================================================================
# CSS & PREMIUM RENDER ENGINE
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
        .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            height: 50px; white-space: pre-wrap; background-color: transparent;
            border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
            color: #374151; font-weight: 600;
        }
        .stTabs [aria-selected="true"] { color: #9A6B00 !important; border-bottom: 2px solid #9A6B00 !important; }
        .caveat { color: #6B7280; font-size: 0.8rem; font-style: italic; margin-top: -4px; margin-bottom: 12px; }
        </style>
    """, unsafe_allow_html=True)

def render_premium_chart(chart_func, *args, **kwargs):
    """Unified engine to process styles with ultra-high contrast for headings & legends."""
    chart_height = kwargs.pop("chart_height", 320)
    custom_legend = kwargs.pop("legend", None)
    layout_theme = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#374151", family="Inter", size=10),
        margin=dict(t=45, b=25, l=25, r=25),
        height=chart_height,
        title_font=dict(size=13, color="#9A6B00", family="Cinzel"),
        showlegend=kwargs.pop("showlegend", True)
    )
    
    try:
        fig = chart_func(*args, **kwargs)
        fig.update_layout(**layout_theme)
        
        if any(x in str(type(fig)) for x in ["Sunburst", "Treemap", "Figure"]):
            fig.update_traces(marker=dict(line=dict(width=0)))
            
        if hasattr(fig, "update_xaxes"):
            # Ensure text labels on axis ticks are dark and readable on white
            fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#374151"))
            fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)", zeroline=False, tickfont=dict(color="#374151"))
            
        if custom_legend is not None:
            fig.update_layout(legend=custom_legend)
        else:
            fig.update_layout(legend=dict(
                bgcolor="rgba(0,0,0,0)", 
                font=dict(size=10, color="#374151"),
                title=dict(font=dict(color="#9A6B00"))
            ))
            
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        logger.error(f"Render Engine Exception: {e}")
        st.error("⚠️ Visual layout rendering format error.")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def render_university_insights():
    inject_custom_css()

    st.markdown("<h1 style='color: #9A6B00; font-family: Cinzel;'>🏛️ Institutional & University Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4B5563; margin-bottom: 2rem;'>Symmetrical macro intelligence dashboard auditing operational tracks across primary alma maters.</p>", unsafe_allow_html=True)

    use_real_db = st.session_state.get("db_connected", False)
    df = load_university_data(use_real_db=use_real_db)

    if df.empty:
        st.warning("No data loaded. Check DB connection / `db_connected` session state.")
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.markdown("<h3 style='color:#9A6B00;'>⚙️ Filter Pipeline</h3>", unsafe_allow_html=True)

    min_year, max_year = int(df["Qualification_1_PassingYear"].min()), int(df["Qualification_1_PassingYear"].max())
    year_range = st.sidebar.slider("Graduation Year Range", min_value=min_year, max_value=2018, value=(min_year, 2018))

    loc_filter = st.sidebar.multiselect("Institution Location (Inferred)", options=["Domestic", "Foreign", "Unknown"],
                                       default=["Domestic", "Foreign", "Unknown"])

    source_opts = sorted(df["source_label"].dropna().unique())
    selected_sources = st.sidebar.selectbox("Series / Collection", ["All"] + source_opts)

    # --- INDEPENDENT DRILL-DOWN SLICERS: each is separate and applies on its own ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("<p style='color:#9A6B00; font-weight:600; margin-bottom:0.3rem;'>🔎 Drill Down Filters</p>", unsafe_allow_html=True)

    prov_opts = sorted([p for p in df["Province"].dropna().unique() if p != "Unknown"])
    prov_filter = st.sidebar.selectbox("Province", ["All Provinces"] + prov_opts)

    domestic_univ_opts = sorted(df[df["Location_Type"] == "Domestic"]["Qualification_1_University"].dropna().unique())
    domestic_univ_filter = st.sidebar.selectbox("Domestic University", ["All Domestic Universities"] + domestic_univ_opts)

    foreign_univ_opts = sorted(df[df["Location_Type"] == "Foreign"]["Qualification_1_University"].dropna().unique())
    foreign_univ_filter = st.sidebar.selectbox("Foreign University", ["All Foreign Universities"] + foreign_univ_opts)

    # Apply Filter Constraints
    df = df[(df["Qualification_1_PassingYear"] >= year_range[0]) & (df["Qualification_1_PassingYear"] <= year_range[1])]
    if loc_filter:
        df = df[df["Location_Type"].isin(loc_filter)]
    if selected_sources != "All":
        df = df[df["source_label"] == selected_sources]
    if prov_filter != "All Provinces":
        df = df[df["Province"] == prov_filter]
    if domestic_univ_filter != "All Domestic Universities":
        df = df[df["Qualification_1_University"] == domestic_univ_filter]
    if foreign_univ_filter != "All Foreign Universities":
        df = df[df["Qualification_1_University"] == foreign_univ_filter]

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        return

    total_grads = len(df)
    unique_unis = df["Qualification_1_University"].nunique()
    top_uni = df["Qualification_1_University"].mode()[0] if not df.empty else "N/A"
    foreign_pct = (df["Location_Type"] == "Foreign").mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Graduates</div><div class="kpi-value">{total_grads:,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Tracked Institutions</div><div class="kpi-value" style="color: #2A9D8F;">{unique_unis}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Top Producing Uni</div><div class="kpi-value" style="font-size: 1.1rem; line-height: 2.4rem; color: #9A6B00;">{top_uni}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Foreign Graduates (Inferred)</div><div class="kpi-value">{foreign_pct:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    color_seq = ["#E8C547", "#2A9D8F", "#C9A84C", "#EF4444", "#64748B"]

    tab1, tab2, tab3 = st.tabs(["🏛️ Global Overview", "🩺 Program Deep-Dive", "🔬 Advanced Analytics"])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1: GLOBAL OVERVIEW (10 VISUALS)
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("<h3 style='color:#9A6B00; font-family:Cinzel;'>Macro Institutional Trends</h3>", unsafe_allow_html=True)
        
        # Row 1
        t1_r1_c1, t1_r1_c2 = st.columns(2)
        with t1_r1_c1:
            # Treemap now includes every university in the data, not just the top 15.
            # Treemaps naturally scale to many items (box size just gets smaller),
            # so no dynamic height is needed here.
            uni_counts = df["Qualification_1_University"].value_counts().reset_index()
            uni_counts.columns = ["University", "Graduates"]
            render_premium_chart(px.treemap, uni_counts, path=["University"], values="Graduates",
                                 title="1. All Universities by Graduate Volume", color_discrete_sequence=color_seq)
        with t1_r1_c2:
            loc_counts = df["Location_Type"].value_counts().reset_index()
            loc_counts.columns = ["Location", "Count"]
            render_premium_chart(px.pie, loc_counts, names="Location", values="Count", hole=0.6,
                                 title="2. Domestic vs. Foreign Sourcing Inflow", color_discrete_sequence=["#2A9D8F", "#E8C547", "#64748B"])
            st.markdown('<p class="caveat">Location inferred from university name keywords</p>', unsafe_allow_html=True)

        # Row 2
        t1_r2_c1, t1_r2_c2 = st.columns(2)
        with t1_r2_c1:
            # FIX FOR CHART 3: Processing explicit aggregated formats to eliminate line trace exceptions
            top_unis = exclude_unknown(df, "Qualification_1_University")["Qualification_1_University"].value_counts().head(5).index.tolist()
            trend_counts = df[df["Qualification_1_University"].isin(top_unis)].groupby(["Qualification_1_PassingYear", "Qualification_1_University"]).size().reset_index(name="Graduates")
            trend_counts = trend_counts.sort_values(by="Qualification_1_PassingYear")
            
            render_premium_chart(px.line, trend_counts, x="Qualification_1_PassingYear", y="Graduates", color="Qualification_1_University",
                                 title="3. Production Velocity Trends: Top 5 Hubs", color_discrete_sequence=color_seq,
                                 legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="left", x=0, font=dict(size=9, color="#374151")))
        with t1_r2_c2:
            stat_loc = df.groupby(["Location_Type", "Status_Norm"]).size().reset_index(name="Count")
            render_premium_chart(px.bar, stat_loc, x="Location_Type", y="Count", color="Status_Norm",
                                 title="4. Licensing Registration Standing by Origin", color_discrete_sequence=color_seq)

        # Row 3
        t1_r3_c1, t1_r3_c2 = st.columns(2)
        with t1_r3_c1:
            timeline_v = df.groupby('Qualification_1_PassingYear').size().reset_index(name='Volume')
            render_premium_chart(px.area, timeline_v, x='Qualification_1_PassingYear', y='Volume',
                                 title="5. National Aggregate Workforce Inflow Profile", color_discrete_sequence=["#2A9D8F"])
        with t1_r3_c2:
            # Full (non-truncated) counts, used by both the Graph and Table views.
            top_hubs_full = exclude_unknown(df, "Qualification_1_University")["Qualification_1_University"].value_counts().reset_index()
            top_hubs_full.columns = ["Institution", "Volume"]

            view_mode_ui1 = st.radio("View", ["📊 Graph", "📋 Table"], horizontal=True, key="ui_hubs_view", label_visibility="collapsed")

            if view_mode_ui1 == "📊 Graph":
                top_hubs = top_hubs_full.copy()
                if len(top_hubs) > 10:
                    top_part = top_hubs.head(10)
                    others_sum = top_hubs.iloc[10:]["Volume"].sum()
                    top_hubs = pd.concat([top_part, pd.DataFrame([{"Institution": "Others", "Volume": others_sum}])], ignore_index=True)
                top_hubs = top_hubs.sort_values("Volume", ascending=True)
                render_premium_chart(px.bar, top_hubs, x="Volume", y="Institution", orientation="h",
                                     title="6. Top 10 Academic Sourcing Institutions", color_discrete_sequence=["#E8C547"])
            else:
                st.dataframe(top_hubs_full.sort_values("Volume", ascending=False), use_container_width=True, hide_index=True, height=340)

        # Row 4
        t1_r4_c1, t1_r4_c2 = st.columns(2)
        with t1_r4_c1:
            series_dist = df.groupby(["source_label", "Location_Type"]).size().reset_index(name="Count")
            render_premium_chart(px.bar, series_dist, x="source_label", y="Count", color="Location_Type",
                                 title="7. Collection Pipeline Inflow Split per Series", barmode="group", color_discrete_sequence=["#2A9D8F", "#E8C547"])
        with t1_r4_c2:
            cum_supply = timeline_v.copy()
            cum_supply['Cumulative'] = cum_supply['Volume'].cumsum()
            render_premium_chart(px.line, cum_supply, x='Qualification_1_PassingYear', y='Cumulative',
                                 title="8. Cumulative Workforce Velocity Growth Vector", color_discrete_sequence=["#E8C547"])

        # Row 5
        t1_r5_c1, t1_r5_c2 = st.columns(2)
        with t1_r5_c1:
            # FIX FOR CHART 9: Handled fallback logic properly to render correctly even with temporary filtered/missing data pools
            prov_mix = exclude_unknown(df, "Province").groupby(["Province", "Location_Type"]).size().reset_index(name="Count")
            render_premium_chart(px.bar, prov_mix, x="Province", y="Count", color="Location_Type",
                                 title="9. Territorial Distribution Mix by Inflow Origin", barmode="stack", color_discrete_sequence=color_seq)
        with t1_r5_c2:
            top5_all = exclude_unknown(df, "Qualification_1_University")["Qualification_1_University"].value_counts().head(5).index.tolist()
            sun_df = df[df["Qualification_1_University"].isin(top5_all)]
            render_premium_chart(px.sunburst, sun_df, path=["Location_Type", "Qualification_1_Degree", "Qualification_1_University"],
                                 title="10. Geo-Institutional Sourcing Architecture Map", color_discrete_sequence=color_seq)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2: PROGRAM DEEP-DIVE (10 VISUALS)
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("<h3 style='color:#9A6B00; font-family:Cinzel;'>Programmatic Track Segmentation</h3>", unsafe_allow_html=True)
        top_degrees = exclude_unknown(df, "Qualification_1_Degree")["Qualification_1_Degree"].value_counts().head(2).index.tolist()
        
        # Row 1
        t2_r1_c1, t2_r1_c2 = st.columns(2)
        with t2_r1_c1:
            if len(top_degrees) >= 1:
                d1_df_full = exclude_unknown(df[df["Qualification_1_Degree"] == top_degrees[0]], "Qualification_1_University")["Qualification_1_University"].value_counts().reset_index()
                d1_df_full.columns = ["University", "Count"]

                view_mode_ui2 = st.radio("View", ["📊 Graph", "📋 Table"], horizontal=True, key="ui_deg1_view", label_visibility="collapsed")

                if view_mode_ui2 == "📊 Graph":
                    d1_df = d1_df_full.copy()
                    if len(d1_df) > 10:
                        top_part = d1_df.head(10)
                        others_sum = d1_df.iloc[10:]["Count"].sum()
                        d1_df = pd.concat([top_part, pd.DataFrame([{"University": "Others", "Count": others_sum}])], ignore_index=True)
                    d1_df = d1_df.sort_values("Count", ascending=True)
                    render_premium_chart(px.bar, d1_df, x="Count", y="University", orientation="h",
                                         title=f"1. Top 10 Universities — {top_degrees[0]}", color_discrete_sequence=["#2A9D8F"])
                else:
                    st.dataframe(d1_df_full.sort_values("Count", ascending=False), use_container_width=True, hide_index=True, height=340)
        with t2_r1_c2:
            if len(top_degrees) >= 2:
                d2_df_full = exclude_unknown(df[df["Qualification_1_Degree"] == top_degrees[1]], "Qualification_1_University")["Qualification_1_University"].value_counts().reset_index()
                d2_df_full.columns = ["University", "Count"]

                view_mode_ui3 = st.radio("View", ["📊 Graph", "📋 Table"], horizontal=True, key="ui_deg2_view", label_visibility="collapsed")

                if view_mode_ui3 == "📊 Graph":
                    d2_df = d2_df_full.copy()
                    if len(d2_df) > 10:
                        top_part = d2_df.head(10)
                        others_sum = d2_df.iloc[10:]["Count"].sum()
                        d2_df = pd.concat([top_part, pd.DataFrame([{"University": "Others", "Count": others_sum}])], ignore_index=True)
                    d2_df = d2_df.sort_values("Count", ascending=True)
                    render_premium_chart(px.bar, d2_df, x="Count", y="University", orientation="h",
                                         title=f"2. Top 10 Universities — {top_degrees[1]}", color_discrete_sequence=["#E8C547"])
                else:
                    st.dataframe(d2_df_full.sort_values("Count", ascending=False), use_container_width=True, hide_index=True, height=340)

        # Row 2
        t2_r2_c1, t2_r2_c2 = st.columns(2)
        with t2_r2_c1:
            deg_mix_global = df["Qualification_1_Degree"].value_counts().reset_index().head(8)
            deg_mix_global.columns = ["Degree", "Volume"]
            render_premium_chart(px.pie, deg_mix_global, names="Degree", values="Volume",
                                 title="3. Structural Core Degree Mix Shares", color_discrete_sequence=color_seq)
        with t2_r2_c2:
            deg_time_trend = df[df["Qualification_1_Degree"].isin(top_degrees)].groupby(["Qualification_1_PassingYear", "Qualification_1_Degree"]).size().reset_index(name="Volume")
            render_premium_chart(px.line, deg_time_trend, x="Qualification_1_PassingYear", y="Volume", color="Qualification_1_Degree",
                                 title="4. Academic Program Graduation Velocity Trajectory", color_discrete_sequence=["#2A9D8F", "#E8C547"])

        # Row 3
        t2_r3_c1, t2_r3_c2 = st.columns(2)
        with t2_r3_c1:
            deg_status_matrix = df.groupby(["Qualification_1_Degree", "Status_Norm"]).size().reset_index(name="Count")
            deg_status_matrix = deg_status_matrix[deg_status_matrix["Qualification_1_Degree"].isin(top_degrees)]
            render_premium_chart(px.bar, deg_status_matrix, x="Qualification_1_Degree", y="Count", color="Status_Norm",
                                 title="5. Compliance & License Class per Program Type", barmode="group", color_discrete_sequence=color_seq)
        with t2_r3_c2:
            render_premium_chart(px.box, df[df["Qualification_1_Degree"].isin(top_degrees)], x="Qualification_1_Degree", y="Qualification_1_PassingYear",
                                 title="6. Historical Program Cohort Dispersal Spread", color_discrete_sequence=["#64748B"])

        # Row 4
        t2_r4_c1, t2_r4_c2 = st.columns(2)
        with t2_r4_c1:
            spec_pool = df[df["Qualification_1_Speciality"].notna() & (df["Qualification_1_Speciality"] != "Unknown")]["Qualification_1_Speciality"].value_counts().reset_index().head(10)
            spec_pool.columns = ["Specialty", "Volume"]
            render_premium_chart(px.bar, spec_pool, x="Volume", y="Specialty", orientation="h",
                                 title="7. Top Primary Specialization Pathways", color_discrete_sequence=["#E8C547"])
        with t2_r4_c2:
            deg_loc_mix = df.groupby(["Qualification_1_Degree", "Location_Type"]).size().reset_index(name="Volume")
            deg_loc_mix = deg_loc_mix[deg_loc_mix["Qualification_1_Degree"].isin(df["Qualification_1_Degree"].value_counts().head(5).index)]
            render_premium_chart(px.bar, deg_loc_mix, x="Qualification_1_Degree", y="Volume", color="Location_Type",
                                 title="8. Degree Modality Classification by Sourcing Stream", barmode="stack", color_discrete_sequence=["#2A9D8F", "#E8C547"])

        # Row 5
        t2_r5_c1, t2_r5_c2 = st.columns(2)
        with t2_r5_c1:
            spec_status_dist = df[df["Qualification_1_Speciality"].isin(spec_pool["Specialty"].head(5))].groupby(["Qualification_1_Speciality", "Status_Norm"]).size().reset_index(name="Count")
            render_premium_chart(px.bar, spec_status_dist, x="Qualification_1_Speciality", y="Count", color="Status_Norm",
                                 title="9. Specialization Domain Audit & Compliance Standing", barmode="group", color_discrete_sequence=color_seq)
        with t2_r5_c2:
            cross_cluster = df.groupby(["source_label", "Qualification_1_Degree"]).size().reset_index(name="Volume")
            cross_cluster = cross_cluster[cross_cluster["Qualification_1_Degree"].isin(df["Qualification_1_Degree"].value_counts().head(4).index)]
            render_premium_chart(px.scatter, cross_cluster, x="source_label", y="Qualification_1_Degree", size="Volume",
                                 title="10. Series-to-Degree Mapping Densities Matrix", color_discrete_sequence=["#EF4444"])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3: ADVANCED ANALYTICS (10 VISUALS)
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("<h3 style='color:#9A6B00; font-family:Cinzel;'>Data Integrity & Inferred Audit Systems</h3>", unsafe_allow_html=True)
        top4 = exclude_unknown(df, "Qualification_1_University")["Qualification_1_University"].value_counts().head(4).index.tolist()
        sankey_df = df[df["Qualification_1_University"].isin(top4)]
        
        # Row 1
        t3_r1_c1, t3_r1_c2 = st.columns(2)
        with t3_r1_c1:
            gen_distribution = exclude_unknown(df, "Gender")["Gender"].value_counts().reset_index()
            gen_distribution.columns = ["Gender", "Count"]
            render_premium_chart(px.pie, gen_distribution, names="Gender", values="Count", hole=0.5,
                                 title="1. Inferred Demographics Gender Allocation Mix", color_discrete_sequence=["#E8C547", "#2A9D8F", "#64748B"])
        with t3_r1_c2:
            gen_time_stream = exclude_unknown(df, "Gender").groupby(["Qualification_1_PassingYear", "Gender"]).size().reset_index(name="Volume")
            render_premium_chart(px.line, gen_time_stream, x="Qualification_1_PassingYear", y="Volume", color="Gender",
                                 title="2. Historical Gender Inflow Trajectory Paradigms", color_discrete_sequence=["#E8C547", "#2A9D8F"])

        # Row 2
        t3_r2_c1, t3_r2_c2 = st.columns(2)
        with t3_r2_c1:
            st.markdown("<p style='color: #9A6B00; font-size:0.85rem; font-weight: bold; margin-bottom:-10px;'>3. Workforce Flow Architecture Map</p>", unsafe_allow_html=True)
            degree_nodes = sankey_df["Qualification_1_Degree"].value_counts().head(5).index.tolist()
            status_nodes = sankey_df["Status_Norm"].unique().tolist()
            nodes = top4 + degree_nodes + status_nodes
            node_dict = {node: i for i, node in enumerate(nodes)}
            source, target, value = [], [], []

            for uni in top4:
                for deg in degree_nodes:
                    count = len(sankey_df[(sankey_df["Qualification_1_University"] == uni) & (sankey_df["Qualification_1_Degree"] == deg)])
                    if count > 0:
                        source.append(node_dict[uni]); target.append(node_dict[deg]); value.append(count)
            for deg in degree_nodes:
                for stat in status_nodes:
                    count = len(sankey_df[(sankey_df["Qualification_1_Degree"] == deg) & (sankey_df["Status_Norm"] == stat)])
                    if count > 0:
                        source.append(node_dict[deg]); target.append(node_dict[stat]); value.append(count)

            if source:
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="black", width=0), label=nodes, color="#E8C547"),
                    link=dict(source=source, target=target, value=value, color="rgba(42, 157, 143, 0.4)"),
                )])
                fig_sankey.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                         font=dict(color="#374151", size=10), margin=dict(t=20, b=20, l=15, r=15), height=270)
                st.plotly_chart(fig_sankey, use_container_width=True, config={'displayModeBar': False})
            st.markdown('<p class="caveat" style="margin-top:-5px;">Status tracking route mappings across primary cohorts</p>', unsafe_allow_html=True)
        with t3_r2_c2:
            render_premium_chart(px.box, sankey_df, x="Qualification_1_University", y="Qualification_1_PassingYear",
                                 title="4. Grad Year Density Spread — Top 4 Universities", color="Qualification_1_University", color_discrete_sequence=color_seq)

        # Row 3
        scatter_counts = df.groupby(["Qualification_1_PassingYear", "Location_Type"]).size().reset_index(name="Volume")
        render_premium_chart(px.scatter, scatter_counts, x="Qualification_1_PassingYear", y="Volume", size="Volume",
                             color="Location_Type", title="5. Graduate Volume Footprint Clustering Over Time", color_discrete_sequence=["#E8C547", "#2A9D8F", "#64748B"])


        # Row 4
        t3_r4_c1, t3_r4_c2 = st.columns(2)
        with t3_r4_c1:
            gen_loc_dist = exclude_unknown(df, "Gender").groupby(["Location_Type", "Gender"]).size().reset_index(name="Count")
            render_premium_chart(px.bar, gen_loc_dist, x="Location_Type", y="Count", color="Gender",
                                 title="6. Sourcing Stream Variation across Gender Identifiers", barmode="group", color_discrete_sequence=["#2A9D8F", "#E8C547", "#64748B"])
        with t3_r4_c2:
            uni_gender_mix = exclude_unknown(sankey_df, "Gender").groupby(["Qualification_1_University", "Gender"]).size().reset_index(name="Volume")
            render_premium_chart(px.bar, uni_gender_mix, x="Qualification_1_University", y="Volume", color="Gender",
                                 title="7. Top 4 Academic Hub Structural Gender Ratios", barmode="stack", color_discrete_sequence=["#E8C547", "#2A9D8F"])

        # Row 5
        t3_r5_c1, t3_r5_c2 = st.columns(2)
        with t3_r5_c1:
            time_status_decay = df.groupby(["Qualification_1_PassingYear", "Status_Norm"]).size().reset_index(name="Count")
            render_premium_chart(px.scatter, time_status_decay, x="Qualification_1_PassingYear", y="Status_Norm", size="Count",
                                 title="8. Dynamic Registration Expiry & Compliance Outliers Spread", color_discrete_sequence=["#E8C547"])
        with t3_r5_c2:
            source_gender_density = exclude_unknown(df, "Gender").groupby(["source_label", "Gender"]).size().reset_index(name="Volume")
            render_premium_chart(px.bar, source_gender_density, x="source_label", y="Volume", color="Gender",
                                 title="9. Collection Batch Distribution Demographics Validation", barmode="stack", color_discrete_sequence=color_seq)


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="PMDC Intelligence Engine")
    render_university_insights()
