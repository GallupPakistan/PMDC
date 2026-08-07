import logging
import re
import traceback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st

from utils.gender_province import (
    infer_gender,
    infer_province,
    SOURCE_LABEL_MAP,
    is_specialist_from_series,
)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# REFERENCE MAPPINGS & HEURISTICS (gender/province now come from utils.gender_province)
# ============================================================================

DEGREE_CANONICAL_MAP = {
    "mbbs": "MBBS", "bds": "BDS", "md": "MD", "ms": "MS",
    "fcps": "FCPS", "mcps": "MCPS", "dpt": "DPT", "phd": "Ph.D",
}


# Keyword list to split universities into Local vs International (same heuristic
# used on the Overview / Doctor Analytics / University Insights pages).
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

def is_foreign_university(university: str) -> str:
    if not isinstance(university, str) or not university.strip():
        return "Unknown"
    u = university.lower()
    for kw in FOREIGN_UNIVERSITY_KEYWORDS:
        if kw in u:
            return "International"
    return "Local"


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


# Buckets messy, free-text Status values (100+ distinct raw strings in the
# real data, e.g. "Cancelled Permanently w.e.f 14th Feb 2022", "DC-Action:
# 1145", "Suspended Till Appearance") down to a handful of clean categories.
# Without this, any chart grouped by raw "Status" ends up with a legend of
# dozens of near-duplicate entries - unreadable and overlapping.
# "in-active"/"inactive" are checked BEFORE "active" since "in-active"
# contains "active" as a substring (same class of bug fixed dashboard-wide).
STATUS_BUCKET_MAP = {"in-active": "In-Active", "inactive": "In-Active", "active": "Active", "suspended": "Suspended", "cancelled": "Cancelled", "cancel": "Cancelled"}

def bucket_status(s):
    if not isinstance(s, str):
        return "Other"
    key = s.strip().lower()
    for frag, buck in STATUS_BUCKET_MAP.items():
        if frag in key:
            return buck
    return "Other"


# ============================================================================
# AUTOMATED PREMIUM WRAPPER ENGINE
# ============================================================================
def render_premium_chart(chart_func, *args, **kwargs):
    """Executes a chart and strictly injects 100% borderless transparency over dark theme."""
    try:
        showlegend = kwargs.pop("showlegend", True)
        chart_height = kwargs.pop("chart_height", 340)
        fig = chart_func(*args, **kwargs)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#374151", family="Inter", size=11),
            title_font=dict(size=14, color="#9A6B00", family="Cinzel"),
            margin=dict(t=50, b=40, l=40, r=20),
            height=chart_height,
            showlegend=showlegend,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#374151", size=10))
        )
        if hasattr(fig, "update_xaxes"):
            fig.update_xaxes(showgrid=False, zeroline=False, title_font=dict(color="#374151", size=10))
            fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)", zeroline=False, title_font=dict(color="#374151", size=10))

        # Data labels on by default: percent+label for pies, percent-of-total
        # for treemaps, value+percent-of-first-stage for funnels. Bar charts
        # get their % labels set explicitly at each call site instead (each
        # one needs its own denominator - "% of all primary doctors", "% of
        # all postgrad rows", etc. - so one generic rule here can't cover them).
        trace_types = {trace.type for trace in fig.data}
        if "pie" in trace_types:
            fig.update_traces(textinfo="percent+label", textposition="inside", selector=dict(type="pie"))
        if "treemap" in trace_types:
            fig.update_traces(textinfo="label+percent entry", selector=dict(type="treemap"))
        if "funnel" in trace_types:
            fig.update_traces(textinfo="value+percent initial", selector=dict(type="funnel"))

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        logger.error(f"Visualization render bypass: {e}")
        st.warning("⚠️ Render bypass on complex visualization entity.")


# ============================================================================
# DATA PIPELINE
# ============================================================================
@st.cache_data(ttl=3600)
def load_qualification_data(use_real_db: bool):
    if not use_real_db:
        return pd.DataFrame()
    try:
        from utils.data_loader import load_all_data
        df = load_all_data()
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"DB load failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def build_qualifications_table(use_real_db: bool) -> pd.DataFrame:
    # NOTE: this used to take the raw DataFrame as its argument. Streamlit's
    # cache has to hash every argument to check for a cache hit, and hashing a
    # 200k+ row DataFrame that has nested list/dict columns (Qualifications)
    # is itself slow - slow enough that it could cost as much time as just
    # redoing the work, on every single filter change / rerun. Taking a plain
    # bool instead makes the cache check nearly instant.
    df = load_qualification_data(use_real_db)
    if df.empty or "Qualifications" not in df.columns:
        return pd.DataFrame()
    
    keep_cols = [
        c for c in [
            "RegistrationNo", "Name", "RegistrationType", "RegistrationDate", 
            "ValidUpto", "Status", "source_table", "Qualifications"
        ] if c in df.columns
    ]
    base = df[keep_cols].copy()
    base["Qualifications"] = base["Qualifications"].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else [{}])
    
    exploded = base.explode("Qualifications", ignore_index=True)
    qual_df = pd.json_normalize(exploded["Qualifications"])
    qual_df.index = exploded.index
    
    result = pd.concat([exploded.drop(columns=["Qualifications"]), qual_df], axis=1)
    result["PassingYear"] = pd.to_numeric(result.get("PassingYear"), errors="coerce")
    result = result[result["PassingYear"].between(1940, 2018, inclusive="both") | result["PassingYear"].isna()]

    # BUG FIX: QualSeq/IsPrimary used to be assigned purely by array position
    # (cumcount() on the rows in whatever order they came out of the scraped
    # Qualifications list). Real PMDC records are NOT always stored
    # chronologically - e.g. one doctor's Qualifications array lists their
    # 2024 FCPS BEFORE their 2017 MBBS - so array position alone can
    # misidentify a postgraduate qualification as "primary" and the doctor's
    # actual base degree as a "secondary" one. This silently corrupted the
    # primary/secondary split every chart on this page is built from.
    #
    # Fixed by sorting each doctor's qualification rows so their base degree
    # (MBBS/BDS) always comes first if they have one on file, then by
    # PassingYear (earliest first) as the tiebreaker/fallback for everything
    # else - THEN assigning QualSeq/IsPrimary from that corrected order.
    result["_degree_canonical_for_sort"] = result.get("Degree", pd.Series(dtype=str)).apply(canonicalize_degree)
    result["_is_not_base_degree"] = ~result["_degree_canonical_for_sort"].isin(["MBBS", "BDS"])
    result = result.sort_values(
        by=["RegistrationNo", "_is_not_base_degree", "PassingYear"],
        na_position="last",
        kind="stable"
    )
    result = result.drop(columns=["_degree_canonical_for_sort", "_is_not_base_degree"])

    result["QualSeq"] = result.groupby("RegistrationNo").cumcount() + 1
    result["IsPrimary"] = result["QualSeq"] == 1
    
    result["Gender"] = result["Name"].apply(infer_gender)
    result["Province"] = result.get("University", pd.Series(dtype=str)).apply(infer_province)
    if "source_table" in result.columns:
        result.loc[result["source_table"].astype(str).str.strip() == "doctors_ajk_series", "Province"] = "AJK"
    result["Location_Type"] = result.get("University", pd.Series(dtype=str)).apply(is_foreign_university)
    result["source_label"] = result["source_table"].map(SOURCE_LABEL_MAP).fillna(result.get("source_table", "P-Series"))
    result["Degree"] = result.get("Degree", pd.Series(dtype=str)).apply(canonicalize_degree)
    result["Speciality"] = result.get("Speciality", pd.Series(dtype=str)).fillna("Unknown").str.strip()

    # Ground-truth specialist flag: True iff this doctor's ORIGINAL record came
    # from the S-series collection (confirmed meaning: postgrad specialists),
    # not from guessing at free-text Speciality values.
    if "source_table" in result.columns:
        result["Is_Specialist_Series"] = is_specialist_from_series(result["source_table"])
    else:
        result["Is_Specialist_Series"] = False

    return result


# ============================================================================
# LAYOUT RENDERING ENGINE
# ============================================================================
def render_qualification_analytics():
    # Structural Dashboard Layout Configuration CSS
    st.markdown("""
        <style>
        .kpi-card { background: #FFFFFF; border: 1px solid rgba(184, 134, 11, 0.3); border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .kpi-title { color: #6B7280; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
        .kpi-value { color: #B8860B; font-size: 2rem; font-weight: 700; margin: 0; font-family: 'Cinzel', serif; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] { height: 50px; color: #6B7280 !important; font-weight: 600; background-color: transparent !important; }
        .stTabs [aria-selected="true"] { color: #9A6B00 !important; border-bottom: 2px solid #B8860B !important; }
        .caveat { color: #6B7280; font-size: 0.78rem; font-style: italic; margin-top: -8px; margin-bottom: 18px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='color: #B8860B; font-family: Cinzel, serif;'>Qualification & Specialization Analytics</h1>", unsafe_allow_html=True)
    
    use_real_db = st.session_state.get("db_connected", False)
    raw_df = load_qualification_data(use_real_db=use_real_db)

    if raw_df.empty:
        st.warning("⚠️ Accessing sandbox ecosystem or database loading inactive. Check connections.")
        return

    qdf = build_qualifications_table(use_real_db)
    if qdf.empty:
        st.error("❌ Array matrix transformation collapsed. No analytical layers identified.")
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.markdown("<h3 style='color:#9A6B00; font-family: Cinzel;'>⚙️ Filter Controls</h3>", unsafe_allow_html=True)
    source_opts = sorted(qdf["source_label"].dropna().unique())
    selected_sources = st.sidebar.selectbox("Series / Collection Scope", ["All"] + source_opts)
    degree_opts = sorted(qdf["Degree"].dropna().unique())
    selected_degrees = st.sidebar.multiselect("Degree Class Filters", degree_opts)
    
    valid_years = qdf["PassingYear"].dropna()
    year_range = st.sidebar.slider(
        "Passing Year Horizon",
        int(valid_years.min()),
        2018,
        (int(valid_years.min()), 2018)
    ) if not valid_years.empty else None

    # --- INDEPENDENT DRILL-DOWN SLICERS: each is separate and applies on its own ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("<p style='color:#9A6B00; font-weight:600; margin-bottom:0.3rem;'>🔎 Drill Down Filters</p>", unsafe_allow_html=True)

    prov_opts = sorted([p for p in qdf["Province"].dropna().unique() if p != "Unknown"])
    prov_filter = st.sidebar.selectbox("Province", ["All Provinces"] + prov_opts)

    country_opts = sorted([c for c in qdf["Location_Type"].dropna().unique() if c != "Unknown"])
    country_filter = st.sidebar.selectbox("Country (Local / International)", ["All"] + country_opts)

    local_univ_opts = sorted(qdf[qdf["Location_Type"] == "Local"]["University"].dropna().unique())
    local_univ_filter = st.sidebar.selectbox("Local University", ["All Local Universities"] + local_univ_opts)

    intl_univ_opts = sorted(qdf[qdf["Location_Type"] == "International"]["University"].dropna().unique())
    intl_univ_filter = st.sidebar.selectbox("International University", ["All International Universities"] + intl_univ_opts)

    # Filter Application
    filtered = qdf if selected_sources == "All" else qdf[qdf["source_label"] == selected_sources]
    if selected_degrees:
        filtered = filtered[filtered["Degree"].isin(selected_degrees)]
    if year_range:
        filtered = filtered[filtered["PassingYear"].between(year_range[0], year_range[1]) | filtered["PassingYear"].isna()]
    if prov_filter != "All Provinces":
        filtered = filtered[filtered["Province"] == prov_filter]
    if country_filter != "All":
        filtered = filtered[filtered["Location_Type"] == country_filter]
    if local_univ_filter != "All Local Universities":
        filtered = filtered[filtered["University"] == local_univ_filter]
    if intl_univ_filter != "All International Universities":
        filtered = filtered[filtered["University"] == intl_univ_filter]

    if filtered.empty:
        st.warning("⚠️ No records match the current filter selection.")
        return

    primary = filtered[filtered["IsPrimary"]]
    secondary = filtered[~filtered["IsPrimary"]]

    # --- HIGH LEVEL PERFORMANCE KPIs ---
    total_doctors = filtered["RegistrationNo"].nunique()
    total_qual_rows = len(filtered)
    multi_qual_doctors = filtered.groupby("RegistrationNo")["QualSeq"].max()
    pct_with_postgrad = (multi_qual_doctors > 1).mean() * 100 if len(multi_qual_doctors) else 0.0
    
    active_condition = filtered["Status"].astype(str).str.upper().str.contains("ACTIVE", na=False) & \
                       ~filtered["Status"].astype(str).str.upper().str.contains("IN-ACTIVE", na=False)
    active_pct = active_condition.mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Unique Doctors</div><div class="kpi-value">{total_doctors:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Qualification Entries</div><div class="kpi-value">{total_qual_rows:,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Postgrad Ratio</div><div class="kpi-value">{pct_with_postgrad:.1f}%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Active License Rate</div><div class="kpi-value">{active_pct:.1f}%</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎓 Base Qualifications", "🔬 Postgrad & Specialties", "🌐 Demographics (Inferred)"])
    COLOR_SEQ = ["#2A9D8F", "#C9A84C", "#E8C547", "#EF4444", "#64748B", "#0D3B2A", "#457B9D"]

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1: BASE QUALIFICATIONS
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        t1_r1_c1, t1_r1_c2 = st.columns(2)
        with t1_r1_c1:
            deg_pool = exclude_unknown(primary, "Degree")
            deg_counts = deg_pool["Degree"].value_counts().reset_index().head(10)
            deg_counts.columns = ["Degree", "Count"]
            # % of all doctors with a known primary degree (not just these 10 bars).
            deg_counts["Pct"] = (deg_counts["Count"] / len(deg_pool) * 100).round(1)
            render_premium_chart(px.bar, deg_counts, x="Degree", y="Count", title="1. Dominant Primary Degrees", color_discrete_sequence=["#2A9D8F"], text=deg_counts["Pct"].map(lambda p: f"{p}%"))
        with t1_r1_c2:
            yr_counts = primary.dropna(subset=["PassingYear"]).groupby("PassingYear").size().reset_index(name="Volume")
            render_premium_chart(px.area, yr_counts, x="PassingYear", y="Volume", title="2. Graduation Volume Profile Over Time", color_discrete_sequence=["#C9A84C"])

        t1_r2_c1, t1_r2_c2 = st.columns(2)
        with t1_r2_c1:
            render_premium_chart(px.treemap, exclude_unknown(primary, "Degree"), path=["source_label", "Degree"], title="3. Registration Series Flow to Degree Model", color_discrete_sequence=COLOR_SEQ)
        with t1_r2_c2:
            age_df = primary.dropna(subset=["PassingYear"]).copy()
            # Was hardcoded to a fixed year (2026) - would silently drift
            # wrong every year after that. Uses today's actual year instead.
            age_df["CareerAge"] = datetime.now().year - age_df["PassingYear"]
            age_df = age_df[age_df["CareerAge"].between(0, 70)]
            render_premium_chart(px.histogram, age_df, x="CareerAge", nbins=20, title="4. Career Age Distribution Metrics", color_discrete_sequence=["#E8C547"])

        t1_r3_c1, t1_r3_c2 = st.columns(2)
        with t1_r3_c1:
            seq_metrics = filtered["QualSeq"].value_counts().reset_index()
            seq_metrics.columns = ["Sequence", "Records"]
            # % of all qualification rows in the current filter.
            seq_metrics["Pct"] = (seq_metrics["Records"] / total_qual_rows * 100).round(1)
            render_premium_chart(px.bar, seq_metrics, x="Sequence", y="Records", title="5. Qualification Depth Density Structure", color_discrete_sequence=["#457B9D"], text=seq_metrics["Pct"].map(lambda p: f"{p}%"))
        with t1_r3_c2:
            src_vol = primary["source_label"].value_counts().reset_index()
            src_vol.columns = ["Series", "Volume"]
            render_premium_chart(px.pie, src_vol, names="Series", values="Volume", title="6. Registrant Share by Inflow Streams", hole=0.4, color_discrete_sequence=COLOR_SEQ)

        t1_r4_c1, t1_r4_c2 = st.columns(2)
        with t1_r4_c1:
            # Full (non-truncated) counts, used by both the Graph and Table views.
            univ_leader_full = exclude_unknown(primary, "University")["University"].value_counts().reset_index()
            univ_leader_full.columns = ["University", "Count"]

            view_mode_qa1 = st.radio("View", ["📊 Graph", "📋 Table"], horizontal=True, key="qa_univ_view", label_visibility="collapsed")

            if view_mode_qa1 == "📊 Graph":
                univ_leader = univ_leader_full.copy()
                if len(univ_leader) > 10:
                    top_part = univ_leader.head(10)
                    others_sum = univ_leader.iloc[10:]["Count"].sum()
                    univ_leader = pd.concat([top_part, pd.DataFrame([{"University": "Others", "Count": others_sum}])], ignore_index=True)
                univ_leader = univ_leader.sort_values("Count", ascending=True)
                # % is of ALL primary doctors with a known university (sums to
                # the same total as univ_leader_full, since "Others" absorbs
                # everything past the top 10).
                univ_leader["Pct"] = (univ_leader["Count"] / univ_leader["Count"].sum() * 100).round(1)
                render_premium_chart(px.bar, univ_leader, x="Count", y="University", orientation="h", title="7. Top 10 Primary Academic Sourcing Institutions", color_discrete_sequence=["#2A9D8F"], text=univ_leader["Pct"].map(lambda p: f"{p}%"))
            else:
                st.dataframe(univ_leader_full.sort_values("Count", ascending=False), use_container_width=True, hide_index=True, height=340)
        with t1_r4_c2:
            pass_box = primary[primary["Degree"].isin(["MBBS", "BDS"])].dropna(subset=["PassingYear"])
            render_premium_chart(px.box, pass_box, x="Degree", y="PassingYear", title="8. Graduation Cohort Era Mapping", color_discrete_sequence=["#C9A84C"])

        t1_r5_c1, t1_r5_c2 = st.columns(2)
        with t1_r5_c1:
            cum_growth = yr_counts.copy()
            cum_growth["Cumulative_Volume"] = cum_growth["Volume"].cumsum()
            render_premium_chart(px.line, cum_growth, x="PassingYear", y="Cumulative_Volume", title="9. Historical Cumulative Inflow Velocity", color_discrete_sequence=["#E8C547"])
        with t1_r5_c2:
            # .str.strip().str.title() normalizes case variants like "PERMANENT"
            # vs "Permanent" so they don't split into separate legend entries.
            type_mix_src = primary.copy()
            type_mix_src["RegistrationType"] = type_mix_src["RegistrationType"].astype(str).str.strip().str.title()
            type_mix = type_mix_src.groupby(["source_label", "RegistrationType"]).size().reset_index(name="Volume")
            # % of all primary doctors (across every series/type combo shown).
            type_mix["Pct"] = (type_mix["Volume"] / len(type_mix_src) * 100).round(1)
            render_premium_chart(px.bar, type_mix, x="source_label", y="Volume", color="RegistrationType", barmode="stack", title="10. Operational Scope Mix inside Series", color_discrete_sequence=COLOR_SEQ, text=type_mix["Pct"].map(lambda p: f"{p}%"))

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2: POSTGRAD & SPECIALTIES
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        # Ground-truth vs text-parsed specialist counts, side by side, so the
        # gap between "declared postgrad qualification row" and "S-series
        # registration" is visible rather than silently picking one.
        n_specialist_series = int(primary["Is_Specialist_Series"].sum()) if "Is_Specialist_Series" in primary.columns else 0
        st.caption(
            f"ℹ️ **{n_specialist_series:,}** doctors in the current filter are confirmed specialists via the "
            f"**S-series registry** (ground truth). The charts below additionally show anyone with a *second* "
            f"qualification row on file, which can include doctors outside the S-series too."
        )
        if secondary.empty:
            st.info("ℹ️ No multi-tier specialized registries found under active criteria filters.")
        else:
            t2_r1_c1, t2_r1_c2 = st.columns(2)
            with t2_r1_c1:
                sp_pool = secondary[~secondary["Speciality"].isin(["", ".", "Unknown"])]
                sp_counts = sp_pool["Speciality"].value_counts().reset_index().head(10)
                sp_counts.columns = ["Specialty", "Count"]
                # % of all secondary (postgrad) rows with a known specialty.
                sp_counts["Pct"] = (sp_counts["Count"] / len(sp_pool) * 100).round(1)
                render_premium_chart(px.bar, sp_counts, x="Count", y="Specialty", orientation="h", title="1. Dominant Specialized Domain Matrix", color_discrete_sequence=["#2A9D8F"], text=sp_counts["Pct"].map(lambda p: f"{p}%"))
            with t2_r1_c2:
                pg_deg_pool = exclude_unknown(secondary, "Degree")
                pg_deg = pg_deg_pool["Degree"].value_counts().reset_index().head(8)
                pg_deg.columns = ["Postgrad Degree", "Count"]
                # % of all secondary rows with a known postgrad degree.
                pg_deg["Pct"] = (pg_deg["Count"] / len(pg_deg_pool) * 100).round(1)
                render_premium_chart(px.bar, pg_deg, x="Postgrad Degree", y="Count", title="2. Postgrad Qualification Typology", color_discrete_sequence=["#C9A84C"], text=pg_deg["Pct"].map(lambda p: f"{p}%"))

            t2_r2_c1, t2_r2_c2 = st.columns(2)
            with t2_r2_c1:
                funnel_data = pd.DataFrame({
                    "Stage": ["Total Base Pool", "1+ Higher Spec.", "2+ Multi-Spec."], 
                    "Count": [total_doctors, (multi_qual_doctors > 1).sum(), (multi_qual_doctors > 2).sum()]
                })
                render_premium_chart(px.funnel, funnel_data, x="Count", y="Stage", title="3. Practitioner Structural Depth Funnel", color_discrete_sequence=["#0D3B2A", "#2A9D8F", "#C9A84C"])
            with t2_r2_c2:
                gap_df = filtered[filtered["QualSeq"].isin([1, 2])].pivot_table(index="RegistrationNo", columns="QualSeq", values="PassingYear", aggfunc="first").dropna()
                gap_df.columns = ["Primary", "Postgrad"]
                render_premium_chart(px.scatter, gap_df, x="Primary", y="Postgrad", title="4. Career Horizon Timeline Correlation", opacity=0.4, color_discrete_sequence=["#E8C547"])

            t2_r3_c1, t2_r3_c2 = st.columns(2)
            with t2_r3_c1:
                gap_df["Years_To_Spec"] = gap_df["Postgrad"] - gap_df["Primary"]
                gap_df = gap_df[gap_df["Years_To_Spec"].between(0, 30)]
                render_premium_chart(px.histogram, gap_df, x="Years_To_Spec", nbins=15, title="5. Specialization Lag Window Distribution", color_discrete_sequence=["#457B9D"])
            with t2_r3_c2:
                # Full (non-truncated) counts, used by both the Graph and Table views.
                sec_univ_full = exclude_unknown(secondary, "University")["University"].value_counts().reset_index()
                sec_univ_full.columns = ["Institution", "Volume"]

                view_mode_qa2 = st.radio("View", ["📊 Graph", "📋 Table"], horizontal=True, key="qa_fellowship_view", label_visibility="collapsed")

                if view_mode_qa2 == "📊 Graph":
                    sec_univ = sec_univ_full.copy()
                    if len(sec_univ) > 10:
                        top_part = sec_univ.head(10)
                        others_sum = sec_univ.iloc[10:]["Volume"].sum()
                        sec_univ = pd.concat([top_part, pd.DataFrame([{"Institution": "Others", "Volume": others_sum}])], ignore_index=True)
                    sec_univ = sec_univ.sort_values("Volume", ascending=True)
                    # % is of ALL secondary rows with a known university (sums
                    # to the same total as sec_univ_full).
                    sec_univ["Pct"] = (sec_univ["Volume"] / sec_univ["Volume"].sum() * 100).round(1)
                    render_premium_chart(px.bar, sec_univ, x="Volume", y="Institution", orientation="h", title="6. Top 10 Fellowship & Postgrad Training Centers", color_discrete_sequence=["#2A9D8F"], text=sec_univ["Pct"].map(lambda p: f"{p}%"))
                else:
                    st.dataframe(sec_univ_full.sort_values("Volume", ascending=False), use_container_width=True, hide_index=True, height=340)

            t2_r4_c1, t2_r4_c2 = st.columns(2)
            with t2_r4_c1:
                sec_yr = secondary.dropna(subset=["PassingYear"]).groupby("PassingYear").size().reset_index(name="Volume")
                render_premium_chart(px.line, sec_yr, x="PassingYear", y="Volume", title="7. Historical Postgrad Inflow Accelerations", markers=True, color_discrete_sequence=["#2A9D8F"])
            with t2_r4_c2:
                # Grouping by raw "Status" text produced 100+ near-duplicate
                # legend entries (see STATUS_BUCKET_MAP note above) - the
                # chart was unreadable, bars and legend overlapping. Fixed by
                # bucketing into a handful of clean categories first.
                spec_status_src = secondary.copy()
                spec_status_src["Status_Bucketed"] = spec_status_src["Status"].apply(bucket_status)
                spec_status_src = spec_status_src[spec_status_src["Status_Bucketed"] != "Other"]
                spec_status = spec_status_src.groupby(["Degree", "Status_Bucketed"]).size().reset_index(name="Count")
                spec_status = spec_status[spec_status["Degree"].isin(secondary["Degree"].value_counts().head(4).index)]
                # % of all secondary (postgrad) rows with a recognized status.
                spec_status["Pct"] = (spec_status["Count"] / len(spec_status_src) * 100).round(1)
                render_premium_chart(px.bar, spec_status, x="Degree", y="Count", color="Status_Bucketed", barmode="group", title="8. Postgrad License Compliance Grid", color_discrete_map={"Active": "#2A9D8F", "In-Active": "#64748B", "Suspended": "#C9A84C", "Cancelled": "#EF4444"}, text=spec_status["Pct"].map(lambda p: f"{p}%"))

            t2_r5_c1, t2_r5_c2 = st.columns(2)
            with t2_r5_c1:
                render_premium_chart(px.box, gap_df, y="Years_To_Spec", title="9. Structural Transition Outliers Map", color_discrete_sequence=["#EF4444"])
            with t2_r5_c2:
                # Same case-normalization as the Tab 1 "Operational Scope Mix" chart above.
                track_spec_src = secondary.copy()
                track_spec_src["RegistrationType"] = track_spec_src["RegistrationType"].astype(str).str.strip().str.title()
                track_spec = track_spec_src.groupby(["source_label", "RegistrationType"]).size().reset_index(name="Count")
                # % of all secondary (postgrad) rows.
                track_spec["Pct"] = (track_spec["Count"] / len(track_spec_src) * 100).round(1)
                render_premium_chart(px.bar, track_spec, x="source_label", y="Count", color="RegistrationType", title="10. Specialized Density Vectors Across Series", color_discrete_sequence=["#C9A84C", "#457B9D"], text=track_spec["Pct"].map(lambda p: f"{p}%"))

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3: DEMOGRAPHICS INFERRED
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<p class="caveat">⚠️ Gender is inferred from the doctor\'s first name only (heuristic, may be Unknown for names outside our reference lists). Province is ground-truth for AJK (from the AJK-series registry) and inferred from university for everyone else. Treat both as approximated where noted.</p>', unsafe_allow_html=True)
        
        t3_r1_c1, t3_r1_c2 = st.columns(2)
        with t3_r1_c1:
            gen_counts = exclude_unknown(primary, "Gender")["Gender"].value_counts().reset_index()
            gen_counts.columns = ["Gender", "Count"]
            render_premium_chart(px.pie, gen_counts, names="Gender", values="Count", hole=0.5, title="1. Inferred Operational Gender Ratios", color_discrete_sequence=["#C9A84C", "#2A9D8F", "#64748B"])
        with t3_r1_c2:
            primary["Specialized_Flag"] = primary["RegistrationNo"].isin(secondary["RegistrationNo"]).map({True: "Specialist", False: "Generalist"})
            primary_known = primary[primary["Province"] != "Unknown"]
            render_premium_chart(px.sunburst, primary_known, path=["Province", "Specialized_Flag"], title="2. Regional Sourcing vs Specialization Architecture", color_discrete_sequence=COLOR_SEQ)

        t3_r2_c1, t3_r2_c2 = st.columns(2)
        with t3_r2_c1:
            top_degs = exclude_unknown(primary, "Degree")["Degree"].value_counts().head(6).index
            box_df = primary[primary["Degree"].isin(top_degs)].dropna(subset=["PassingYear"])
            render_premium_chart(px.box, box_df, x="Degree", y="PassingYear", color="Degree", title="3. Cohort Longevity Variations by Degree", showlegend=False)
        with t3_r2_c2:
            # Uses the shared bucket_status() defined near the top of this
            # file (fixed to check "in-active"/"inactive" before "active",
            # and shared here to avoid two copies of the same fix drifting
            # apart).
            primary_b = primary.copy()
            primary_b["Status_Bucketed"] = primary_b["Status"].apply(bucket_status)
            primary_b = primary_b[primary_b["Status_Bucketed"] != "Other"]
            stat_deg = primary_b.groupby(["Degree", "Status_Bucketed"]).size().reset_index(name="Volume")
            stat_deg = stat_deg[stat_deg["Degree"].isin(top_degs)]
            render_premium_chart(px.bar, stat_deg, x="Degree", y="Volume", color="Status_Bucketed", title="4. Audit Compliance Class per Degree", color_discrete_map={"Active": "#2A9D8F", "In-Active": "#64748B", "Suspended": "#C9A84C", "Cancelled": "#EF4444"})

        t3_r3_c1, t3_r3_c2 = st.columns(2)
        with t3_r3_c1:
            time_spec = primary.dropna(subset=["PassingYear"]).copy()
            time_spec["Specialized_Flag"] = time_spec["RegistrationNo"].isin(secondary["RegistrationNo"])
            time_spec = time_spec.groupby("PassingYear").agg(Total=("RegistrationNo", "count"), Specialists=("Specialized_Flag", "sum")).reset_index()
            time_spec["Spec_Rate"] = (time_spec["Specialists"] / time_spec["Total"]) * 100
            render_premium_chart(px.line, time_spec, x="PassingYear", y="Spec_Rate", title="5. Historical Specialization Progression (%)", markers=True, color_discrete_sequence=["#C9A84C"])
        with t3_r3_c2:
            prov_vol = primary_known["Province"].value_counts().reset_index()
            prov_vol.columns = ["Province", "Count"]
            render_premium_chart(px.bar, prov_vol, x="Count", y="Province", orientation="h", title="6. Inferred Territorial Base Supply Pool", color_discrete_sequence=["#E8C547"])

        t3_r4_c1, t3_r4_c2 = st.columns(2)
        with t3_r4_c1:
            gen_time = exclude_unknown(primary.dropna(subset=["PassingYear"]), "Gender").groupby(["PassingYear", "Gender"]).size().reset_index(name="Volume")
            render_premium_chart(px.line, gen_time, x="PassingYear", y="Volume", color="Gender", title="7. Gender Inflow Trajectory Paradigm Shift", color_discrete_sequence=["#C9A84C", "#2A9D8F"])
        with t3_r4_c2:
            prov_gen = exclude_unknown(primary_known, "Gender").groupby(["Province", "Gender"]).size().reset_index(name="Volume")
            render_premium_chart(px.bar, prov_gen, x="Province", y="Volume", color="Gender", barmode="group", title="8. Gender Distribution Across Provinces", color_discrete_sequence=["#2A9D8F", "#E8C547"])

        t3_r5_c1, t3_r5_c2 = st.columns(2)
        with t3_r5_c1:
            spec_gen = exclude_unknown(secondary, "Gender").groupby(["Gender", "Speciality"]).size().reset_index(name="Count")
            spec_gen = spec_gen[spec_gen["Speciality"].isin(secondary["Speciality"].value_counts().head(4).index)]
            render_premium_chart(px.bar, spec_gen, x="Speciality", y="Count", color="Gender", barmode="stack", title="9. Gender Dispersion in Postgrad Domains", color_discrete_sequence=["#457B9D", "#C9A84C"])
        with t3_r5_c2:
            prov_src = primary_known.groupby(["Province", "source_label"]).size().reset_index(name="Count")
            render_premium_chart(px.bar, prov_src, x="Province", y="Count", color="source_label", barmode="stack", title="10. Series Dominance Grid Across Geography", color_discrete_sequence=COLOR_SEQ)


if __name__ == "__main__":
    render_qualification_analytics()