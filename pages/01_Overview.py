import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

from utils.gender_province import (
    SOURCE_LABEL_MAP,
    infer_province,
    infer_province_series,
)

# Keyword list to split universities into Local vs International (same heuristic
# used on the University Insights page, kept in sync for consistency).
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
    if not isinstance(university, str) or not university.strip() or university == "N/A":
        return "Unknown"
    u = university.lower()
    for kw in FOREIGN_UNIVERSITY_KEYWORDS:
        if kw in u:
            return "International"
    return "Local"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
#  LOADING & FALLBACK
# ============================================================================
@st.cache_data(ttl=3600)
def load_overview_data(use_real_db: bool):
    if not use_real_db:
        return pd.DataFrame()
    try:
        from utils.data_loader import load_all_data
        df = load_all_data()
        if df is not None and not df.empty:
            return df
        logger.warning("DB returned empty dataframe.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"DB load failed: {e}")
        return pd.DataFrame()

# ============================================================================
# AGGRESSIVE CLEANING & OVERLAP PROTECTION
# ============================================================================
def safe_truncate(val, max_len=22):
    if pd.isna(val) or str(val).strip() == "": return "N/A"
    s = str(val).strip()
    return s[:max_len] + "..." if len(s) > max_len else s

def clean_status(val):
    if pd.isna(val): return "Unknown"
    return str(val).strip().title()

def normalize_degree_name(degree):
    if pd.isna(degree) or str(degree).strip() in ["", "N/A"]:
        return "N/A"
    cleaned = str(degree).strip().upper().replace(" ", "").replace(".", "")
    if "MBBS" in cleaned: return "MBBS"
    if "BDS" in cleaned: return "BDS"
    if "FCPS" in cleaned: return "FCPS"
    if "MD" in cleaned: return "MD"
    if "MS" in cleaned: return "MS"
    return str(degree).strip()

@st.cache_data(ttl=3600)
def load_and_normalize_overview(use_real_db: bool):
    # NOTE: _normalize_data() below does real work (parsing dates, inferring
    # Province, truncating strings, etc.) over every row. Previously it ran
    # UNCACHED on every single script rerun - meaning every filter/slider
    # change re-did this full pass over 200k+ rows before the filters even
    # got applied. Caching it here, keyed only on the simple use_real_db
    # bool, means it only actually runs once per hour (ttl) instead of on
    # every interaction.
    raw_df = load_overview_data(use_real_db=use_real_db)
    if raw_df.empty:
        return raw_df
    return _normalize_data(raw_df.copy())


def _normalize_data(df):
    fallback_cols = {
        "Status": "Unknown",
        "RegistrationType": "Unspecified",
        "RegistrationDate": None,
        "ValidUpto": None,
        "source_table": "doctors"
    }
    for col, default in fallback_cols.items():
        if col not in df.columns:
            df[col] = default

    df["Status_Clean"] = df["Status"].apply(clean_status)
    df["source_label"] = df["source_table"].map(SOURCE_LABEL_MAP).fillna(df["source_table"])
    
    if "Qualifications" in df.columns:
        def extract_field(qs, field):
            try:
                if isinstance(qs, list) and len(qs) > 0:
                    return qs[0].get(field) or "N/A"
                return "N/A"
            except Exception:
                return "N/A"
        df["Primary_Degree"] = df["Qualifications"].apply(lambda x: extract_field(x, "Degree"))
        df["Primary_University"] = df["Qualifications"].apply(lambda x: extract_field(x, "University"))
        df["Primary_Speciality"] = df["Qualifications"].apply(lambda x: extract_field(x, "Speciality"))
    else:
        df["Primary_Degree"] = df["Qualification_1_Degree"].fillna("N/A") if "Qualification_1_Degree" in df.columns else "N/A"
        df["Primary_University"] = df["Qualification_1_University"].fillna("N/A") if "Qualification_1_University" in df.columns else "N/A"
        df["Primary_Speciality"] = df["Qualification_1_Speciality"].fillna("General Practice") if "Qualification_1_Speciality" in df.columns else "General Practice"

    df["Location_Type"] = df["Primary_University"].apply(is_foreign_university)

    # Province: use the existing column if the real data provides one, otherwise
    # (or wherever it's blank) infer it from the university name — AJK-series
    # records get Province="AJK" with full confidence (ground truth from the
    # collection they came from), everything else is a university-name guess.
    inferred_province = infer_province_series(df["Primary_University"], df.get("source_table"))
    if "Province" in df.columns:
        df["Province"] = df["Province"].fillna(inferred_province)
        df.loc[df["Province"].astype(str).str.strip().isin(["", "N/A", "Unknown", "nan"]), "Province"] = inferred_province
    else:
        df["Province"] = inferred_province

    # City: keep existing column if present, else default so the slicer never disappears.
    if "City" not in df.columns:
        df["City"] = "Unknown"
    df["City"] = df["City"].fillna("Unknown")

    df["Primary_Degree"] = df["Primary_Degree"].apply(normalize_degree_name).apply(lambda x: safe_truncate(x, 12))
    df["Primary_University"] = df["Primary_University"].fillna("N/A")
    df["Primary_Speciality"] = df["Primary_Speciality"].fillna("N/A")
    
    return df

# ============================================================================
# CROSS-THEME VISIBILITY CSS
# ============================================================================
def inject_custom_css():
    st.markdown("""
        <style>
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid rgba(184, 134, 11, 0.35);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .kpi-title {
            color: #6B7280 !important;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }
        .kpi-value {
            color: #B8860B;
            font-size: 2.1rem;
            font-weight: 700;
            margin: 0;
            font-family: 'Cinzel', serif;
        }
        .section-header {
            color: #9A6B00;
            font-family: 'Cinzel', serif;
            border-bottom: 1px solid rgba(184, 134, 11, 0.35);
            padding-bottom: 6px;
            margin-top: 2.5rem;
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
        }
        .insight-text {
            color: #6B7280 !important;
            font-size: 0.85rem;
            margin-top: 2px;
            margin-bottom: 20px;
            font-style: italic;
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN PAGE LOGIC
# ============================================================================
def render_overview():
    inject_custom_css()
    
    st.markdown("<h1 style='color: #B8860B; font-family: Cinzel, serif;'>Executive Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4B5563; margin-bottom: 2rem;'>National Medical Registry at a Glance — Simple metrics for high-level decision making.</p>", unsafe_allow_html=True)
    
    use_real_db = st.session_state.get("db_connected", False)
    df = load_and_normalize_overview(use_real_db=use_real_db)

    if df.empty:
        st.warning("⚠️ No database connection detected. Please check your DB connection to view live registry data.")
        return

    today = pd.to_datetime(datetime.utcnow().date())
    df["RegistrationDate_dt"] = pd.to_datetime(df["RegistrationDate"], errors="coerce")
    df["ValidUpto_dt"] = pd.to_datetime(df["ValidUpto"], errors="coerce")
    df["IsExpired"] = df["ValidUpto_dt"].lt(today) | (df["Status_Clean"].str.lower().str.contains("exp", na=False))
    df["DaysToExpiry"] = (df["ValidUpto_dt"] - today).dt.days
    df["Year"] = df["RegistrationDate_dt"].dt.year
    
    source_opts = sorted(df["source_label"].dropna().unique())
    table_filter = st.sidebar.selectbox("Filter Data Source (Series)", ["All Series"] + source_opts)
    if table_filter != "All Series":
        df = df[df["source_label"] == table_filter]

    # --- INDEPENDENT SLICERS: each filter below is separate and applies on its own ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("<p style='color:#9A6B00; font-weight:600; margin-bottom:0.3rem;'>🔎 Drill Down Filters</p>", unsafe_allow_html=True)

    # Country / Location Type slicer (Local vs International) - independent
    country_opts = sorted(df["Location_Type"].dropna().unique())
    country_filter = st.sidebar.selectbox("Country (Local / International)", ["All"] + country_opts)
    if country_filter != "All":
        df = df[df["Location_Type"] == country_filter]

    # Province slicer - independent
    if "Province" in df.columns:
        prov_opts = sorted(df["Province"].dropna().unique())
        prov_filter = st.sidebar.selectbox("Province", ["All Provinces"] + prov_opts)
        if prov_filter != "All Provinces":
            df = df[df["Province"] == prov_filter]

    # District / City slicer - independent
    if "City" in df.columns:
        city_opts = sorted(df["City"].dropna().unique())
        city_filter = st.sidebar.selectbox("District / City", ["All Districts"] + city_opts)
        if city_filter != "All Districts":
            df = df[df["City"] == city_filter]

    # Local University slicer - independent, only lists Local universities
    local_univ_opts = sorted(df[df["Location_Type"] == "Local"]["Primary_University"].dropna().unique())
    local_univ_filter = st.sidebar.selectbox("Local University", ["All Local Universities"] + local_univ_opts)
    if local_univ_filter != "All Local Universities":
        df = df[df["Primary_University"] == local_univ_filter]

    # International University slicer - independent, only lists International universities
    intl_univ_opts = sorted(df[df["Location_Type"] == "International"]["Primary_University"].dropna().unique())
    intl_univ_filter = st.sidebar.selectbox("International University", ["All International Universities"] + intl_univ_opts)
    if intl_univ_filter != "All International Universities":
        df = df[df["Primary_University"] == intl_univ_filter]

    # Registration Year slicer
    if "Year" in df.columns:
        valid_years = df["Year"].dropna()
        if not valid_years.empty:
            year_range = st.sidebar.slider(
                "Registration Year",
                int(valid_years.min()),
                2018,
                (int(valid_years.min()), 2018)
            )
            df = df[df["Year"].between(year_range[0], year_range[1]) | df["Year"].isna()]

    if df.empty:
        st.warning("⚠️ No records match the current filter selection.")
        return

    # GLOBAL HIGH-CONTRAST THEME CONFIG
    layout_theme = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#374151", family="Inter", size=11), 
        title_font=dict(size=14, color="#9A6B00", family="Cinzel"),
        margin=dict(t=50, b=50, l=60, r=20),
        height=380, # Lock height explicitly so it never collapses
        legend=dict(
            font=dict(color="#374151", size=10),
            orientation="h", 
            yanchor="top", 
            y=-0.15, 
            xanchor="center", 
            x=0.5
        )
    )

    # ─── CORE COUNTERS (KPIs 1-4) ───
    total_docs = len(df)
    active_docs = int((df["Status_Clean"] == "Active").sum())
    expired_licenses = int(df["IsExpired"].sum())
    top_degree = df["Primary_Degree"].mode()[0] if not df["Primary_Degree"].empty else "N/A"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Registered</div><div class="kpi-value">{total_docs:,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Active Doctors</div><div class="kpi-value" style="color: #2A9D8F;">{active_docs:,}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Expired Licenses</div><div class="kpi-value" style="color: #EF4444;">{expired_licenses:,}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Main Stream Degree</div><div class="kpi-value" style="font-size: 1.6rem; line-height: 2.8rem;">{top_degree}</div></div>', unsafe_allow_html=True)

    # ─── SECTION 1: WORKFORCE SPLIT & SERIES (Visuals 1, 2 & 3) ───
    st.markdown("<div class='section-header'>1. Workforce Composition & License Status</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        # Secure top 4 + Others grouping to avoid infinite legend crash
        t_counts = df["RegistrationType"].fillna("Unspecified").value_counts().reset_index()
        t_counts.columns = ["Type", "Count"]
        if len(t_counts) > 4:
            top_part = t_counts.head(4)
            others_sum = t_counts.iloc[4:]["Count"].sum()
            t_counts = pd.concat([top_part, pd.DataFrame([{"Type": "Others", "Count": others_sum}])], ignore_index=True)
            
        fig1 = px.pie(t_counts, values="Count", names="Type", hole=0.5, title="Medical vs Dental Share", 
                      color_discrete_sequence=["#C9A84C", "#2A9D8F", "#457B9D", "#E8C547", "#718096"])
        fig1.update_layout(**layout_theme)
        fig1.update_traces(textinfo="percent+label", textposition="inside")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("<p class='insight-text'>Proportion of General Medical practitioners versus Dental specialists.</p>", unsafe_allow_html=True)
        
    with c2:
        # Secure top 4 + Others grouping for Status
        s_counts = df["Status_Clean"].value_counts().reset_index()
        s_counts.columns = ["Status", "Count"]
        if len(s_counts) > 4:
            top_part = s_counts.head(4)
            others_sum = s_counts.iloc[4:]["Count"].sum()
            s_counts = pd.concat([top_part, pd.DataFrame([{"Status": "Others", "Count": others_sum}])], ignore_index=True)

        fig2 = px.pie(s_counts, values="Count", names="Status", hole=0.5, title="Current Status Distribution", 
                      color_discrete_sequence=["#2A9D8F", "#EF4444", "#E8C547", "#457B9D", "#718096"])
        fig2.update_layout(**layout_theme)
        fig2.update_traces(textinfo="percent+label", textposition="inside")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<p class='insight-text'>Operational health check: Percentage of records currently active.</p>", unsafe_allow_html=True)

    # Visual 3
    series_counts = df["source_label"].value_counts().reset_index()
    series_counts.columns = ["Series", "Count"]
    fig3 = px.bar(series_counts.head(6), x="Count", y="Series", orientation="h", title="Registration Volume by Series", 
                  color_discrete_sequence=["#C9A84C"])
    layout_series = layout_theme.copy()
    layout_series["margin"] = dict(t=50, b=40, l=120, r=20)
    fig3.update_layout(**layout_series)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("<p class='insight-text'>Classification system load across PMDC registry series maps.</p>", unsafe_allow_html=True)


    # ─── SECTION 2: INSTITUTIONAL DEPTH (Visuals 4, 5 & 6) ───
    st.markdown("<div class='section-header'>2. Institutional Distribution & Specialities</div>", unsafe_allow_html=True)
    
    c3, c4 = st.columns(2)
    with c3:
        # Full (non-truncated) counts, used by both the Graph and Table views.
        local_df = df[(df["Primary_University"] != "N/A") & (df["Location_Type"] == "Local")]
        univ_counts_full = local_df["Primary_University"].value_counts().reset_index()
        univ_counts_full.columns = ["University", "Count"]

        view_mode_local = st.radio("View", ["📊 Graph", "📋 Table"], horizontal=True, key="local_univ_view", label_visibility="collapsed")

        if view_mode_local == "📊 Graph":
            # Top 10 LOCAL universities only, with an "Others" bucket for the rest.
            univ_counts = univ_counts_full.copy()
            if len(univ_counts) > 10:
                top_part = univ_counts.head(10)
                others_sum = univ_counts.iloc[10:]["Count"].sum()
                univ_counts = pd.concat([top_part, pd.DataFrame([{"University": "Others", "Count": others_sum}])], ignore_index=True)
            layout_univ = layout_theme.copy()
            layout_univ["margin"] = dict(t=50, b=40, l=180, r=20)
            fig4 = px.bar(univ_counts, x="Count", y="University", orientation="h", title="Top 10 Local Universities",
                          color_discrete_sequence=["#2A9D8F"])
            fig4.update_layout(**layout_univ)
            fig4.update_yaxes(autorange="reversed")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<p class='insight-text'>Top 10 local institutions — switch to Table above to see every local university.</p>", unsafe_allow_html=True)
        else:
            st.dataframe(univ_counts_full, use_container_width=True, hide_index=True, height=380)
            st.markdown("<p class='insight-text'>Every local university in the current filter selection, sorted by doctor count.</p>", unsafe_allow_html=True)

    with c4:
        # Full (non-truncated) counts, used by both the Graph and Table views.
        intl_df = df[(df["Primary_University"] != "N/A") & (df["Location_Type"] == "International")]
        intl_counts_full = intl_df["Primary_University"].value_counts().reset_index()
        intl_counts_full.columns = ["University", "Count"]

        if intl_counts_full.empty:
            st.info("No international university records in the current filter selection.")
        else:
            view_mode_intl = st.radio("View", ["📊 Graph", "📋 Table"], horizontal=True, key="intl_univ_view", label_visibility="collapsed")

            if view_mode_intl == "📊 Graph":
                # Top 10 INTERNATIONAL universities only, with an "Others" bucket for the rest.
                intl_counts = intl_counts_full.copy()
                if len(intl_counts) > 10:
                    top_part = intl_counts.head(10)
                    others_sum = intl_counts.iloc[10:]["Count"].sum()
                    intl_counts = pd.concat([top_part, pd.DataFrame([{"University": "Others", "Count": others_sum}])], ignore_index=True)
                layout_intl = layout_theme.copy()
                layout_intl["margin"] = dict(t=50, b=40, l=180, r=20)
                fig4b = px.bar(intl_counts, x="Count", y="University", orientation="h", title="Top 10 International Universities",
                              color_discrete_sequence=["#E8C547"])
                fig4b.update_layout(**layout_intl)
                fig4b.update_yaxes(autorange="reversed")
                st.plotly_chart(fig4b, use_container_width=True)
                st.markdown("<p class='insight-text'>Top 10 international institutions — switch to Table above to see every international university.</p>", unsafe_allow_html=True)
            else:
                st.dataframe(intl_counts_full, use_container_width=True, hide_index=True, height=380)
                st.markdown("<p class='insight-text'>Every international university in the current filter selection, sorted by doctor count.</p>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        deg_counts = df[df["Primary_Degree"] != "N/A"]["Primary_Degree"].value_counts().reset_index().head(5)
        deg_counts.columns = ["Degree", "Count"]
        fig5 = px.bar(deg_counts, x="Degree", y="Count", title="Dominant Qualifications", color_discrete_sequence=["#E8C547"])
        fig5.update_layout(**layout_theme)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("<p class='insight-text'>Type of medical or dental degrees most commonly held in the database.</p>", unsafe_allow_html=True)

    with c6:
        spec_counts = df["Primary_Speciality"].value_counts().reset_index()
        spec_counts.columns = ["Speciality", "Count"]
        if len(spec_counts) > 5:
            top_part = spec_counts.head(5)
            others_sum = spec_counts.iloc[5:]["Count"].sum()
            spec_counts = pd.concat([top_part, pd.DataFrame([{"Speciality": "Others", "Count": others_sum}])], ignore_index=True)

        fig6 = px.pie(spec_counts, values="Count", names="Speciality", title="Top Medical Fields / Specialities Breakdown", 
                      color_discrete_sequence=["#C9A84C", "#2A9D8F", "#E8C547", "#457B9D", "#718096", "#EF4444"])
        fig6.update_layout(**layout_theme)
        fig6.update_traces(textinfo="percent+label", textposition="inside")
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("<p class='insight-text'>Breakdown of the top active medical areas or general setups across the registry database.</p>", unsafe_allow_html=True)


    # ─── SECTION 3: HISTORICAL GROWTH TIMELINES (Visuals 7, 8 & 9) ───
    st.markdown("<div class='section-header'>3. Historical Growth & Active Ratios</div>", unsafe_allow_html=True)
    
    c5, c6 = st.columns(2)
    reg_year = df.dropna(subset=["Year"]).copy()
    reg_year = reg_year[(reg_year["Year"] >= 1990) & (reg_year["Year"] <= datetime.now().year)]
    
    with c5:
        reg_by_year = reg_year.groupby("Year").size().reset_index(name="Registrations")
        fig7 = px.line(reg_by_year, x="Year", y="Registrations", title="Yearly Registration Velocity", 
                       color_discrete_sequence=["#C9A84C"], markers=True)
        fig7.update_layout(**layout_theme)
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("<p class='insight-text'>Timeline trajectory showing how fast new doctors register year after year.</p>", unsafe_allow_html=True)

    with c6:
        reg_year["IsActive"] = reg_year["Status_Clean"] == "Active"
        active_rate = reg_year.groupby("Year")["IsActive"].mean() * 100
        active_rate = active_rate.reset_index(name="Active_Rate")
        fig8 = px.bar(active_rate, x="Year", y="Active_Rate", title="Retention Rate by Registration Year (%)", 
                      color_discrete_sequence=["#2A9D8F"])
        fig8.update_layout(**layout_theme)
        fig8.update_yaxes(range=[0, 100])
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("<p class='insight-text'>Percentage of doctors registered in a specific year who still remain active.</p>", unsafe_allow_html=True)


    # ─── SECTION 4: CRITICAL OPERATIONS (Visual 9 & 10) ───
    st.markdown("<div class='section-header'>4. Critical Actions & Expiring Warnings</div>", unsafe_allow_html=True)
    
    c7, c8 = st.columns([1, 1.2])
    with c7:
        yoy = reg_year.groupby("Year").size().reset_index(name="Count")
        yoy["Growth_Rate"] = yoy["Count"].pct_change() * 100
        yoy = yoy.dropna().tail(15)
        fig9 = px.bar(yoy, x="Year", y="Growth_Rate", title="Year-over-Year Growth Rate (%)", color_discrete_sequence=["#EF4444"])
        fig9.add_hline(y=0, line_color="rgba(148,163,184,0.4)")
        fig9.update_layout(**layout_theme)
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown("<p class='insight-text'>Year-on-year percentage shift in registrations.</p>", unsafe_allow_html=True)

    with c8:
        st.markdown("<p style='color:#9A6B00; font-weight:600; font-size:1rem; margin-bottom:5px;'>⚠️ License Expiry Monitoring (Next 90 Days)</p>", unsafe_allow_html=True)
        
        if "DaysToExpiry" in df.columns and not df.empty:
            urgent = df[(df["DaysToExpiry"] >= 0) & (df["DaysToExpiry"] <= 90)].sort_values("DaysToExpiry").head(15)
        else:
            urgent = pd.DataFrame()
        
        if not urgent.empty:
            display_cols = [col for col in ["Name", "Primary_Degree", "ValidUpto", "DaysToExpiry"] if col in urgent.columns]
            st.dataframe(
                urgent[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Name": "Doctor Name",
                    "Primary_Degree": "Degree",
                    "ValidUpto": "Expiry Date",
                    "DaysToExpiry": st.column_config.ProgressColumn(
                        "Days Left",
                        min_value=0, max_value=90, format="%d days"
                    )
                }
            )
            st.markdown("<p class='insight-text' style='margin-top:5px;'>List of practitioners requiring immediate notification.</p>", unsafe_allow_html=True)
        else:
            st.success("✅ Excellent! All checked data rows show valid licenses yards beyond 90 days.")

# ============================================================================
# EXECUTION TRIGGER
# ============================================================================
if __name__ == "__main__":
    render_overview()
