import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback

from utils.gender_province import (
    infer_gender_series,
    infer_province_series,
)

# ============================================================================
# ULTRA-STABLE FIXED TRANSPARENT THEME WRAPPER
# ============================================================================
def render_premium_chart(chart_func, *args, chart_height=340, **kwargs):
    """Executes a chart and strictly injects 100% borderless transparency over dark theme.
    chart_height can be overridden for charts with many categories (e.g. full district/
    university lists) so labels never overlap or get squeezed."""
    try:
        fig = chart_func(*args, **kwargs)
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#374151", family="Inter", size=11),
            title_font=dict(size=13, color="#9A6B00", family="Cinzel"),
            margin=dict(t=50, b=40, l=50, r=20),
            height=chart_height,
            showlegend=True,
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#374151", size=10)
            )
        )
        fig.update_xaxes(showgrid=False, zeroline=False, title_font=dict(color="#374151", size=10))
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)", zeroline=False, title_font=dict(color="#374151", size=10))
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        st.warning(f"⚠️ Render skip on this element.")

def safe_truncate(val, max_len=18):
    if pd.isna(val) or str(val).strip() == "" or str(val).lower() in ["nan", "none", "null", "<na>"]: 
        return "Unknown"
    s = str(val).strip()
    return s[:max_len] + "..." if len(s) > max_len else s

def exclude_unknown(data, *cols):
    """Drop rows where any of the given columns is 'Unknown', so charts never
    show an 'Unknown' bar/slice/line. Only rows lacking THAT chart's specific
    field(s) are excluded - other charts using different fields are unaffected."""
    mask = pd.Series(True, index=data.index)
    for c in cols:
        if c in data.columns:
            mask &= (data[c].astype(str).str.strip() != "Unknown")
    return data[mask]

# Keyword list to split universities into Local vs International (same heuristic
# used on the Overview and University Insights pages, kept in sync).
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

def is_foreign_university(university):
    if not isinstance(university, str) or not university.strip() or university == "Unknown":
        return "Unknown"
    u = university.lower()
    for kw in FOREIGN_UNIVERSITY_KEYWORDS:
        if kw in u:
            return "International"
    return "Local"


@st.cache_data(ttl=3600)
def load_and_normalize_doctor_analytics():
    # NOTE: _normalize_analytics_data() below does real work (parsing dates,
    # inferring Gender/Province, extracting nested Qualification fields,
    # etc.) over every row. Previously it ran UNCACHED on every single script
    # rerun - meaning every filter/slider change re-did this full pass over
    # 200k+ rows before the filters even got applied. Caching it here means
    # it only actually runs once per hour (ttl) instead of on every click.
    from utils.data_loader import load_all_data
    raw_df = load_all_data()
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    return _normalize_analytics_data(raw_df.copy())


def _normalize_analytics_data(df):
    # The real DB stores qualifications as a nested list under "Qualifications"
    # (each entry a dict with Degree/University/Speciality/PassingYear keys) -
    # same shape used on the Qualification Analytics / University Insights pages.
    # Extract the primary (first) qualification into flat columns here so the
    # rest of this page's logic (which expects flat Qualification_1_* columns)
    # actually finds real data instead of falling through to "Unknown".
    def extract_qual_field(quals, key):
        if isinstance(quals, list) and len(quals) > 0 and isinstance(quals[0], dict):
            val = quals[0].get(key)
            return val if val not in (None, "") else None
        return None

    if "Qualifications" in df.columns:
        if "Qualification_1_Degree" not in df.columns:
            df["Qualification_1_Degree"] = df["Qualifications"].apply(lambda q: extract_qual_field(q, "Degree"))
        if "Qualification_1_University" not in df.columns:
            df["Qualification_1_University"] = df["Qualifications"].apply(lambda q: extract_qual_field(q, "University"))
        if "Qualification_1_Speciality" not in df.columns:
            df["Qualification_1_Speciality"] = df["Qualifications"].apply(lambda q: extract_qual_field(q, "Speciality"))

    col_mappings = {
        "RegistrationDate": ["RegistrationDate", "RegDate", "date", "IssueDate"],
        "Status": ["Status", "status", "RegistrationStatus"],
        "RegistrationType": ["RegistrationType", "type", "Type", "Category"],
        "Province": ["Province", "province", "Region"],
        "City": ["City", "city", "District", "district"],
        "Gender": ["Gender", "gender", "Sex"],
        "Qualification_1_Speciality": ["Qualification_1_Speciality", "Speciality", "specialization"],
        "Qualification_1_Degree": ["Qualification_1_Degree", "Degree", "degree"],
        "Qualification_1_University": ["Qualification_1_University", "University", "university"]
    }
    
    # Map matching aliases dynamically - this just recognizes differently-named
    # real columns, it never invents data.
    for standard_col, possible_cols in col_mappings.items():
        if standard_col not in df.columns:
            for col in possible_cols:
                if col in df.columns:
                    df[standard_col] = df[col]
                    break

    # Any field genuinely absent from the data is simply labeled "Unknown" -
    # no random/fabricated values are ever generated.
    for col_name in ["Province", "City", "Qualification_1_Degree", "Qualification_1_University",
                      "Qualification_1_Speciality", "RegistrationType", "Gender", "Status"]:
        if col_name not in df.columns:
            df[col_name] = "Unknown"
        df[col_name] = df[col_name].fillna("Unknown")

    # Province fallback: only fill in rows where Province is genuinely missing,
    # inferring from the university name (AJK-series records get Province="AJK"
    # with full confidence). Real Province values are never touched.
    inferred_province = infer_province_series(df["Qualification_1_University"], df.get("source_table"))
    df.loc[df["Province"].astype(str).str.strip().isin(["", "Unknown", "N/A", "nan"]), "Province"] = inferred_province

    # Clean compliance flags
    def clean_pmdc_status(val):
        raw = str(val).upper()
        if any(x in raw for x in ["CANCEL", "REMOVED", "DE-REG"]): return "Cancelled"
        if any(x in raw for x in ["SUSPEND", "HOLD", "BLOCK"]): return "Suspended"
        if "EXPIRE" in raw: return "Expired"
        if "PROVISIONAL" in raw: return "Provisional"
        if "ACTIVE" in raw: return "Active"
        return "Unknown"

    df["Status_Clean"] = df["Status"].apply(clean_pmdc_status)
    df["Gender_Clean"] = df["Gender"].astype(str).str.strip().str.title().replace(["Nan", "None", ""], "Unknown")

    # Gender fallback: only for rows where the real Gender value is genuinely
    # missing, infer it from the doctor's Name using the shared first-name-only
    # matcher (avoids the old substring-match bug where family names like
    # "Arif"/"Khan"/"Ali" leaked into the wrong gender bucket). Real Gender
    # values already present in the data are never touched.
    if "Name" in df.columns:
        inferred_gender = infer_gender_series(df["Name"], df.get("Qualification_1_University"))
        unknown_mask = df["Gender_Clean"] == "Unknown"
        df.loc[unknown_mask, "Gender_Clean"] = inferred_gender[unknown_mask]

    # Registration Year - parsed only from the real RegistrationDate field.
    # Rows where it can't be parsed are left blank (NaN) rather than fabricated,
    # and are naturally excluded from year-based charts.
    if "RegistrationDate" in df.columns:
        extracted = df["RegistrationDate"].astype(str).str.extract(r'(19[5-9]\d|20[0-2]\d)')[0]
        df["RegYear"] = pd.to_numeric(extracted, errors="coerce")
    else:
        df["RegYear"] = float("nan")

    df["ExperienceYears"] = 2026 - df["RegYear"]

    # Final string cleaning for charts
    df["RegType_Clean"] = df["RegistrationType"].astype(str).str.title().apply(lambda x: safe_truncate(x, 15))
    df["Speciality_Clean"] = df["Qualification_1_Speciality"].astype(str).fillna("N/A")
    df["Degree_Clean"] = df["Qualification_1_Degree"].astype(str).fillna("N/A")
    df["Univ_Clean"] = df["Qualification_1_University"].astype(str).fillna("N/A")
    df["Province_Clean"] = df["Province"].astype(str).fillna("N/A")
    df["City_Clean"] = df["City"].astype(str).fillna("N/A")
    df["Location_Type"] = df["Qualification_1_University"].apply(is_foreign_university)

    return df

# ============================================================================
# MAIN CODESPACE
# ============================================================================
def render_doctor_analytics():
    try:
        # Premium Dashboard CSS Configurations
        st.markdown("""
            <style>
            .kpi-card { background: #FFFFFF; border: 1px solid rgba(184, 134, 11, 0.3); border-radius: 12px; padding: 22px; text-align: center; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
            .kpi-title { color: #6B7280 !important; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
            .kpi-value { color: #B8860B; font-size: 2.2rem; font-weight: 700; margin: 0; font-family: 'Cinzel', serif; }
            .stTabs [data-baseweb="tab-list"] { gap: 28px; background-color: transparent; }
            .stTabs [data-baseweb="tab"] { height: 48px; color: #6B7280 !important; font-weight: 600; background-color: transparent !important; }
            .stTabs [aria-selected="true"] { color: #9A6B00 !important; border-bottom: 2px solid #B8860B !important; }
            .insight-text { color: #6B7280 !important; font-size: 0.8rem; margin-top: -6px; margin-bottom: 24px; font-style: italic; }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='color: #B8860B; font-family: Cinzel, serif; margin-bottom: 1.5rem;'>Workforce Deep-Dive</h2>", unsafe_allow_html=True)
        
        df = load_and_normalize_doctor_analytics()

        if df is None or df.empty:
            st.warning("⚠️ No records parsed from active data pool.")
            return

        # --- INDEPENDENT SIDEBAR SLICERS: each filter is separate and applies on its own ---
        st.sidebar.markdown("<p style='color:#9A6B00; font-weight:600; margin-bottom:0.3rem;'>🔎 Drill Down Filters</p>", unsafe_allow_html=True)

        country_opts = sorted([c for c in df["Location_Type"].dropna().unique() if c != "Unknown"])
        country_filter = st.sidebar.selectbox("Country (Local / International)", ["All"] + country_opts)
        if country_filter != "All":
            df = df[df["Location_Type"] == country_filter]

        prov_opts = sorted([p for p in df["Province"].dropna().unique() if p != "Unknown"])
        prov_filter = st.sidebar.selectbox("Province", ["All Provinces"] + prov_opts)
        if prov_filter != "All Provinces":
            df = df[df["Province"] == prov_filter]

        city_opts = sorted([c for c in df["City"].dropna().unique() if c != "Unknown"])
        city_filter = st.sidebar.selectbox("District / City", ["All Districts"] + city_opts)
        if city_filter != "All Districts":
            df = df[df["City"] == city_filter]

        local_univ_opts = sorted(df[df["Location_Type"] == "Local"]["Qualification_1_University"].dropna().unique())
        local_univ_filter = st.sidebar.selectbox("Local University", ["All Local Universities"] + local_univ_opts)
        if local_univ_filter != "All Local Universities":
            df = df[df["Qualification_1_University"] == local_univ_filter]

        intl_univ_opts = sorted(df[df["Location_Type"] == "International"]["Qualification_1_University"].dropna().unique())
        intl_univ_filter = st.sidebar.selectbox("International University", ["All International Universities"] + intl_univ_opts)
        if intl_univ_filter != "All International Universities":
            df = df[df["Qualification_1_University"] == intl_univ_filter]

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
            st.warning("⚠️ No records match the current filter selection.")
            return
        
        # KPI calculations
        total_docs = len(df)
        active_count = int((df["Status_Clean"] == "Active").sum())
        active_rate = (active_count / total_docs * 100) if total_docs > 0 else 100.0
        avg_exp = df["ExperienceYears"].mean() if total_docs > 0 else 0.0
        
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Active Registry</div><div class="kpi-value">{total_docs:,}</div></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Verified Active Status</div><div class="kpi-value" style="color: #2A9D8F;">{active_rate:.1f}%</div></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Average Experience Model</div><div class="kpi-value">{avg_exp:.1f} Yrs</div></div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📊 Experience & Professional Spread", "📈 Educational & Regional Trends"])

        # ────────────────────────────────────────────────────────────────────────
        # SECTION 1: EXPERIENCE & PROFESSIONAL SPREAD (10 VISUALS)
        # ────────────────────────────────────────────────────────────────────────
        with tab1:
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                render_premium_chart(px.histogram, df, x="ExperienceYears", nbins=20, title="1. Career Seniority Spread (Yrs)", color_discrete_sequence=["#2A9D8F"])
                st.markdown("<p class='insight-text'>Density curve showing overall workforce practice longevity distribution.</p>", unsafe_allow_html=True)
            with r1c2:
                t_data = exclude_unknown(df, "RegType_Clean")["RegType_Clean"].value_counts().reset_index()
                t_data.columns = ["Track", "Count"]
                render_premium_chart(px.pie, t_data, values="Count", names="Track", hole=0.4, title="2. Professional Track Distribution", color_discrete_sequence=["#C9A84C", "#2A9D8F", "#E8C547"])
                st.markdown("<p class='insight-text'>Proportional breakdown separating specialized streams from medical systems.</p>", unsafe_allow_html=True)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                s_data = exclude_unknown(df, "Status_Clean")["Status_Clean"].value_counts().reset_index()
                s_data.columns = ["Status", "Count"]
                render_premium_chart(px.pie, s_data, values="Count", names="Status", hole=0.4, title="3. Legal Compliance Audit Distribution", color_discrete_sequence=["#2A9D8F", "#E8C547", "#EF4444"])
                st.markdown("<p class='insight-text'>Parsed audit categories clearing legal operational boundaries.</p>", unsafe_allow_html=True)
            with r2c2:
                box_df = exclude_unknown(df, "RegType_Clean")
                render_premium_chart(px.box, box_df, x="RegType_Clean", y="ExperienceYears", color="RegType_Clean", title="4. Experience Spread vs License Track", color_discrete_sequence=["#C9A84C", "#2A9D8F"])
                st.markdown("<p class='insight-text'>Quartile analysis isolating longevity trends between registration scopes.</p>", unsafe_allow_html=True)

            r3c1, r3c2 = st.columns(2)
            with r3c1:
                g_track = exclude_unknown(df, "RegType_Clean", "Gender_Clean").groupby(["RegType_Clean", "Gender_Clean"]).size().reset_index(name="Count")
                render_premium_chart(px.bar, g_track, x="RegType_Clean", y="Count", color="Gender_Clean", barmode="group", title="5. Gender Representation by License Model", color_discrete_sequence=["#2A9D8F", "#E8C547"])
                st.markdown("<p class='insight-text'>Gender distribution across administrative license parameters. <em>Gender inferred from first name where not explicitly provided — treat as approximate.</em></p>", unsafe_allow_html=True)
            with r3c2:
                sp_data = exclude_unknown(df, "Speciality_Clean")["Speciality_Clean"].value_counts().reset_index().head(5)
                sp_data.columns = ["Speciality", "Count"]
                render_premium_chart(px.bar, sp_data, x="Count", y="Speciality", orientation="h", title="6. Primary Specialized Domains Leadership", color_discrete_sequence=["#C9A84C"])
                st.markdown("<p class='insight-text'>Personnel capacity alignment mapping the top 5 clinical categories.</p>", unsafe_allow_html=True)

            r4c1, r4c2 = st.columns(2)
            with r4c1:
                df["Seniority_Group"] = pd.cut(df["ExperienceYears"], bins=[-1, 5, 15, 30, 120], labels=["Junior (<5Y)", "Mid-Career", "Senior", "Veteran (30Y+)"])
                sg_data = df["Seniority_Group"].value_counts().reset_index()
                sg_data.columns = ["Segment", "Count"]
                render_premium_chart(px.bar, sg_data, x="Segment", y="Count", title="7. Structural Seniority Tier Breakdowns", color_discrete_sequence=["#E8C547"])
                st.markdown("<p class='insight-text'>Workforce segmentation sorted by professional maturity levels.</p>", unsafe_allow_html=True)
            with r4c2:
                violin_df = exclude_unknown(df, "Status_Clean")
                render_premium_chart(px.violin, violin_df, x="Status_Clean", y="ExperienceYears", color="Status_Clean", title="8. Longevity Distribution Density vs Status", color_discrete_sequence=["#2A9D8F", "#EF4444"])
                st.markdown("<p class='insight-text'>Density curve showing if legal status flags map to specific career ages.</p>", unsafe_allow_html=True)

            r5c1, r5c2 = st.columns(2)
            with r5c1:
                spec_known = exclude_unknown(df, "Speciality_Clean", "Gender_Clean")
                g_spec = spec_known[spec_known["Speciality_Clean"].isin(spec_known["Speciality_Clean"].value_counts().head(4).index)]
                g_spec = g_spec.groupby(["Speciality_Clean", "Gender_Clean"]).size().reset_index(name="Count")
                render_premium_chart(px.bar, g_spec, x="Speciality_Clean", y="Count", color="Gender_Clean", barmode="stack", title="9. Gender Dispersal in Core Fields", color_discrete_sequence=["#C9A84C", "#457B9D"])
                st.markdown("<p class='insight-text'>Stacked volume distribution of male vs female counts across primary domains. <em>Gender inferred from first name where not explicitly provided — treat as approximate.</em></p>", unsafe_allow_html=True)
            with r5c2:
                track_status = exclude_unknown(df, "RegType_Clean", "Status_Clean").groupby(["RegType_Clean", "Status_Clean"]).size().reset_index(name="Count")
                render_premium_chart(px.bar, track_status, x="RegType_Clean", y="Count", color="Status_Clean", barmode="stack", title="10. Compliance Status Breakdown by Track", color_discrete_sequence=["#2A9D8F", "#E8C547", "#EF4444"])
                st.markdown("<p class='insight-text'>Hierarchical compliance mapping nested directly within license types.</p>", unsafe_allow_html=True)

        # ────────────────────────────────────────────────────────────────────────
        # SECTION 2: EDUCATIONAL & REGIONAL TRENDS (10 VISUALS)
        # ────────────────────────────────────────────────────────────────────────
        with tab2:
            r6c1, r6c2 = st.columns(2)
            with r6c1:
                v_data = df[df["RegYear"] >= 1995].groupby("RegYear").size().reset_index(name="Count")
                render_premium_chart(px.line, v_data, x="RegYear", y="Count", title="1. Annual Inflow Registration Velocity", markers=True, color_discrete_sequence=["#C9A84C"])
                st.markdown("<p class='insight-text'>Historical long-term pipeline capacity acceleration since 1995.</p>", unsafe_allow_html=True)
            with r6c2:
                c_data = exclude_unknown(df[df["RegYear"] >= 2010], "RegType_Clean").groupby(["RegYear", "RegType_Clean"]).size().reset_index(name="Count")
                render_premium_chart(px.bar, c_data, x="RegYear", y="Count", color="RegType_Clean", barmode="group", title="2. Cohort Expansion Vectors (Post-2010)", color_discrete_sequence=["#2A9D8F", "#E8C547"])
                st.markdown("<p class='insight-text'>Annual development of category intake streams over the last 15 years.</p>", unsafe_allow_html=True)

            r7c1, r7c2 = st.columns(2)
            with r7c1:
                # Full (non-truncated) counts, used by both the Graph and Table views.
                un_known = exclude_unknown(df, "Univ_Clean")
                un_data_full = un_known["Univ_Clean"].value_counts().reset_index()
                un_data_full.columns = ["University", "Count"]

                view_mode_univ = st.radio("View", ["📊 Graph", "📋 Table"], horizontal=True, key="da_univ_view", label_visibility="collapsed")

                if view_mode_univ == "📊 Graph":
                    un_data = un_data_full.copy()
                    if len(un_data) > 10:
                        top_part = un_data.head(10)
                        others_sum = un_data.iloc[10:]["Count"].sum()
                        un_data = pd.concat([top_part, pd.DataFrame([{"University": "Others", "Count": others_sum}])], ignore_index=True)
                    un_data = un_data.sort_values("Count", ascending=True)
                    render_premium_chart(px.bar, un_data, x="Count", y="University", orientation="h",
                                         title="3. Top 10 Universities — Educational Leaderboard",
                                         color_discrete_sequence=["#C9A84C"])
                    st.markdown("<p class='insight-text'>Top 10 universities by workforce volume — switch to Table above to see every university.</p>", unsafe_allow_html=True)
                else:
                    st.dataframe(un_data_full.sort_values("Count", ascending=False), use_container_width=True, hide_index=True, height=340)
                    st.markdown("<p class='insight-text'>Every university in the current filter selection, sorted by doctor count.</p>", unsafe_allow_html=True)
            with r7c2:
                dg_data = exclude_unknown(df, "Degree_Clean")["Degree_Clean"].value_counts().reset_index().head(5)
                dg_data.columns = ["Degree", "Count"]
                render_premium_chart(px.bar, dg_data, x="Degree", y="Count", title="4. Primary Credentials Tiers Breakdown", color_discrete_sequence=["#E8C547"])
                st.markdown("<p class='insight-text'>Distribution of initial qualification titles across total active records.</p>", unsafe_allow_html=True)

            r8c1, r8c2 = st.columns(2)
            with r8c1:
                # Shows every KNOWN province (not a subset) - simple sorted bar reads
                # clearer to a non-technical audience than a funnel shape. Rows with
                # no identifiable province are excluded rather than shown as "Unknown".
                pr_data = exclude_unknown(df, "Province_Clean")["Province_Clean"].value_counts().reset_index()
                pr_data.columns = ["Province", "Count"]
                pr_data = pr_data.sort_values("Count", ascending=True)
                render_premium_chart(px.bar, pr_data, x="Count", y="Province", orientation="h",
                                     title="5. Full Provincial Share of Registered Doctors",
                                     color_discrete_sequence=["#2A9D8F"], text="Count")
                st.markdown("<p class='insight-text'>Every identifiable province is shown here — AJK is ground-truth (from its own registry series); other provinces are inferred from the doctor's primary university.</p>", unsafe_allow_html=True)
            with r8c2:
                # Shows every known district/city, not just a top-N slice. Height grows
                # with the number of districts so labels never overlap. Rows with no
                # identifiable district are excluded rather than shown as "Unknown".
                ct_data = exclude_unknown(df, "City_Clean")["City_Clean"].value_counts().reset_index()
                ct_data.columns = ["City", "Count"]
                if ct_data.empty:
                    st.info("ℹ️ No District/City data is available in the current registry export — this field appears to not be tracked in the source data.")
                else:
                    ct_data = ct_data.sort_values("Count", ascending=True)
                    city_chart_height = max(340, 24 * len(ct_data))
                    render_premium_chart(px.bar, ct_data, x="Count", y="City", orientation="h",
                                         title="6. All Districts — Doctor Sourcing Concentration",
                                         color_discrete_sequence=["#E8C547"], text="Count",
                                         chart_height=city_chart_height)
                    st.markdown("<p class='insight-text'>Every identifiable district/city in the registry is listed here, sorted by doctor count.</p>", unsafe_allow_html=True)

            r9c1, r9c2 = st.columns(2)
            with r9c1:
                reg_v = exclude_unknown(df[df["RegYear"] >= 1990], "Province_Clean").groupby(["RegYear", "Province_Clean"]).size().reset_index(name="Count")
                yearly_totals = reg_v.groupby("RegYear")["Count"].transform("sum")
                reg_v["Percentage"] = (reg_v["Count"] / yearly_totals * 100).round(2)

                prov_toggle = st.toggle("Show as % of yearly total", key="prov_trend_toggle")
                y_col = "Percentage" if prov_toggle else "Count"
                y_label = "% of Yearly Total" if prov_toggle else "Count"

                fig_prov = px.line(reg_v, x="RegYear", y=y_col, color="Province_Clean",
                                title="7. Provincial Registration Trend (Every Year, Every Province)",
                                color_discrete_sequence=["#2A9D8F", "#C9A84C", "#E8C547", "#EF4444", "#64748B", "#457B9D"],
                                labels={y_col: y_label})
                fig_prov.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#374151", family="Inter", size=11),
                    title_font=dict(size=13, color="#9A6B00", family="Cinzel"),
                    margin=dict(t=50, b=40, l=50, r=20), height=340,
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#374151", size=10))
                )
                fig_prov.update_xaxes(showgrid=False, zeroline=False)
                fig_prov.update_yaxes(gridcolor="rgba(0,0,0,0.08)", zeroline=False)
                st.plotly_chart(fig_prov, use_container_width=True, config={'displayModeBar': False})
                st.markdown("<p class='insight-text'>Shows what each identifiable province's registration volume looked like in every year on record.</p>", unsafe_allow_html=True)
            with r9c2:
                un_mix = exclude_unknown(df, "Univ_Clean", "RegType_Clean")
                un_mix = un_mix[un_mix["Univ_Clean"].isin(un_mix["Univ_Clean"].value_counts().head(3).index)]
                un_mix = un_mix.groupby(["Univ_Clean", "RegType_Clean"]).size().reset_index(name="Count")
                render_premium_chart(px.bar, un_mix, x="Univ_Clean", y="Count", color="RegType_Clean", barmode="stack", title="8. Academic Output vs License Track", color_discrete_sequence=["#C9A84C", "#2A9D8F"])
                st.markdown("<p class='insight-text'>Stacked comparison mapping institutional output to practice parameters.</p>", unsafe_allow_html=True)

            r10c1, r10c2 = st.columns(2)
            with r10c1:
                deg_exp = exclude_unknown(df, "Degree_Clean").groupby("Degree_Clean")["ExperienceYears"].mean().reset_index(name="AvgExp").sort_values(by="AvgExp", ascending=False).head(5)
                render_premium_chart(px.bar, deg_exp, x="Degree_Clean", y="AvgExp", title="9. Seniority Index Across Core Degrees", color_discrete_sequence=["#2A9D8F"])
                st.markdown("<p class='insight-text'>Average practitioner practice years mapped across primary initial degrees.</p>", unsafe_allow_html=True)
            with r10c2:
                matrix = exclude_unknown(df, "Province_Clean", "RegType_Clean").groupby(["Province_Clean", "RegType_Clean"]).size().reset_index(name="Count")
                render_premium_chart(px.bar, matrix, x="Province_Clean", y="Count", color="RegType_Clean", barmode="group", title="10. Regional Workforce Layout Grid", color_discrete_sequence=["#E8C547", "#C9A84C"])
                st.markdown("<p class='insight-text'>Comparative workforce spread mapping track distributions across all provinces.</p>", unsafe_allow_html=True)

            # ────────────────────────────────────────────────────────────
            # SECTION 3: GENDER & REGIONAL DEMOGRAPHICS (added per review)
            # ────────────────────────────────────────────────────────────
            st.markdown("<h4 style='color:#9A6B00; font-family:Cinzel; margin-top:1.5rem;'>Gender & Regional Demographics</h4>", unsafe_allow_html=True)
            st.markdown("<p class='insight-text' style='margin-top:-8px;'>⚠️ Where the registry has no explicit gender field, gender is inferred from the doctor's first name only — treat as approximate. AJK province is ground-truth (from its own registry series); other provinces are inferred from university.</p>", unsafe_allow_html=True)

            r11c1, r11c2 = st.columns(2)
            with r11c1:
                prov_gender = exclude_unknown(df, "Province_Clean", "Gender_Clean").groupby(["Province_Clean", "Gender_Clean"]).size().reset_index(name="Count")
                render_premium_chart(px.bar, prov_gender, x="Province_Clean", y="Count", color="Gender_Clean",
                                     barmode="group", title="11. Male vs Female Doctors — Every Province",
                                     color_discrete_sequence=["#2A9D8F", "#E8C547"])
                st.markdown("<p class='insight-text'>Side-by-side male and female doctor counts for every identifiable province.</p>", unsafe_allow_html=True)
            with r11c2:
                city_gender = exclude_unknown(df, "City_Clean", "Gender_Clean").groupby(["City_Clean", "Gender_Clean"]).size().reset_index(name="Count")
                if city_gender.empty:
                    st.info("ℹ️ No District/City data is available in the current registry export — this field appears to not be tracked in the source data.")
                else:
                    total_by_city = city_gender.groupby("City_Clean")["Count"].sum().sort_values(ascending=True)
                    city_gender["City_Clean"] = pd.Categorical(city_gender["City_Clean"], categories=total_by_city.index, ordered=True)
                    city_gender = city_gender.sort_values("City_Clean")
                    city_gender_height = max(340, 26 * total_by_city.shape[0])
                    render_premium_chart(px.bar, city_gender, x="Count", y="City_Clean", color="Gender_Clean",
                                         orientation="h", barmode="stack",
                                         title="12. Male vs Female Doctors — Every District/City",
                                         color_discrete_sequence=["#2A9D8F", "#E8C547"],
                                         chart_height=city_gender_height)
                    st.markdown("<p class='insight-text'>Every identifiable district/city broken down by male and female doctor counts.</p>", unsafe_allow_html=True)

            r12c1, r12c2 = st.columns(2)
            with r12c1:
                gender_trend = exclude_unknown(df[df["RegYear"] >= 1990], "Gender_Clean").groupby(["RegYear", "Gender_Clean"]).size().reset_index(name="Count")
                render_premium_chart(px.line, gender_trend, x="RegYear", y="Count", color="Gender_Clean", markers=True,
                                     title="13. Gender Registration Trend Over the Years",
                                     color_discrete_sequence=["#2A9D8F", "#E8C547"])
                st.markdown("<p class='insight-text'>How many male vs female doctors registered each year, over time.</p>", unsafe_allow_html=True)
            with r12c2:
                yoy_gender = gender_trend.sort_values("RegYear").copy()
                yoy_gender["YoY_Change_%"] = yoy_gender.groupby("Gender_Clean")["Count"].pct_change() * 100
                yoy_gender = yoy_gender.dropna(subset=["YoY_Change_%"])
                yoy_gender = yoy_gender[yoy_gender["RegYear"] >= 2000]
                render_premium_chart(px.bar, yoy_gender, x="RegYear", y="YoY_Change_%", color="Gender_Clean",
                                     barmode="group", title="14. Year-over-Year % Change — Male vs Female",
                                     color_discrete_sequence=["#2A9D8F", "#E8C547"])
                st.markdown("<p class='insight-text'>Percentage growth or decline in registrations each year, compared to the year before, split by gender.</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ An unexpected configuration error occurred inside Doctor Analytics module.")
        with st.expander("Technical Diagnostic Log (Traceback)"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    render_doctor_analytics()
