"""
Yearly Deep Dive
A single page to answer, for any year (through 2018):
- How many doctors registered that year?
- Which field (including plain MBBS/BDS, not just specialists) had the most
  doctors that year?
- Within a chosen year, how many doctors of each gender were in each field?
- For a chosen field, how has its gender mix moved year by year?

Every section pairs a chart with the underlying table, so numbers are never
locked inside a visual only.

Data source: reads from MongoDB via utils.data_loader.load_all_data(), the
same pipeline used by the rest of the dashboard. Gender/Province/Specialist
signals come from the shared utils.gender_province module (see that file for
the accuracy notes / fixes already applied dashboard-wide).

MBBS/BDS INCLUSION:
Earlier version of this page only tracked doctors with a real specialization
(Speciality field filled in) — so MBBS/BDS-only doctors, who are the large
majority, never showed up as a "field" anywhere. This version adds a unified
"Field_All" column: specialists keep their actual specialty name, and every
other doctor is labeled by their base degree (MBBS / BDS / etc., taken from
their first qualification's Degree field). A toggle lets you include or
exclude MBBS/BDS from the field charts, since it dwarfs every specialty by
volume and can make the rest hard to compare on the same chart.
"""
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_YEAR = 2018
MIN_YEAR = 1990


def render_premium_chart(chart_func, *args, chart_height=380, **kwargs):
    """Same themed wrapper used on the Gender & Specialization page — data
    labels on by default for bars/pies."""
    try:
        fig = chart_func(*args, **kwargs)
        trace_types = {trace.type for trace in fig.data}
        if "pie" in trace_types:
            fig.update_traces(textinfo="percent+label", textposition="inside", selector=dict(type="pie"))
        if "bar" in trace_types:
            fig.update_traces(textposition="outside", cliponaxis=False, selector=dict(type="bar"))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#374151", family="Inter", size=11),
            title_font=dict(size=13, color="#9A6B00", family="Cinzel"),
            margin=dict(t=50, b=40, l=50, r=20), height=chart_height, showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#374151", size=10)),
        )
        fig.update_xaxes(showgrid=False, zeroline=False, title_font=dict(color="#374151", size=10))
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)", zeroline=False, title_font=dict(color="#374151", size=10))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        logger.error(f"Chart render skip: {e}")
        st.warning("⚠️ Render skip on this element.")


def kpi_card(label, value, sub=""):
    st.markdown(
        f"""<div style="background:#FFFFFF;border:1px solid rgba(184,134,11,0.3);
        border-radius:12px;padding:18px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <p style="color:#6B7280;font-size:0.75rem;font-weight:600;text-transform:uppercase;
        letter-spacing:1px;margin:0 0 6px 0;">{label}</p>
        <p style="color:#9A6B00;font-size:1.6rem;font-weight:700;margin:0;font-family:'Inter',sans-serif;">{value}</p>
        <p style="color:#6B7280;font-size:0.72rem;margin:4px 0 0 0;">{sub}</p>
        </div>""",
        unsafe_allow_html=True,
    )


# NOTE: flatten_qualifications() and load_and_enrich() used to be defined
# HERE, duplicated almost identically in the Gender & Specialization page.
# Since @st.cache_data caches by function identity, those were two totally
# separate caches - visiting one page never helped the other, so the
# expensive MongoDB fetch + qualification-flattening + gender/province
# inference ran again from scratch every time you switched pages (the
# "stuck on Running load_and_enrich()" symptom). Now both pages share ONE
# cached computation from utils/enriched_data.py.
from utils.enriched_data import load_and_enrich_all as load_and_enrich




@st.cache_data(ttl=3600)
def compute_field_df(include_mbbs: bool) -> pd.DataFrame:
    """The filtered field-level DataFrame (specialties, or specialties+MBBS/BDS
    if the toggle is on). Cached and keyed only on the include_mbbs bool -
    NOT on the underlying DataFrame - so flipping the year/field dropdowns
    elsewhere on the page never re-triggers this."""
    df = load_and_enrich()
    field_col = "Field_All" if include_mbbs else "Primary_Specialty"
    year_col = "Field_All_Year" if include_mbbs else "Specialty_Year"
    field_base_df = df if include_mbbs else df[df["Is_Specialist"]]
    return field_base_df[field_base_df[year_col].between(MIN_YEAR, MAX_YEAR) & field_base_df[field_col].notna()]


@st.cache_data(ttl=3600)
def compute_yearly_registration_counts() -> pd.Series:
    """Doctors registered per year - this never depends on the year/field
    pickers below, so it should only ever be computed once (per cache ttl),
    not on every widget interaction."""
    df = load_and_enrich()
    reg_df = df[df["RegYear"].between(MIN_YEAR, MAX_YEAR)]
    return reg_df.groupby("RegYear").size()


@st.cache_data(ttl=3600)
def compute_top_field_per_year(include_mbbs: bool) -> pd.DataFrame:
    """Which field led each year - independent of any live selection, so this
    is cached separately from the per-year/per-field drill-down sections."""
    field_col = "Field_All" if include_mbbs else "Primary_Specialty"
    year_col = "Field_All_Year" if include_mbbs else "Specialty_Year"
    field_df = compute_field_df(include_mbbs)
    top_field_per_year = (
        field_df.groupby([year_col, field_col]).size()
        .reset_index(name="Count")
        .sort_values([year_col, "Count"], ascending=[True, False])
        .groupby(year_col).first()
        .reset_index()
    )
    top_field_per_year[year_col] = top_field_per_year[year_col].astype(int)
    top_field_per_year.columns = ["Year", "Leading Field", "Doctors"]
    return top_field_per_year


@st.cache_data(ttl=3600)
def compute_field_heatmap(include_mbbs: bool) -> pd.DataFrame:
    """Top-10-fields-by-year matrix for the heatmap - independent of any live
    selection, cached separately so it's computed once, not on every click."""
    field_col = "Field_All" if include_mbbs else "Primary_Specialty"
    year_col = "Field_All_Year" if include_mbbs else "Specialty_Year"
    field_df = compute_field_df(include_mbbs)
    top_fields_overall = field_df[field_col].value_counts().head(10).index.tolist()
    heat_src = field_df[field_df[field_col].isin(top_fields_overall)]
    heat_tab = heat_src.groupby([year_col, field_col]).size().unstack(fill_value=0)
    heat_tab.index = heat_tab.index.astype(int)
    return heat_tab


def render_yearly_deep_dive():
    st.markdown("<h2 style='color:#B8860B;font-family:Cinzel;'>Yearly Deep Dive</h2>", unsafe_allow_html=True)
    st.caption(
        f"Every view on this page is capped at {MAX_YEAR}. Registration counts use each doctor's "
        "RegistrationDate year; field breakdowns use the year the relevant degree/specialization was passed. "
        "Gender is a first-name heuristic — treat as best-effort."
    )

    df = load_and_enrich()
    if df.empty:
        st.error("❌ No records returned from the database. Check the MongoDB connection/credentials.")
        return

    include_mbbs = st.checkbox(
        "Include MBBS/BDS as a field in the charts below (recommended — otherwise you only see specialists)",
        value=True, key="include_mbbs_toggle"
    )
    field_col = "Field_All" if include_mbbs else "Primary_Specialty"
    year_col = "Field_All_Year" if include_mbbs else "Specialty_Year"

    reg_df = df[df["RegYear"].between(MIN_YEAR, MAX_YEAR)]
    field_df = compute_field_df(include_mbbs)

    # ------------------------------------------------------------------
    # Top KPIs
    # ------------------------------------------------------------------
    yearly_counts = compute_yearly_registration_counts()
    peak_year = int(yearly_counts.idxmax()) if not yearly_counts.empty else MIN_YEAR
    peak_count = int(yearly_counts.max()) if not yearly_counts.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Years Covered", f"{reg_df['RegYear'].nunique()}", f"{int(reg_df['RegYear'].min())}–{MAX_YEAR}")
    with c2: kpi_card("Total Registered (in range)", f"{len(reg_df):,}")
    with c3: kpi_card("Busiest Year", f"{peak_year}", f"{peak_count:,} doctors registered")
    with c4: kpi_card("Avg Doctors / Year", f"{yearly_counts.mean():,.0f}")

    # ------------------------------------------------------------------
    # Section 1: Registration volume, year by year
    # ------------------------------------------------------------------
    st.markdown("### Doctors Registered, Every Year")
    yearly_tab = yearly_counts.reset_index()
    yearly_tab.columns = ["Year", "Doctors Registered"]
    render_premium_chart(px.bar, yearly_tab, x="Year", y="Doctors Registered", text_auto=True,
                          title=f"Registration Volume by Year (through {MAX_YEAR})",
                          color_discrete_sequence=["#2A9D8F"])
    st.dataframe(yearly_tab.sort_values("Year", ascending=False), use_container_width=True, hide_index=True, height=300)

    # ------------------------------------------------------------------
    # Section 2: Which field led each year (MBBS/BDS included if toggled on)
    # ------------------------------------------------------------------
    field_label = "Field (incl. MBBS/BDS)" if include_mbbs else "Specialty Field"
    st.markdown(f"### Top {field_label}, Every Year")
    top_field_per_year = compute_top_field_per_year(include_mbbs)
    render_premium_chart(px.bar, top_field_per_year, x="Year", y="Doctors", color="Leading Field",
                          title=f"Doctors in That Year's Leading {field_label}",
                          chart_height=440)
    st.dataframe(top_field_per_year.sort_values("Year", ascending=False), use_container_width=True, hide_index=True, height=320)

    st.markdown(f"#### {field_label} Popularity Heatmap — Top 10 Across the Years")
    heat_tab = compute_field_heatmap(include_mbbs)
    if not heat_tab.empty:
        render_premium_chart(px.imshow, heat_tab.T, aspect="auto", color_continuous_scale="YlGnBu",
                              labels=dict(x="Year", y="Field", color="Doctors"),
                              title=f"Doctors per {field_label}, per Year (Top 10)",
                              chart_height=460)
    else:
        st.info("Not enough dated records to build the heatmap.")

    # ------------------------------------------------------------------
    # Section 3: Pick a year — full breakdown
    # ------------------------------------------------------------------
    st.markdown("### Pick a Year — Full Breakdown")
    available_years = sorted(reg_df["RegYear"].dropna().unique().astype(int), reverse=True)
    if available_years:
        chosen_year = st.selectbox("Year", options=available_years, index=0, key="yearly_deep_dive_year")

        y_reg = reg_df[reg_df["RegYear"] == chosen_year]
        y_field = field_df[field_df[year_col] == chosen_year]

        yc1, yc2, yc3 = st.columns(3)
        with yc1: kpi_card(f"Doctors Registered in {chosen_year}", f"{len(y_reg):,}")
        with yc2:
            top_field_row = y_field[field_col].value_counts().head(1)
            top_field_name = top_field_row.index[0] if not top_field_row.empty else "N/A"
            top_field_count = int(top_field_row.iloc[0]) if not top_field_row.empty else 0
            kpi_card(f"Leading {field_label} in {chosen_year}", top_field_name, f"{top_field_count:,} doctors")
        with yc3:
            y_known_g = y_reg[y_reg["Gender"] != "Unknown"]
            fem_pct = (y_known_g["Gender"] == "Female").mean() * 100 if len(y_known_g) else 0
            kpi_card(f"Female Share in {chosen_year}", f"{fem_pct:.1f}%", f"of {len(y_known_g):,} gender-known doctors")

        yfc1, yfc2 = st.columns(2)
        with yfc1:
            y_field_counts = y_field[field_col].value_counts().head(10).reset_index()
            y_field_counts.columns = ["Field", "Doctors"]
            if not y_field_counts.empty:
                render_premium_chart(px.bar, y_field_counts, x="Doctors", y="Field", orientation="h",
                                      text_auto=True, title=f"Top {field_label}s in {chosen_year}",
                                      color_discrete_sequence=["#C9A84C"])
            else:
                st.info(f"No dated field records for {chosen_year}.")
        with yfc2:
            y_gender_counts = y_known_g["Gender"].value_counts().reset_index()
            y_gender_counts.columns = ["Gender", "Count"]
            if not y_gender_counts.empty:
                render_premium_chart(px.pie, y_gender_counts, names="Gender", values="Count", hole=0.5,
                                      title=f"Gender Split — {chosen_year}",
                                      color_discrete_sequence=["#2A9D8F", "#E8C547"])
            else:
                st.info(f"No gender-known doctors registered in {chosen_year}.")

        st.markdown(f"#### Gender by {field_label} — {chosen_year}")
        y_field_g = y_field[y_field["Gender"] != "Unknown"]
        top_fields_this_year = y_field_g[field_col].value_counts().head(10).index.tolist()
        y_field_gender = y_field_g[y_field_g[field_col].isin(top_fields_this_year)]
        gfy_tab = y_field_gender.groupby([field_col, "Gender"]).size().reset_index(name="Count")
        if not gfy_tab.empty:
            render_premium_chart(px.bar, gfy_tab, x=field_col, y="Count", color="Gender", barmode="group",
                                  text_auto=True, title=f"Doctors by {field_label} & Gender — {chosen_year} (top 10)",
                                  color_discrete_sequence=["#2A9D8F", "#E8C547"], chart_height=440)
            gfy_wide = y_field_gender.groupby([field_col, "Gender"]).size().unstack(fill_value=0)
            gfy_wide["Total"] = gfy_wide.sum(axis=1)
            st.dataframe(gfy_wide.sort_values("Total", ascending=False), use_container_width=True)
        else:
            st.info(f"No gender-known field records for {chosen_year}.")
    else:
        st.info("No registration years available in the current data.")

    # ------------------------------------------------------------------
    # Section 4: Pick a field (incl. MBBS/BDS if toggled) — gender trend
    # ------------------------------------------------------------------
    st.markdown(f"### Pick a {field_label} — Gender Trend Across the Years")
    field_options = field_df[field_col].value_counts().index.tolist()
    if field_options:
        default_idx = field_options.index("Obstetrics & Gynaecology") if "Obstetrics & Gynaecology" in field_options else 0
        chosen_field = st.selectbox(field_label, options=field_options, index=default_idx, key="yearly_deep_dive_field")

        f_df = field_df[(field_df[field_col] == chosen_field) & (field_df["Gender"] != "Unknown")]
        f_yearly = f_df.groupby([year_col, "Gender"]).size().reset_index(name="Count")
        f_yearly[year_col] = f_yearly[year_col].astype(int)

        if not f_yearly.empty:
            render_premium_chart(px.line, f_yearly, x=year_col, y="Count", color="Gender", markers=True,
                                  title=f"{chosen_field} — Male vs Female Doctors, By Year",
                                  color_discrete_sequence=["#2A9D8F", "#E8C547"])
            f_wide = f_df.groupby([year_col, "Gender"]).size().unstack(fill_value=0)
            f_wide.index = f_wide.index.astype(int)
            f_wide["Total"] = f_wide.sum(axis=1)
            f_wide["Female_%"] = (f_wide.get("Female", 0) / f_wide["Total"] * 100).round(1)
            st.dataframe(f_wide.sort_index(ascending=False), use_container_width=True, height=340)
        else:
            st.info(f"Not enough dated, gender-known records for {chosen_field}.")
    else:
        st.info("No field data available.")


if __name__ == "__main__":
    render_yearly_deep_dive()
