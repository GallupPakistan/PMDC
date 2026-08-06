"""
Gender & Specialization Deep Dive

Covers, with clear visuals and % figures:
1. Male vs Female share of all registered doctors
2. Gender ratio trend over the years
3. Gender split by province
4. % of doctors who hold a qualification beyond MBBS (specialization),
   plus a breakdown by actual specialization degree type (FCPS, MD, MS, etc.)
5. Specialization rate by gender
6. Number & % of specialists by field
7. Each field's share of new specialists, year over year
8. Gender ratio within each specialization, and its trend over time

Data source: reads from the CSV export via utils.data_loader.load_all_data(), the same
pipeline used by the rest of the dashboard.

NOTES ON THIS VERSION:
- All year-based charts are capped at 2018 (no later years shown), matching
  the rest of the dashboard's data horizon.
- Data labels are turned on for every bar/pie chart (via Plotly's text_auto
  for bars, textinfo for pies).
- Fixed a real bug in "Specialist Population by Field": the field-name column
  from value_counts().reset_index() was not actually named "index" (it kept
  the original Series name "Primary_Specialty"), so the old
  .rename(columns={"index": "Field"}) silently did nothing and the chart's
  y="Field" reference didn't exist, which crashed and got swallowed by the
  chart wrapper's try/except ("Render skip on this element."). Fixed by
  setting the index name explicitly before reset_index().
- Gender charts show BOTH Male and Female wherever the comparison is
  inherently two-sided (overall split, by province, by year, specialization
  rate). Where a chart is specifically about "how female a field is" (the
  female-dominance ranking and its trend), only Female is shown, because
  that is literally what those two charts are measuring.
- NEW: the Specialization Overview section now also breaks specialists down
  by their actual postgraduate DEGREE type (FCPS, MD, MS, MCPS, Diploma,
  Ph.D, etc.) — pulled straight from each doctor's qualification record,
  canonicalized the same way as the rest of the dashboard (see
  DEGREE_CANONICAL_MAP below) — instead of only showing one aggregate bar.

FIXES CARRIED FROM utils/gender_province.py:
- Gender inferred from the doctor's FIRST NAME ONLY (fixes the "Abiha Arif
  Butt" style false-positive where a family name in the full string flipped
  gender). A women-only-university override recovers additional doctors.
- Province is ground-truth ("AJK") for AJK-series doctors; every other
  province is still a best-effort university-name guess.
- Is_Specialist has two signals: ground-truth "from the S-series registry",
  and text-parsed "holds a qualification beyond MBBS".
"""
import logging
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SPEC_COLS = [f"Qualification_{i}_Speciality" for i in range(1, 8)]
YEAR_COLS = [f"Qualification_{i}_PassingYear" for i in range(1, 8)]

_NON_SPECIALTY_VALUES = {"", ".", "BASIC MEDICAL QUALIFICATION", "NAN"}

# Every year-based view on this page stops here, matching the rest of the dashboard.
MAX_YEAR = 2018

# Same degree-name canonicalization used elsewhere in the dashboard (Qualification
# Analytics / University Insights), so "FCPS", "F.C.P.S", "fcps" etc. all collapse
# to one proper label instead of showing up as separate near-duplicate bars.
DEGREE_CANONICAL_MAP = {
    "mbbs": "MBBS", "bds": "BDS", "md": "MD", "ms": "MS",
    "fcps": "FCPS", "mcps": "MCPS", "dpt": "DPT", "phd": "Ph.D",
}


def normalize_token(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def canonicalize_degree(raw) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"
    token = normalize_token(raw)
    return DEGREE_CANONICAL_MAP.get(token, raw.strip().title())


def real_specialty_mask(col: pd.Series) -> pd.Series:
    """Vectorized: True where a Speciality cell holds a real (non-blank,
    non-placeholder) value — used for the text-parsed specialist signal."""
    cleaned = col.astype(str).str.strip().str.upper()
    return col.notna() & ~cleaned.isin(_NON_SPECIALTY_VALUES)


def render_premium_chart(chart_func, *args, chart_height=380, **kwargs):
    """Renders a chart with the dashboard's standard theme AND turns on data
    labels automatically: percent+label for pies, value labels for bars."""
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


# NOTE: flatten_qualifications() and load_and_enrich() used to be defined
# HERE, duplicated almost identically in Yearly Deep Dive too. Since
# @st.cache_data caches by function identity, those were two totally
# separate caches - visiting one page never helped the other, so the
# expensive CSV load + qualification-flattening + gender/province
# inference ran again from scratch every time you switched pages (the
# "stuck on Running load_and_enrich()" symptom). Now both pages share ONE
# cached computation from utils/enriched_data.py.
from utils.enriched_data import load_and_enrich_all as load_and_enrich




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


def render_gender_specialization_deep_dive():
    st.markdown("<h2 style='color:#B8860B;font-family:Cinzel;'>Gender & Specialization Deep Dive</h2>", unsafe_allow_html=True)
    st.caption(
        "Gender inferred from first name (heuristic). Province is ground-truth for AJK, inferred from university "
        f"for every other province. All years shown are capped at {MAX_YEAR}. Treat non-AJK province and gender "
        "figures as best-effort."
    )

    df = load_and_enrich()
    if df.empty:
        st.error("❌ No records returned. Check that data/doctors_combined_full_all_qualifications.csv is present.")
        return

    known_g = df[df["Gender"] != "Unknown"]
    known_p = df[df["Province"] != "Unknown"]

    # ------------------------------------------------------------------
    # 1. Overall gender split — BOTH genders shown (this is a two-sided split)
    # ------------------------------------------------------------------
    st.markdown("### Gender Distribution — All Registered Doctors")
    total = len(df)
    male_n, female_n, unk_n = (df["Gender"] == "Male").sum(), (df["Gender"] == "Female").sum(), (df["Gender"] == "Unknown").sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Registered", f"{total:,}")
    with c2: kpi_card("Male", f"{male_n:,}", f"{male_n/total*100:.1f}% of all")
    with c3: kpi_card("Female", f"{female_n:,}", f"{female_n/total*100:.1f}% of all")
    with c4: kpi_card("Gender Unresolved", f"{unk_n:,}", f"{unk_n/total*100:.1f}% of all")
    gpie = known_g["Gender"].value_counts().reset_index()
    gpie.columns = ["Gender", "Count"]
    render_premium_chart(px.pie, gpie, names="Gender", values="Count", hole=0.55,
                          title="Gender Share (Known)", color_discrete_sequence=["#2A9D8F", "#E8C547"])
    st.markdown(f"<p class='insight-text' style='color:#6B7280;font-size:0.8rem;font-style:italic;'>"
                f"Among gender-known doctors: {male_n/len(known_g)*100:.1f}% Male vs {female_n/len(known_g)*100:.1f}% Female.</p>",
                unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 2. Gender ratio trend over years — BOTH genders shown, capped at MAX_YEAR
    # ------------------------------------------------------------------
    st.markdown("### Gender Trend Over the Years")
    yr_df = known_g[known_g["RegYear"].between(1990, MAX_YEAR)]
    yr_tab = yr_df.groupby(["RegYear", "Gender"]).size().reset_index(name="Count")
    render_premium_chart(px.line, yr_tab, x="RegYear", y="Count", color="Gender", markers=True,
                          title=f"Male vs Female New Registrations, By Year (through {MAX_YEAR})",
                          color_discrete_sequence=["#2A9D8F", "#E8C547"])

    yr_wide = yr_df.groupby(["RegYear", "Gender"]).size().unstack(fill_value=0)
    yr_wide["Total"] = yr_wide.sum(axis=1)
    yr_wide["Female_%"] = (yr_wide.get("Female", 0) / yr_wide["Total"] * 100).round(1)
    early = yr_wide["Female_%"].head(5).mean()
    recent = yr_wide["Female_%"].tail(5).mean()
    direction = "risen" if recent > early else "fallen"
    st.markdown(f"<p class='insight-text' style='color:#6B7280;font-size:0.8rem;font-style:italic;'>"
                f"Female share has {direction}: {early:.1f}% (earliest 5 yrs) vs {recent:.1f}% (most recent 5 yrs, through {MAX_YEAR}).</p>",
                unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3. Gender by province — BOTH genders shown
    # ------------------------------------------------------------------
    st.markdown("### Gender Split by Province")
    pg = known_g[known_g["Province"] != "Unknown"]
    prov_tab = pg.groupby(["Province", "Gender"]).size().unstack(fill_value=0)
    prov_tab["Total"] = prov_tab.sum(axis=1)
    prov_tab["Female_%"] = (prov_tab.get("Female", 0) / prov_tab["Total"] * 100).round(1)
    prov_tab["Male_%"] = (prov_tab.get("Male", 0) / prov_tab["Total"] * 100).round(1)
    prov_display = prov_tab[["Total", "Female_%", "Male_%"]].sort_values("Female_%", ascending=False)
    prov_plot = prov_tab.reset_index()
    render_premium_chart(px.bar, prov_plot, x="Province", y=["Female_%", "Male_%"], barmode="group",
                          title="Gender % by Province", text_auto=".1f",
                          color_discrete_sequence=["#E8C547", "#2A9D8F"])
    st.dataframe(prov_display, use_container_width=True)
    st.markdown("<p class='insight-text' style='color:#6B7280;font-size:0.8rem;font-style:italic;'>"
                "AJK is ground-truth (from the AJK-series registry). Other provinces are inferred from primary university.</p>",
                unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 4. % pursuing more than MBBS — not a gender split, no gender shown.
    #    Now includes a breakdown by actual specialization degree type.
    # ------------------------------------------------------------------
    st.markdown("### Specialization Overview (Beyond MBBS/BDS)")
    n_spec_series = int(df["Is_Specialist_Series"].sum())
    n_spec_text = int(df["Is_Specialist"].sum())
    spec_kpi_df = pd.DataFrame({
        "Category": ["Confirmed Specialists (S-Series)", "Declared 2nd Qualification", "MBBS/BDS Only"],
        "Percent": [n_spec_series/total*100, n_spec_text/total*100, (total-n_spec_text)/total*100],
    })
    render_premium_chart(px.bar, spec_kpi_df, x="Category", y="Percent", text_auto=".2f",
                          title="% of Registered Doctors by Specialization Status",
                          color_discrete_sequence=["#2A9D8F"])
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Confirmed Specialists (S-Series)", f"{n_spec_series/total*100:.2f}%", f"{n_spec_series:,} doctors")
    with c2: kpi_card("Declared 2nd Qualification", f"{n_spec_text/total*100:.2f}%", f"{n_spec_text:,} doctors")
    with c3: kpi_card("MBBS/BDS Only", f"{(total-n_spec_text)/total*100:.2f}%", f"{total-n_spec_text:,} doctors")

    st.markdown("#### Specialization Degree Types")
    deg_counts = df["Specialist_Degree"].dropna()
    deg_counts = deg_counts[deg_counts != "Unknown"].value_counts()
    if not deg_counts.empty:
        deg_tab_full = pd.DataFrame({
            "Degree": deg_counts.index,
            "Doctors": deg_counts.values,
        }).sort_values("Doctors", ascending=False)
        n_spec_for_pct = len(df[df["Is_Specialist"]])
        deg_tab_full["% of Specialists"] = (deg_tab_full["Doctors"] / n_spec_for_pct * 100).round(2)

        all_degree_names = deg_tab_full["Degree"].tolist()
        default_top10 = all_degree_names[:10]
        degree_pick = st.multiselect(
            "Degree types to show (rest are grouped into 'Others')",
            options=all_degree_names, default=default_top10, key="degree_type_slicer"
        )

        shown = deg_tab_full[deg_tab_full["Degree"].isin(degree_pick)].copy()
        rest = deg_tab_full[~deg_tab_full["Degree"].isin(degree_pick)]
        others_sum = rest["Doctors"].sum()
        if others_sum > 0:
            others_row = pd.DataFrame([{
                "Degree": "Others",
                "Doctors": others_sum,
                "% of Specialists": round(others_sum / n_spec_for_pct * 100, 2),
            }])
            shown = pd.concat([shown, others_row], ignore_index=True)
        shown = shown.sort_values("Doctors", ascending=False)

        render_premium_chart(px.bar, shown, x="Degree", y="Doctors", text_auto=True,
                              title="Specialists by Degree Type (Top 10 + Others)",
                              color_discrete_sequence=["#457B9D"])
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.markdown("<p class='insight-text' style='color:#6B7280;font-size:0.8rem;font-style:italic;'>"
                    "Degree names are taken directly from each doctor's qualification record and canonicalized "
                    "(e.g. 'F.C.P.S' and 'FCPS' both count as FCPS). Use the selector above to swap any degree "
                    "type in or out of 'Others'.</p>", unsafe_allow_html=True)
    else:
        st.info("No specialization degree data available to break down.")

    # ------------------------------------------------------------------
    # 5. Men vs women specialization rate — BOTH genders shown
    # ------------------------------------------------------------------
    st.markdown("### Specialization Rate by Gender")
    sr = known_g.groupby("Gender")["Is_Specialist"].agg(Specialists="sum", Total="count")
    sr["Specialization_Rate_%"] = (sr["Specialists"] / sr["Total"] * 100).round(2)
    render_premium_chart(px.bar, sr.reset_index(), x="Gender", y="Specialization_Rate_%", text_auto=".2f",
                          title="Specialization Rate by Gender", color="Gender",
                          color_discrete_sequence=["#E8C547", "#2A9D8F"])
    st.dataframe(sr, use_container_width=True)

    # ------------------------------------------------------------------
    # 6. Specialists by field — no gender dimension here
    # ------------------------------------------------------------------
    st.markdown("### Specialist Population by Field")
    spec_only = df[df["Is_Specialist"]]
    fld = spec_only["Primary_Specialty"].value_counts()
    fld_tab = pd.DataFrame({"Doctors": fld, "% of All Specialists": (fld / len(spec_only) * 100).round(2)})
    # FIX: name the index explicitly BEFORE reset_index() — value_counts()'s
    # index keeps the original column name ("Primary_Specialty"), not "index",
    # so the old rename(columns={"index": "Field"}) was a no-op and the chart
    # below referenced a "Field" column that didn't exist, crashing silently.
    fld_tab.index.name = "Field"
    top20 = fld_tab.head(20).reset_index()
    render_premium_chart(px.bar, top20, x="Doctors", y="Field", orientation="h", text_auto=True,
                          title="Top 20 Specialty Fields by Doctor Count",
                          color_discrete_sequence=["#C9A84C"], chart_height=560)
    st.dataframe(fld_tab, use_container_width=True, height=420)

    # ------------------------------------------------------------------
    # 7. Field share over time — no gender dimension, capped at MAX_YEAR
    # ------------------------------------------------------------------
    st.markdown("### Specialty Field Share, Over Time")
    top_fields = fld.head(8).index.tolist()
    field_pick = st.multiselect("Fields to compare", options=fld.index.tolist(), default=top_fields[:6])
    sp_yr = spec_only[spec_only["Primary_Specialty"].isin(field_pick) & spec_only["Specialty_Year"].between(1995, MAX_YEAR)]
    if not sp_yr.empty:
        yr_field = sp_yr.groupby([sp_yr["Specialty_Year"].astype(int), "Primary_Specialty"]).size().unstack(fill_value=0)
        yr_field_pct = yr_field.div(yr_field.sum(axis=1), axis=0).mul(100).round(1).reset_index()
        yr_field_long = yr_field_pct.melt(id_vars="Specialty_Year", var_name="Field", value_name="Share_%")
        render_premium_chart(px.line, yr_field_long, x="Specialty_Year", y="Share_%", color="Field", markers=True,
                              title=f"Field Share of New Specialists, By Year of Passing (through {MAX_YEAR})", chart_height=460)
    else:
        st.info("Not enough dated records for the selected fields.")

    # ------------------------------------------------------------------
    # 8. Gender ratio within each specialization — table shows BOTH, but the
    #    "how female-dominated" ranking chart intentionally shows FEMALE ONLY
    #    (that is literally what it measures).
    # ------------------------------------------------------------------
    st.markdown("### Gender Ratio Within Each Specialization")
    sg = spec_only[spec_only["Gender"] != "Unknown"]
    sgt = sg.groupby(["Primary_Specialty", "Gender"]).size().unstack(fill_value=0)
    sgt["Total"] = sgt.sum(axis=1)
    sgt = sgt[sgt["Total"] >= 15]
    sgt["Female_%"] = (sgt.get("Female", 0) / sgt["Total"] * 100).round(1)
    sgt["Male_%"] = (sgt.get("Male", 0) / sgt["Total"] * 100).round(1)
    sgt_display = sgt[["Total", "Female_%", "Male_%"]].sort_values("Female_%", ascending=False)
    top_gender_fields = sgt_display.head(15).reset_index()
    render_premium_chart(px.bar, top_gender_fields, x="Female_%", y="Primary_Specialty", orientation="h", text_auto=".1f",
                          title="Most Female-Dominated Specializations (min. 15 known doctors)",
                          color_discrete_sequence=["#E8C547"], chart_height=520)
    st.dataframe(sgt_display, use_container_width=True, height=420)

    st.markdown("### Gender Ratio Within a Specialization, Over Time")
    field_for_trend = st.selectbox("Pick a specialization", options=sgt_display.index.tolist(),
                                    index=sgt_display.index.tolist().index("Obstetrics & Gynaecology")
                                    if "Obstetrics & Gynaecology" in sgt_display.index else 0)
    fd = spec_only[(spec_only["Primary_Specialty"] == field_for_trend) & (spec_only["Gender"] != "Unknown")
                   & spec_only["Specialty_Year"].between(1995, MAX_YEAR)]
    if len(fd) >= 10:
        fd = fd.copy()
        fd["Period"] = pd.cut(fd["Specialty_Year"].astype(int), bins=[1994, 2005, 2015, MAX_YEAR],
                               labels=["1995-2005", "2006-2015", f"2016-{MAX_YEAR}"])
        pt = fd.groupby(["Period", "Gender"]).size().unstack(fill_value=0)
        pt["Total"] = pt.sum(axis=1)
        pt["Female_%"] = (pt.get("Female", 0) / pt["Total"] * 100).round(1)
        render_premium_chart(px.bar, pt.reset_index(), x="Period", y="Female_%", text_auto=".1f",
                              title=f"Female Share of {field_for_trend} Over Time (through {MAX_YEAR})",
                              color_discrete_sequence=["#C9A84C"])
        st.dataframe(pt[["Total", "Female_%"]], use_container_width=True)
    else:
        st.info("Not enough dated records for this field to show a reliable trend.")


if __name__ == "__main__":
    render_gender_specialization_deep_dive()
