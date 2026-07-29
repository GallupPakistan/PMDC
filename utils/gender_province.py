"""
Centralized, fixed heuristics for Gender, Province, and Series/Specialist inference.

WHY THIS FILE EXISTS
---------------------
Every page in this dashboard used to carry its OWN copy-pasted version of
FEMALE_NAME_HINTS / MALE_NAME_HINTS / infer_gender() / UNIVERSITY_PROVINCE_MAP /
infer_province(). That meant:
  1. A bug fixed on one page did NOT apply to the other 7 pages.
  2. The gender heuristic matched keywords ANYWHERE in the full name string
     (substring match), not just the first name. Pakistani names very often
     carry a family/middle name that collides with the opposite gender's
     keyword list — e.g. "ABIHA ARIF BUTT" contains "Arif", which was in the
     MALE list, so a female doctor was silently mislabeled Male.

WHAT CHANGED
------------
1. infer_gender() / infer_gender_series() now match ONLY the first token of
   the Name field (the actual first name) — not the whole string. This alone
   removes the large majority of false Male/Female classifications caused by
   family names (Khan, Ali, Shah, Arif, Butt, etc.) leaking into the match.
2. Name-hint lists have been expanded (merged from every page's list, which
   had drifted out of sync with each other).
3. infer_province() now gives Province = "AJK" directly and with full
   confidence whenever the record's source collection is the AJK series
   (doctors_ajk_series) — no guessing needed for that slice of the data.
   All other provinces still rely on the university-name heuristic, because
   the registry has no other field that reveals it (S/D/N/B/F series encode
   qualification TYPE, not province — confirmed with the PMDC data owner).
4. Series (S/D/N/B/F/P/AJK) is now surfaced as an explicit field, and
   Is_Specialist can be derived directly from "is this doctor's collection
   the S-series" rather than guessing from free-text Speciality parsing.
   This is far more reliable, since S-series membership is a ground-truth
   fact about the record, not a text-parsing guess.

WHAT DID NOT CHANGE (and why)
------------------------------
- Gender for records whose first name isn't in either list still resolves
  to "Unknown". This is intentional: an honest "we don't know" is better
  than a wrong "Male"/"Female" guess. Expect the Unknown % to be higher
  after this fix than before — that is the accuracy improving, not
  regressing. If you want to shrink Unknown further, extend
  FEMALE_NAME_HINTS / MALE_NAME_HINTS below with more first names observed
  in your own data (the qualification-analytics "Unmapped" style report can
  help surface the most frequent unmatched first names).
- Province for non-AJK doctors is still a best-effort guess from the
  university name. There is no ground-truth field for Punjab/Sindh/KPK/etc.
  in the source data.
"""
from __future__ import annotations

import re
import pandas as pd
import numpy as np

# ============================================================================
# SOURCE COLLECTION / SERIES LABELS
# ============================================================================
SOURCE_LABEL_MAP = {
    "doctors": "P-Series",
    "doctors_s_series": "S-Series",
    "doctors_d_series": "D-Series",
    "doctors_f_series": "F-Series",
    "doctors_n_series": "N-Series",
    "doctors_b_series": "B-Series",
    "doctors_ajk_series": "AJK-Series",
}

# What each series actually means (confirmed definitions) — for tooltips/help text.
SERIES_DEFINITIONS = {
    "P-Series": "Punjab / General registration (largest series, ~100K records)",
    "S-Series": "Specialists — holders of a postgraduate qualification (FCPS/MS/MD etc.)",
    "D-Series": "Dental Surgeons (BDS)",
    "N-Series": "National graduates (local Pakistani medical/dental colleges)",
    "B-Series": "Basic / other registration block",
    "AJK-Series": "Azad Jammu & Kashmir region",
    "F-Series": "Foreign graduates (degree earned outside Pakistan)",
}


def is_specialist_from_series(source_table: pd.Series) -> pd.Series:
    """Ground-truth specialist flag: True iff the record comes from the
    S-series collection. Far more reliable than parsing free-text Speciality
    fields, since S-series membership is a fact about the record's origin,
    not a guess."""
    return source_table.astype(str).str.strip() == "doctors_s_series"


def is_dentist_from_series(source_table: pd.Series) -> pd.Series:
    """Ground-truth dentist flag: True iff the record comes from the D-series."""
    return source_table.astype(str).str.strip() == "doctors_d_series"


# ============================================================================
# GENDER — first-token name matching (fixes the substring false-positive bug)
# ============================================================================
FEMALE_NAME_HINTS = [
    "fatima", "ayesha", "aisha", "sana", "saba", "amna", "amina", "hina", "sara", "sarah",
    "iqra", "maria", "mariam", "rabia", "rabbia", "zainab", "zara", "uzma", "shazia", "naila",
    "farah", "asma", "humaira", "samina", "shabana", "rukhsana", "yasmin", "noreen", "bushra",
    "tahira", "nazia", "saima", "sumaira", "kiran", "mahnoor", "anum", "anam", "rida", "hira",
    "javeria", "khadija", "khadeeja", "nida", "nadia", "sehrish", "tehmina", "afshan",
    "sadia", "sidra", "mehwish", "mehreen", "komal", "zoya", "alina", "laiba", "amber",
    "asifa", "tayyaba", "tuba", "aiman", "aleena", "misbah", "summaya", "sumaiya", "sumayya",
    "nayab", "quratulain", "qurat", "sundas", "warda", "wajiha", "zoha", "zunaira", "areeba",
    "aqsa", "anila", "atiya", "ambreen", "erum", "faiza", "ghazala", "gulnaz", "hafsa", "huma",
    "ismat", "kausar", "kalsoom", "lubna", "madiha", "mubeen", "munazza", "nabila", "nazish",
    "nighat", "nimra", "parveen", "perveen", "rafia", "rahat", "raheela", "razia", "rehana",
    "robina", "romana", "rozina", "rukhsar", "saadia", "sabahat", "sabeen", "sabiha", "sadaf",
    "safia", "sajida", "samia", "samra", "sania", "seemab", "shaista", "shamim", "sobia",
    "sonia", "sumbal", "tania", "tasneem", "ushna", "yumna", "zeenat", "zahida", "begum",
]
MALE_NAME_HINTS = [
    "muhammad", "mohammad", "mohd", "ahmed", "ahmad", "ali", "hassan", "hussain", "husain",
    "khan", "raja", "malik", "sheikh", "chaudhry", "mian", "abdul", "umar", "omar", "usman",
    "bilal", "imran", "tariq", "shahid", "rashid", "farooq", "iqbal", "asif", "naveed",
    "kamran", "waqar", "zafar", "saeed", "rizwan", "naeem", "shoaib", "javed", "amir",
    "nasir", "naseer", "yousaf", "yousuf", "ghulam", "mirza", "syed", "sardar", "shah",
    "zeeshan", "faisal", "kashif", "adnan", "arslan", "danish", "fahad", "fahim", "hamza",
    "haroon", "ibrahim", "junaid", "kamal", "khalid", "mansoor", "mubashir", "nadeem",
    "qasim", "raheel", "salman", "sohail", "tahir", "yasir", "zubair", "aamir", "abid",
    "arif", "asad", "aurangzeb", "azhar", "babar", "farhan", "ghazanfar", "haider", "irfan",
    "jamil", "kaleem", "liaquat", "mazhar", "munir", "nawaz", "noman", "parvez", "qaiser",
    "rafiq", "riaz", "sabir", "sajjad", "shakeel", "tanveer", "waseem", "yaqoob", "zahid",
    "zaman", "zia", "javed",
]

FEMALE_PATTERN = "|".join(re.escape(h) for h in FEMALE_NAME_HINTS)
MALE_PATTERN = "|".join(re.escape(h) for h in MALE_NAME_HINTS)

# Women-only medical/dental colleges. If a doctor's primary university is one
# of these, we can label Gender = Female with full confidence, REGARDLESS of
# what the name-matching found — this recovers doctors whose first name isn't
# in our hint list yet.
WOMEN_ONLY_UNIVERSITY_KEYWORDS = [
    "khyber girls medical college",
    "women medical college",
    "fatima jinnah medical university",  # historically women-only
    "sardar begum",
]


def _first_token(name: str) -> str:
    if not isinstance(name, str):
        return ""
    stripped = name.strip().lower()
    return stripped.split()[0] if stripped else ""


def infer_gender(name: str) -> str:
    """Single-name version (for pages using df["Name"].apply(...))."""
    first = _first_token(name)
    if not first:
        return "Unknown"
    if re.search(FEMALE_PATTERN, first):
        return "Female"
    if re.search(MALE_PATTERN, first):
        return "Male"
    return "Unknown"


def infer_gender_series(names: pd.Series, universities: pd.Series | None = None) -> pd.Series:
    """Vectorized version. Optionally pass the primary-university Series to
    apply the women-only-college override for names the hint list misses."""
    first_tokens = names.fillna("").astype(str).str.strip().str.lower().str.split().str[0].fillna("")
    is_female = first_tokens.str.contains(FEMALE_PATTERN, regex=True, na=False)
    is_male = first_tokens.str.contains(MALE_PATTERN, regex=True, na=False)
    result = pd.Series(
        np.select([is_female, is_male], ["Female", "Male"], default="Unknown"),
        index=names.index,
    )

    if universities is not None:
        u = universities.fillna("").astype(str).str.lower()
        women_only_mask = pd.Series(False, index=names.index)
        for kw in WOMEN_ONLY_UNIVERSITY_KEYWORDS:
            women_only_mask |= u.str.contains(re.escape(kw), regex=True, na=False)
        result.loc[women_only_mask] = "Female"

    return result


# ============================================================================
# PROVINCE — AJK is ground-truth (from series), others remain university-based
# ============================================================================
UNIVERSITY_PROVINCE_MAP = {
    # Punjab
    "university of the punjab": "Punjab", "king edward medical university": "Punjab",
    "allama iqbal medical college": "Punjab", "university of health sciences": "Punjab",
    "nishtar medical university": "Punjab", "nishtar medical college": "Punjab",
    "fatima jinnah medical university": "Punjab", "punjab medical college": "Punjab",
    "rawalpindi medical university": "Punjab", "uhs lahore": "Punjab",
    "services institute of medical sciences": "Punjab", "sahiwal medical college": "Punjab",
    "gujranwala medical college": "Punjab", "dg khan medical college": "Punjab",
    "d.g. khan medical college": "Punjab", "sialkot medical college": "Punjab",
    "cmh lahore medical college": "Punjab", "continental medical college": "Punjab",
    "akhtar saeed medical college": "Punjab", "avicenna medical college": "Punjab",
    "azra naheed medical college": "Punjab", "central park medical college": "Punjab",
    "fatima memorial": "Punjab", "independent medical college": "Punjab",
    "islam medical college": "Punjab", "lahore medical": "Punjab",
    "multan medical": "Punjab", "rashid latif medical college": "Punjab",
    "sahara medical college": "Punjab", "university college of medicine": "Punjab",
    "wah medical college": "Punjab", "rai medical college": "Punjab",
    "m. islam medical college": "Punjab", "shalamar medical": "Punjab",
    "amna inayat medical college": "Punjab", "sargodha medical college": "Punjab",
    "quaid-e-azam medical college": "Punjab", "faisalabad medical university": "Punjab",
    "punjab medical university": "Punjab",
    # Sindh
    "dow university of health sciences": "Sindh", "jinnah sindh medical university": "Sindh",
    "liaquat university of medical": "Sindh", "liaquat national medical college": "Sindh",
    "ziauddin university": "Sindh", "karachi medical": "Sindh", "lumhs jamshoro": "Sindh",
    "baqai medical university": "Sindh", "hamdard college of medicine": "Sindh",
    "sindh medical college": "Sindh", "bahria university medical": "Sindh",
    "aga khan university": "Sindh", "peoples university of medical": "Sindh",
    "isra university": "Sindh", "muhammad medical college": "Sindh",
    "chandka medical college": "Sindh", "al-tibri medical college": "Sindh",
    "peoples medical college": "Sindh", "karachi university": "Sindh",
    # KPK
    "khyber medical university": "KPK", "khyber medical college": "KPK",
    "khyber girls medical college": "KPK", "ayub medical college": "KPK",
    "kabir medical college": "KPK", "gomal medical college": "KPK",
    "saidu medical college": "KPK", "bannu medical college": "KPK",
    "nowshera medical college": "KPK", "women medical college": "KPK",
    "rehman medical college": "KPK", "frontier medical college": "KPK",
    "swat medical college": "KPK", "northwest school of medicine": "KPK",
    # Balochistan
    "bolan university of medical": "Balochistan", "bolan medical college": "Balochistan",
    "loralai medical college": "Balochistan", "quetta": "Balochistan",
    # Islamabad
    "shaheed zulfiqar ali bhutto medical university": "Islamabad", "pims": "Islamabad",
    "shifa college of medicine": "Islamabad", "islamic international medical college": "Islamabad",
    "foundation university medical college": "Islamabad", "hbs medical": "Islamabad",
    "yusra medical": "Islamabad", "federal medical college": "Islamabad",
    "capital university of medical sciences": "Islamabad", "national university of medical sciences": "Islamabad",
    "rawal institute of health sciences": "Islamabad", "army medical college": "Islamabad",
    # AJK (university-name fallback; series-based override below is authoritative)
    "azad jammu and kashmir medical college": "AJK", "mohtarma benazir bhutto shaheed medical college": "AJK",
}


def infer_province(university: str) -> str:
    if not isinstance(university, str) or not university.strip():
        return "Unknown"
    u = university.lower()
    for key, prov in UNIVERSITY_PROVINCE_MAP.items():
        if key in u:
            return prov
    return "Unknown"


def infer_province_series(universities: pd.Series, source_table: pd.Series | None = None) -> pd.Series:
    """Vectorized. If source_table is provided, AJK-series records are
    force-set to Province='AJK' with full confidence (ground truth from the
    collection they came from), overriding the university-name guess."""
    u = universities.fillna("").astype(str).str.lower()
    province = pd.Series("Unknown", index=universities.index)
    for key, prov in UNIVERSITY_PROVINCE_MAP.items():
        unresolved = province == "Unknown"
        if not unresolved.any():
            break
        match = unresolved & u.str.contains(re.escape(key), regex=True, na=False)
        province.loc[match] = prov

    if source_table is not None:
        ajk_mask = source_table.astype(str).str.strip() == "doctors_ajk_series"
        province.loc[ajk_mask] = "AJK"

    return province
