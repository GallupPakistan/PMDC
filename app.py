"""
Pakistan Medical Workforce Analytics Platform (PMDC Dashboard)
v2.0 — Institutional Theme with Glassmorphism, Gold Accents & Smooth Animations
"""
import streamlit as st
import logging
import re
from datetime import datetime
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PMDC Medical Workforce Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://pmdc.org.pk",
        "Report a bug": "https://github.com/yourusername/pmdc-dashboard/issues",
        "About": "# PMDC Dashboard\nPakistan Medical Workforce Intelligence Platform v2.0\nInspired by Judicial Excellence"
    }
)

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;800&family=Inter:wght@300;400;600;700&display=swap');
    [data-testid="stSidebarNav"] { display: none !important; }
    .stApp { background-color: #FFFFFF !important; }
    [data-testid="stSidebar"] { background-color: #F7F8FA !important; border-right: 1px solid rgba(184, 134, 11, 0.2) !important; }
    h1, h2, h3, .title { font-family: 'Cinzel', serif !important; letter-spacing: 0.5px; }
    body, p, span, div, .subtitle { font-family: 'Inter', sans-serif !important; color: #1F2937; }
    .header-banner {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 249, 251, 0.95) 100%);
        backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(184, 134, 11, 0.3); padding: 24px 32px; border-radius: 16px;
        display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 24px;
        margin-bottom: 2rem; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.08); transition: border 0.4s ease;
    }
    .header-banner:hover { border: 1px solid rgba(184, 134, 11, 0.55); }
    .header-left { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 24px; }
    .title { color: #B8860B; font-size: 2.4rem; font-weight: 700; margin: 0; text-shadow: 0 2px 6px rgba(0,0,0,0.06); }
    .subtitle { color: #1F2937; font-size: 1.1rem; font-weight: 400; margin: 6px 0 0 0; display: flex; align-items: center; gap: 12px; }
    .gold-text { color: rgba(184, 134, 11, 0.85); font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin: 8px 0 0 0; }
    .header-right { display: flex; gap: 40px; background: rgba(0, 0, 0, 0.02); padding: 12px 24px; border-radius: 12px; border: 1px solid rgba(0, 0, 0, 0.06); margin-left: auto; margin-right: auto; }
    .stat-block { text-align: center; padding: 0 16px; }
    .stat-block:not(:last-child) { border-right: 1px solid rgba(0, 0, 0, 0.1); }
    .muted { color: #6B7280; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 4px 0; white-space: nowrap; }
    .gold-bold { color: #9A6B00; font-size: 1.25rem; font-weight: 700; margin: 0; font-family: 'Inter', sans-serif !important; }
    .sidebar-divider { margin: 1.5rem 0; border-bottom: 1px solid rgba(184, 134, 11, 0.2); }
    .sidebar-footer { font-size: 0.8rem; color: #4B5563; line-height: 1.6; background: rgba(0, 0, 0, 0.02); padding: 12px; border-radius: 8px; border: 1px solid rgba(0, 0, 0, 0.06); }
    .sidebar-footer b { color: #1F2937; }
    .status-badge { padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; display: inline-flex; align-items: center; gap: 6px; }
    .status-live { background: rgba(42, 157, 143, 0.12); border: 1px solid rgba(42, 157, 143, 0.5); color: #1E7A6D; }
    .status-dot { height: 6px; width: 6px; background-color: #2A9D8F; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #2A9D8F; }
    div.stButton > button {
        background: #FFFFFF !important; color: #1F2937 !important; border: 1px solid rgba(0, 0, 0, 0.12) !important;
        padding: 10px 16px !important; border-radius: 8px !important; text-align: left !important;
        font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        background: rgba(184, 134, 11, 0.08) !important; color: #9A6B00 !important;
        border: 1px solid rgba(184, 134, 11, 0.6) !important; box-shadow: 0 4px 12px rgba(184, 134, 11, 0.12) !important;
        transform: translateX(4px);
    }
    div.stButton > button:focus:not(:active) { border-color: #B8860B !important; color: #000000 !important; }
    [data-testid="stMetricValue"] { color: #1F2937 !important; }
    [data-testid="stMetricLabel"] { color: #6B7280 !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def initialize_session_state():
    defaults = {
        "page": "Overview",
        "data_refresh_count": 0,
        "last_refresh": None,
        "db_connected": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_state()

def load_page(page_name):
    try:
        pages_dir = Path(__file__).parent / "pages"
        page_files = {
            "Overview": "01_Overview.py",
            "Doctor Analytics": "1_Doctor_Analytics.py",
            "Qualification Analytics": "2_Qualification_Analytics.py",
            "University Insights": "3_University_Insights.py",
            "Status License": "4_Status_License.py",
            "Data Quality": "5_Data_Quality.py",
            "Doctor Search": "6_Doctor_Search.py",
            "Gender & Specialization Deep Dive": "7_Gender_Specialization_Deep_Dive.py",
            "Yearly Deep Dive": "8_Yearly_Deep_Dive.py"
        }
        page_file = page_files.get(page_name)
        if not page_file:
            st.error(f"❌ Page '{page_name}' not found in configuration.")
            return
        page_path = pages_dir / page_file
        if not page_path.exists():
            st.error(f"❌ Page file not found: {page_path}")
            return
        import importlib.util

        project_root = str(Path(__file__).parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # CRITICAL FIX: this used to call spec.loader.exec_module(module) on a
        # BRAND NEW module object every single time load_page() ran - which is
        # every rerun, i.e. every click/filter/widget interaction, for every
        # page. Re-executing a .py file from scratch re-creates every function
        # in it (including every @st.cache_data-wrapped one) as a brand new
        # function object each time. Streamlit's cache treats a "new" function
        # object as having no prior cache history, so EVERY @st.cache_data in
        # EVERY page was being silently reset on every interaction - no amount
        # of adding more caching inside the pages could fix this, because the
        # page itself was being reloaded from scratch underneath it.
        #
        # Fix: give each page module a stable name and only exec_module() it
        # ONCE, then reuse the already-loaded module (and its already-warm
        # caches) on every subsequent call, exactly like a normal `import`
        # statement would.
        module_key = f"_pmdc_page_{Path(page_file).stem}"
        if module_key in sys.modules:
            module = sys.modules[module_key]
        else:
            spec = importlib.util.spec_from_file_location(module_key, page_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                # Loading failed - don't leave a broken half-loaded module
                # cached under this key, or every future attempt will reuse
                # the broken object and never retry.
                del sys.modules[module_key]
                raise

        sanitized = re.sub(r"[^a-z0-9]+", "_", page_name.lower()).strip("_")
        render_fn_name = f"render_{sanitized}"
        render_fn = getattr(module, render_fn_name, None)
        if render_fn:
            render_fn()
    except Exception as e:
        logger.error(f"Error loading page '{page_name}': {str(e)}")
        st.error(f"❌ Failed to load page: {page_name}")
        with st.expander("🔧 Technical Details"):
            import traceback
            st.code(traceback.format_exc(), language="python")

def show_sidebar_navigation():
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center; margin-top: 1rem; margin-bottom:2rem;">
                <div style="font-size:3.5rem; margin-bottom:0.2rem; filter: drop-shadow(0 4px 12px rgba(184,134,11,0.2));">🏥</div>
                <h2 style="margin:0; font-size:1.45rem; color:#B8860B; font-family:'Cinzel', serif; font-weight:700; letter-spacing:1px;">
                    PMDC Dashboard
                </h2>
                <p style="margin:0.6rem 0 0 0; font-size:0.72rem;
                          color:rgba(184,134,11,0.85); letter-spacing:2px;
                          font-weight:700; text-transform:uppercase; font-family:'Inter', sans-serif;">
                    Medical Workforce Intelligence
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#9A6B00; font-size:0.75rem; font-weight:700;"
            "letter-spacing:2px; margin-bottom:1rem; font-family:\'Inter\', sans-serif;"
            "text-transform:uppercase;'>📊 Navigation Framework</p>",
            unsafe_allow_html=True
        )
        pages = [
            ("📈 Overview", "Overview"),
            ("🩺 Doctor Analytics", "Doctor Analytics"),
            ("♀️♂️ Gender & Specialization", "Gender & Specialization Deep Dive"),
            ("📅 Yearly Deep Dive", "Yearly Deep Dive"),
            ("🎓 Qualification Analytics", "Qualification Analytics"),
            ("🏛️ University Insights", "University Insights"),
            ("⏰ Status License", "Status License"),
            ("📊 Data Quality", "Data Quality"),
            ("🔍 Doctor Search", "Doctor Search"),
        ]
        for label, page_name in pages:
            is_active = st.session_state.page == page_name
            display_label = f"✨ {label}" if is_active else label
            if st.button(
                display_label,
                key=f"nav_{page_name}",
                use_container_width=True,
                help=f"Route context to: {page_name}"
            ):
                st.session_state.page = page_name
                st.rerun()
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#9A6B00; font-size:0.75rem; font-weight:700;"
            "letter-spacing:2px; margin-bottom:1rem; font-family:\'Inter\', sans-serif;"
            "text-transform:uppercase;'>ℹ️ Engine Environment</p>",
            unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Timeline", datetime.now().strftime("%b %d"))
        with col2:
            st.metric("Session Year", "2026")
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#9A6B00; font-size:0.75rem; font-weight:700;"
            "letter-spacing:2px; margin-bottom:1rem; font-family:\'Inter\', sans-serif;"
            "text-transform:uppercase;'>🔄 Cache Architecture</p>",
            unsafe_allow_html=True
        )
        if st.button("⚡ Invalidate Runtime Caches", use_container_width=True, key="sidebar_refresh"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.data_refresh_count += 1
            st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
            st.rerun()
        if st.session_state.data_refresh_count > 0:
            st.caption(
                f"**Cycles Completed:** {st.session_state.data_refresh_count}  \n"
                f"**Timestamp:** {st.session_state.last_refresh or '—'}"
            )
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-footer">
                <p style="margin:0 0 4px 0;"><b>Application Version:</b> 2.0.3</p>
                <p style="margin:0 0 4px 0;"><b>Data Node:</b> Atlas Cluster Pipeline</p>
                <p style="margin:0 0 4px 0;"><b>Target Dataset Heap:</b> ~80k rows</p>
                <p style="margin:0;"><b>Synchronized:</b> 2026-06-05</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def check_database_connection():
    try:
        from utils.mongodb import get_mongo_collection
        from utils.data_loader import COLLECTIONS
        total = 0
        for name in COLLECTIONS:
            collection = get_mongo_collection(name)
            total += collection.count_documents({})
        return True, total
    except Exception as e:
        logger.error(f"Database connection layer faulty execution: {e}")
        return False, 0

def show_header_banner(db_connected, doc_count):
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    search_tag = (
        '<span class="status-badge status-live">'
        '<span class="status-dot"></span> LIVE DATA COMPLIANT</span>'
        if db_connected else
        '<span class="status-badge" style="background:rgba(239,68,68,0.1);'
        'border:1px solid rgba(239,68,68,0.4); color:#DC2626;">'
        '⚠️ PIPELINE FAILURE</span>'
    )
    st.markdown(
        f"""
        <div class="header-banner">
            <div class="header-left">
                <div style="font-size:3.8rem; filter:drop-shadow(0 4px 16px rgba(184,134,11,0.25)); line-height: 1;">
                    🏥
                </div>
                <div>
                    <p class="title">PMDC Workforce Intelligence</p>
                    <p class="subtitle">National Medical Registry Analytics Grid &nbsp; {search_tag}</p>
                    <p class="gold-text">Governance & Regulatory Oversight Infrastructure • Islamabad Division</p>
                </div>
            </div>
            <div class="header-right">
                <div class="stat-block">
                    <p class="muted">Infrastructure</p>
                    <p class="gold-bold" style="color: {'#1E7A6D' if db_connected else '#DC2626'};">
                        {'ONLINE' if db_connected else 'OFFLINE'}
                    </p>
                </div>
                <div class="stat-block">
                    <p class="muted">Indexed Nodes</p>
                    <p class="gold-bold">{doc_count:,}</p>
                </div>
                <div class="stat-block">
                    <p class="muted">Temporal Sync</p>
                    <p class="gold-bold">{date_str}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    try:
        db_connected, doc_count = check_database_connection()
        st.session_state.db_connected = db_connected
        show_sidebar_navigation()
        show_header_banner(db_connected, doc_count)
        if not db_connected:
            st.warning(
                "⚠️ **No Database Detected (Running in Local/Mock Mode)**\n\n"
                "To connect real data, ensure MongoDB Atlas credentials are added to **Settings -> Variables and Secrets** on Hugging Face."
            )
        load_page(st.session_state.page)
    except Exception as e:
        logger.critical(f"Unhandled app exception thread killed: {str(e)}")
        st.error("❌ **System Kernel Fault Encountered**")
        st.error(f"Context payload trace: {str(e)}")
        with st.expander("🔧 Diagnostics Stack trace"):
            import traceback
            st.code(traceback.format_exc(), language="python")

if __name__ == "__main__":
    main()