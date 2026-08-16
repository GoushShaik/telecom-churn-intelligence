"""
app.py (V3) — application router.

V2 relied on Streamlit's legacy filename-based multipage system, with emoji
embedded directly in filenames (e.g. "1_📊_Executive_Overview.py"). On
Windows, that produced broken/mojibake sidebar labels and URLs (visible as
"=f0e Executive Overview" and "localhost:8501/=f0e_Executive_Overview" in
the V2 screenshots) — a filesystem/OS text-encoding issue, not something
fixable by "trying a different emoji."

V3 fixes this at the root by switching to Streamlit's explicit st.Page /
st.navigation API: page titles and icons are plain Python strings passed as
arguments, never embedded in a filename, so there is nothing for the
filesystem or URL router to mis-encode. This also gives the grouped sidebar
sections (Overview / Customer Analytics / Model / Documentation) natively.
"""

import streamlit as st
from theme import inject_css

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

st.markdown(
    """
    <div style="padding: 0.5rem 0 1rem 0;">
        <span style="font-size:1.3rem; font-weight:800;">📡 Telecom Churn</span><br>
        <span style="font-size:1.3rem; font-weight:800;">Intelligence</span>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = {
    "Overview": [
        st.Page("views/executive_overview.py", title="Executive Overview", icon="📊", default=True),
    ],
    "Customer Analytics": [
        st.Page("views/customer_risk.py", title="Customer Risk", icon="🎯"),
        st.Page("views/what_if.py", title="What-If Simulator", icon="🔄"),
        st.Page("views/customer_segments.py", title="Customer Segments", icon="👥"),
        st.Page("views/churn_insights.py", title="Churn Insights", icon="📈"),
    ],
    "Model": [
        st.Page("views/model_performance.py", title="Model Performance", icon="🧪"),
        st.Page("views/external_validation.py", title="External Validation", icon="🌐"),
    ],
    "Documentation": [
        st.Page("views/methodology.py", title="Methodology", icon="📘"),
        st.Page("views/about.py", title="About", icon="ℹ️"),
    ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()
