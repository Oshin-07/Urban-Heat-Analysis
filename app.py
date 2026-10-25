"""Urban Heat Analysis: Map-First Explorer Dashboard.

Run:  streamlit run app.py
(The earlier sidebar dashboard is kept as app_classic.py.)
"""
import streamlit as st

from dashboard import theme

st.set_page_config(page_title="Urban Heat Analysis", page_icon="🌡️", layout="wide", initial_sidebar_state="collapsed")
st.markdown(theme.CSS, unsafe_allow_html=True)
try:
    st.logo("assets/logo_dark.svg" if theme.MODE == "dark" else "assets/logo.svg", size="large")
except Exception:  # very old Streamlit versions: the title in the page header still identifies the app
    pass

pages = [
    st.Page("views/overview.py", title="Overview", icon="🌡️", default=True),
    st.Page("views/explore.py", title="Explore Map", icon="🗺️", url_path="explore"),
    st.Page("views/compare.py", title="Compare Cities", icon="⚖️", url_path="compare"),
    st.Page("views/ml_insights.py", title="ML Insights", icon="🧠", url_path="ml"),
    st.Page("views/about.py", title="About", icon="📘", url_path="about"),
]
st.navigation(pages, position="top").run()
