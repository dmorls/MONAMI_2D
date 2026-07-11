"""MONAMI 2D Categorical Geostatistical Workflow."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.sidebar import inject_sidebar_nav_styles, render_bootstrap_caption
from app.state import (
    init_session_state,
    page_title,
    results_ready,
    sampling_ready,
    training_ready,
    workflow_ready,
)

st.set_page_config(
    page_title="MONAMI Workflow",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
inject_sidebar_nav_styles()

pages = [
    st.Page(
        "views/1_Data.py",
        title=page_title("Data", workflow_ready()),
        icon=":material/database:",
        default=True,
    ),
    st.Page(
        "views/2_Sampling.py",
        title=page_title("Sampling", sampling_ready()),
        icon=":material/grid_on:",
    ),
    st.Page(
        "views/3_Training.py",
        title=page_title("Training", training_ready()),
        icon=":material/model_training:",
    ),
    st.Page(
        "views/4_Results.py",
        title=page_title("Results", results_ready()),
        icon=":material/analytics:",
    ),
]

pg = st.navigation(pages)
render_bootstrap_caption()
pg.run()
