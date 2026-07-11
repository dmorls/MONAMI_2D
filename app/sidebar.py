"""Sidebar navigation styling helpers."""

from __future__ import annotations

import streamlit as st

from app.state import results_ready, sampling_ready, training_ready, workflow_ready


def inject_sidebar_nav_styles() -> None:
    """Color completed workflow nav buttons green (nth-child matches page order)."""
    readiness = [
        workflow_ready(),
        sampling_ready(),
        training_ready(),
        results_ready(),
    ]
    rules = []
    for index, ready in enumerate(readiness, start=1):
        if ready:
            rules.append(
                "[data-testid='stSidebarNav'] ul li:nth-child("
                f"{index}) span {{ color: #2ecc71 !important; font-weight: 600; }}"
            )
    if rules:
        st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)


def render_bootstrap_caption() -> None:
    """Optional one-line bootstrap summary (no duplicate step list)."""
    bootstrap_status = st.session_state.get("bootstrap_status")
    if not bootstrap_status:
        return
    if "failed" in bootstrap_status.lower() or "not found" in bootstrap_status.lower():
        st.sidebar.warning(bootstrap_status)
    elif workflow_ready() and sampling_ready():
        st.sidebar.caption(bootstrap_status)
