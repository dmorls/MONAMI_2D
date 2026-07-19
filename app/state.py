"""Shared Streamlit session state helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.bootstrap import ensure_default_workflow
from app.invalidation import (
    algorithm_is_committed,
    migrate_legacy_fingerprints,
    results_is_committed,
    sampling_is_committed,
    training_is_committed,
)
from monami.algorithms.registry import DEFAULT_ALGORITHM_ID, resolve_algorithm_id


def init_session_state() -> None:
    defaults = {
        "volume_3d": None,
        "grid_meta": None,
        "selected_level": 0,
        "slice_2d": None,
        "continuous_slice": None,
        "categorized_2d": None,
        "categories": 3,
        "categorization_method": "quantile",
        "category_thresholds": None,
        "category_threshold_text": "",
        "n_categories_effective": 3,
        "samples_df": None,
        "train_df": None,
        "neighbor_pool_df": None,
        "test_df": None,
        "model": None,
        "model_meta": None,
        "model_path": None,
        "history": None,
        "live_training_history": None,
        "training_log": [],
        "training_preview_frames": [],
        "training_preview_gif": None,
        "training_preview_last_metrics": None,
        "prediction_2d": None,
        "simulations": [],
        "last_report_path": None,
        "property_name": "porosity",
        "source_name": "porosity_3d.txt",
        "random_seed": 42,
        "sample_pct": 5.0,
        "selected_algorithm_id": DEFAULT_ALGORITHM_ID,
        "algorithm_config": {},
        "_workflow_bootstrapped": False,
        "bootstrap_status": None,
        "_committed_sampling_fingerprint": None,
        "_current_sampling_fingerprint": None,
        "_committed_algorithm_fingerprint": None,
        "_current_algorithm_fingerprint": None,
        "_committed_training_fingerprint": None,
        "_current_training_fingerprint": None,
        "_sampling_vmin": None,
        "_sampling_vmax": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Migrate pre-numbering algorithm ids still sitting in the session.
    if st.session_state.get("selected_algorithm_id"):
        st.session_state.selected_algorithm_id = resolve_algorithm_id(
            st.session_state.selected_algorithm_id
        )

    project_root = Path(__file__).resolve().parent.parent
    ensure_default_workflow(project_root)
    migrate_legacy_fingerprints()


def workflow_ready() -> bool:
    return st.session_state.slice_2d is not None and st.session_state.grid_meta is not None


def sampling_ready() -> bool:
    return sampling_is_committed()


def algorithm_ready() -> bool:
    return algorithm_is_committed()


def training_ready() -> bool:
    return training_is_committed()


def results_ready() -> bool:
    return results_is_committed()


def page_title(label: str, ready: bool) -> str:
    return f"✓ {label}" if ready else label
