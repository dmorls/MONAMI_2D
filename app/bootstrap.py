"""Auto-bootstrap default Data and Sampling workflow on first app load."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.invalidation import commit_sampling_fingerprint, make_sampling_fingerprint
from monami.io import extract_slice, load_from_path_or_bytes
from monami.sampling import sample_fraction_to_strata, stratified_sample_dataframe
from monami.transform import categorize_slice, compute_thresholds, thresholds_to_text

DEFAULT_LEVEL = 0
DEFAULT_CATEGORIES = 3
DEFAULT_SAMPLE_PCT = 5.0
DEFAULT_CATEGORIZATION_METHOD = "quantile"


def ensure_default_workflow(project_root: Path) -> None:
    """
    Load demo data (slice level 0) and run default sampling (3 categories, ~5%).

    Runs once per session when samples are not yet available.
    """
    if st.session_state.get("_workflow_bootstrapped"):
        return
    if st.session_state.samples_df is not None and st.session_state.categorized_2d is not None:
        st.session_state._workflow_bootstrapped = True
        return

    default_file = project_root / "1_original_exhaustive" / "porosity_3d.txt"
    if not default_file.exists():
        st.session_state.bootstrap_status = f"Demo file not found: {default_file}"
        st.session_state._workflow_bootstrapped = True
        return

    try:
        volume, meta = load_from_path_or_bytes(default_file)
        slice_2d = extract_slice(volume, DEFAULT_LEVEL, meta)

        st.session_state.volume_3d = volume
        st.session_state.grid_meta = meta
        st.session_state.source_name = default_file.name
        st.session_state.property_name = meta.property_name
        st.session_state.selected_level = DEFAULT_LEVEL
        st.session_state.slice_2d = slice_2d
        st.session_state.continuous_slice = slice_2d.copy()

        st.session_state.categories = DEFAULT_CATEGORIES
        st.session_state.n_categories_effective = DEFAULT_CATEGORIES
        st.session_state.sample_pct = DEFAULT_SAMPLE_PCT
        st.session_state.categorization_method = DEFAULT_CATEGORIZATION_METHOD
        st.session_state.random_seed = int(st.session_state.get("random_seed", 42))

        v_data = slice_2d.ravel()
        edges = compute_thresholds(
            v_data,
            DEFAULT_CATEGORIES,
            method=DEFAULT_CATEGORIZATION_METHOD,
        )
        st.session_state.category_threshold_text = thresholds_to_text(edges)
        st.session_state._threshold_config_key = (
            DEFAULT_CATEGORIZATION_METHOD,
            DEFAULT_CATEGORIES,
            float(v_data.min()),
            float(v_data.max()),
        )

        _, categorized, applied_edges = categorize_slice(
            slice_2d,
            n_categories=DEFAULT_CATEGORIES,
            method=DEFAULT_CATEGORIZATION_METHOD,
            bin_edges=edges,
        )
        grid_height, grid_width = categorized.shape
        n_h, n_v, n_samples = sample_fraction_to_strata(
            grid_height,
            grid_width,
            DEFAULT_SAMPLE_PCT / 100.0,
        )
        samples = stratified_sample_dataframe(
            categorized,
            n_h,
            n_v,
            seed=st.session_state.random_seed,
        )

        st.session_state.categorized_2d = categorized
        st.session_state.samples_df = samples
        st.session_state.category_thresholds = applied_edges
        st.session_state.n_categories_effective = len(applied_edges) - 1
        st.session_state.categories = len(applied_edges) - 1

        st.session_state.bootstrap_status = (
            f"Loaded {default_file.name} (level {DEFAULT_LEVEL}), "
            f"{st.session_state.categories} categories, "
            f"{len(samples):,} samples (~{DEFAULT_SAMPLE_PCT}% density)."
        )
        commit_sampling_fingerprint(
            make_sampling_fingerprint(
                selected_level=DEFAULT_LEVEL,
                slice_shape=tuple(slice_2d.shape),
                source_name=default_file.name,
                method=DEFAULT_CATEGORIZATION_METHOD,
                categories=DEFAULT_CATEGORIES,
                sample_pct=DEFAULT_SAMPLE_PCT,
                seed=st.session_state.random_seed,
                threshold_text=st.session_state.category_threshold_text,
                vmin=float(v_data.min()),
                vmax=float(v_data.max()),
            )
        )
        st.session_state._sampling_vmin = float(v_data.min())
        st.session_state._sampling_vmax = float(v_data.max())
    except Exception as exc:
        st.session_state.bootstrap_status = f"Auto-bootstrap failed: {exc}"

    st.session_state._workflow_bootstrapped = True
