"""Workflow invalidation when upstream configuration changes."""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from monami.config import MLConfig


def make_sampling_fingerprint(
    *,
    selected_level: int,
    slice_shape: Optional[tuple[int, int]],
    source_name: str,
    method: str,
    categories: int,
    sample_pct: float,
    seed: int,
    threshold_text: str,
    vmin: float,
    vmax: float,
) -> tuple[Any, ...]:
    return (
        int(selected_level),
        tuple(slice_shape) if slice_shape else None,
        str(source_name),
        str(method),
        int(categories),
        float(sample_pct),
        int(seed),
        str(threshold_text).strip(),
        round(float(vmin), 8),
        round(float(vmax), 8),
    )


def make_training_fingerprint(ml_config: MLConfig) -> tuple[Any, ...]:
    return (
        st.session_state.get("_committed_sampling_fingerprint"),
        float(ml_config.test_ratio),
        float(ml_config.dropout),
        int(ml_config.batch_size),
        int(ml_config.epochs),
        tuple(int(n) for n in ml_config.nodes_per_layer),
        str(ml_config.optimizer),
        int(ml_config.early_stopping_patience),
        str(ml_config.suffix),
        int(ml_config.n_nearest),
        str(ml_config.hidden_activation),
        str(ml_config.out_activation),
        int(ml_config.preview_interval),
        int(st.session_state.random_seed),
    )


def clear_training_state() -> None:
    st.session_state.model = None
    st.session_state.model_meta = None
    st.session_state.model_path = None
    st.session_state.history = None
    st.session_state.live_training_history = None
    st.session_state._committed_training_fingerprint = None


def clear_results_state() -> None:
    st.session_state.prediction_2d = None


def clear_downstream_from_sampling() -> None:
    clear_training_state()
    clear_results_state()


def commit_sampling_fingerprint(fingerprint: tuple[Any, ...]) -> None:
    st.session_state._committed_sampling_fingerprint = fingerprint
    st.session_state._current_sampling_fingerprint = fingerprint
    clear_downstream_from_sampling()


def commit_training_fingerprint(fingerprint: tuple[Any, ...]) -> None:
    st.session_state._committed_training_fingerprint = fingerprint
    st.session_state._current_training_fingerprint = fingerprint


def set_current_sampling_fingerprint(fingerprint: tuple[Any, ...]) -> None:
    previous = st.session_state.get("_current_sampling_fingerprint")
    st.session_state._current_sampling_fingerprint = fingerprint
    if (
        previous is not None
        and previous != fingerprint
        and st.session_state.get("_committed_sampling_fingerprint") is not None
    ):
        clear_downstream_from_sampling()


def set_current_training_fingerprint(fingerprint: tuple[Any, ...]) -> None:
    previous = st.session_state.get("_current_training_fingerprint")
    st.session_state._current_training_fingerprint = fingerprint
    if (
        previous is not None
        and previous != fingerprint
        and st.session_state.get("_committed_training_fingerprint") is not None
    ):
        clear_training_state()
        clear_results_state()


def refresh_sampling_fingerprint_from_data(
    selected_level: int,
    slice_shape: tuple[int, int],
    source_name: str,
) -> None:
    fingerprint = make_sampling_fingerprint(
        selected_level=selected_level,
        slice_shape=slice_shape,
        source_name=source_name,
        method=st.session_state.categorization_method,
        categories=int(st.session_state.categories),
        sample_pct=float(st.session_state.sample_pct),
        seed=int(st.session_state.random_seed),
        threshold_text=st.session_state.category_threshold_text,
        vmin=float(st.session_state.get("_sampling_vmin", 0.0)),
        vmax=float(st.session_state.get("_sampling_vmax", 0.0)),
    )
    set_current_sampling_fingerprint(fingerprint)


def migrate_legacy_fingerprints() -> None:
    """Backfill fingerprints for sessions created before invalidation tracking."""
    if (
        st.session_state.samples_df is not None
        and st.session_state.categorized_2d is not None
        and st.session_state.get("_committed_sampling_fingerprint") is None
    ):
        slice_2d = st.session_state.slice_2d
        vmin = st.session_state.get("_sampling_vmin")
        vmax = st.session_state.get("_sampling_vmax")
        if vmin is None or vmax is None:
            vmin = float(slice_2d.min()) if slice_2d is not None else 0.0
            vmax = float(slice_2d.max()) if slice_2d is not None else 0.0
        commit_sampling_fingerprint(
            make_sampling_fingerprint(
                selected_level=int(st.session_state.selected_level),
                slice_shape=tuple(slice_2d.shape) if slice_2d is not None else None,
                source_name=str(st.session_state.source_name),
                method=str(st.session_state.categorization_method),
                categories=int(st.session_state.categories),
                sample_pct=float(st.session_state.sample_pct),
                seed=int(st.session_state.random_seed),
                threshold_text=str(st.session_state.category_threshold_text),
                vmin=float(vmin),
                vmax=float(vmax),
            )
        )


def sampling_is_committed() -> bool:
    committed = st.session_state.get("_committed_sampling_fingerprint")
    current = st.session_state.get("_current_sampling_fingerprint")
    return (
        committed is not None
        and current is not None
        and committed == current
        and st.session_state.samples_df is not None
        and st.session_state.categorized_2d is not None
    )


def training_is_committed() -> bool:
    committed = st.session_state.get("_committed_training_fingerprint")
    current = st.session_state.get("_current_training_fingerprint")
    has_model = (
        st.session_state.model is not None and st.session_state.model_meta is not None
    ) or (
        st.session_state.get("model_path") is not None
        and st.session_state.history is not None
    )
    return (
        sampling_is_committed()
        and has_model
        and committed is not None
        and current is not None
        and committed == current
    )


def results_is_committed() -> bool:
    return training_is_committed() and st.session_state.prediction_2d is not None
