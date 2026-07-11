"""Page 1: Load 3D data and select slice."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.invalidation import refresh_sampling_fingerprint_from_data
from app.state import init_session_state, workflow_ready
from monami.io import extract_slice, load_from_path_or_bytes
from monami.viz import heatmap_slice, slice_gallery

init_session_state()

st.title("1. Data — 3D load and slice selection")

if workflow_ready() and st.session_state.get("bootstrap_status"):
    status = st.session_state.bootstrap_status
    if "failed" not in status.lower() and "not found" not in status.lower():
        st.info(
            f"**Default workflow ready:** {status} "
            "You can change the slice or data source below."
        )

project_root = Path(__file__).resolve().parents[2]
default_file = project_root / "1_original_exhaustive" / "porosity_3d.txt"

source_option = st.radio("Data source", ["Default demo file", "Upload file"], horizontal=True)

if source_option == "Upload file":
    uploaded = st.file_uploader("SGEMS-style exhaustive file (.txt)", type=["txt", "sgems"])
    if uploaded is not None:
        if st.button("Load uploaded file", type="primary"):
            with st.spinner("Loading 3D volume..."):
                volume, meta = load_from_path_or_bytes(uploaded)
                st.session_state.volume_3d = volume
                st.session_state.grid_meta = meta
                st.session_state.source_name = uploaded.name
                st.session_state.property_name = meta.property_name
                st.session_state.selected_level = 0
                st.session_state.slice_2d = extract_slice(volume, 0, meta)
                st.session_state.continuous_slice = st.session_state.slice_2d.copy()
                refresh_sampling_fingerprint_from_data(
                    0,
                    tuple(st.session_state.slice_2d.shape),
                    st.session_state.source_name,
                )
            st.success(f"Loaded {uploaded.name}: {meta.nx}×{meta.ny}×{meta.nz}")
else:
    if st.button("Reload demo file", type="secondary") or st.session_state.volume_3d is None:
        if default_file.exists():
            with st.spinner("Loading demo volume..."):
                volume, meta = load_from_path_or_bytes(default_file)
                st.session_state.volume_3d = volume
                st.session_state.grid_meta = meta
                st.session_state.source_name = default_file.name
                st.session_state.property_name = meta.property_name
                level = st.session_state.selected_level
                st.session_state.slice_2d = extract_slice(volume, level, meta)
                st.session_state.continuous_slice = st.session_state.slice_2d.copy()
                refresh_sampling_fingerprint_from_data(
                    level,
                    tuple(st.session_state.slice_2d.shape),
                    st.session_state.source_name,
                )
            st.success(f"Loaded {default_file.name}: {meta.nx}×{meta.ny}×{meta.nz}")
        else:
            st.error(f"Demo file not found: {default_file}")

if workflow_ready():
    meta = st.session_state.grid_meta
    volume = st.session_state.volume_3d

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Grid nx", meta.nx)
    col2.metric("Grid ny", meta.ny)
    col3.metric("Levels nz", meta.nz)
    col4.metric("Property", meta.property_name)

    level = st.slider("Slice level (Z index)", 0, meta.nz - 1, st.session_state.selected_level)
    st.session_state.selected_level = level
    slice_2d = extract_slice(volume, level, meta)
    st.session_state.slice_2d = slice_2d
    st.session_state.continuous_slice = slice_2d.copy()
    refresh_sampling_fingerprint_from_data(
        level,
        tuple(slice_2d.shape),
        st.session_state.source_name,
    )

    vmin, vmax = float(slice_2d.min()), float(slice_2d.max())
    st.plotly_chart(
        heatmap_slice(
            slice_2d,
            title=f"{st.session_state.source_name} — level {level}",
            zmin=vmin,
            zmax=vmax,
        ),
        use_container_width=False,
    )

    preview_levels = list(range(0, meta.nz, max(1, meta.nz // 4)))[:4]
    if len(preview_levels) > 1:
        st.subheader("Slice gallery")
        st.plotly_chart(slice_gallery(volume, preview_levels), use_container_width=False)

    if st.button("Confirm slice for workflow", type="secondary"):
        st.session_state.slice_2d = slice_2d
        st.session_state.continuous_slice = slice_2d.copy()
        st.success(f"Slice level {level} confirmed ({slice_2d.shape[0]}×{slice_2d.shape[1]}). Proceed to Sampling.")
    elif st.session_state.selected_level != level:
        st.caption("Slice preview updated. Click **Discretize and sample** on Sampling after changing level.")
