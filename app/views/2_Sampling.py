"""Page 2: Discretize, sample, histograms, indicator variograms."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import streamlit as st

from app.invalidation import (
    commit_sampling_fingerprint,
    make_sampling_fingerprint,
    set_current_sampling_fingerprint,
)
from app.state import init_session_state, sampling_ready, workflow_ready
from monami.geostats import (
    indicator_variograms_by_direction,
    resolve_variogram_directions,
    top_categories_by_frequency,
)
from monami.io import extract_slice
from monami.sampling import sample_fraction_to_strata, stratified_sample_dataframe
from monami.transform import (
    categorize_slice,
    compute_thresholds,
    parse_threshold_text,
    thresholds_to_text,
)
from monami.viz import (
    heatmap_slice,
    histogram_categorical,
    histogram_continuous_with_thresholds,
    sample_overlay_on_slice,
    sample_scatter,
    variogram_plot,
)

init_session_state()

st.title("2. Sampling and geostatistics")

if sampling_ready():
    st.info(
        "**Samples ready.** Default: 3 categories, ~5% density (or your last run). "
        "Change settings and click **Discretize and sample** to refresh."
    )
elif st.session_state.samples_df is not None:
    st.warning(
        "Sampling settings changed since the last run. "
        "Click **Discretize and sample** to refresh, then re-train."
    )

if not workflow_ready():
    st.warning("Load and confirm a slice on the **Data** page first.")
    st.stop()

slice_2d = st.session_state.slice_2d
continuous = st.session_state.continuous_slice if st.session_state.continuous_slice is not None else slice_2d
v_data = continuous.ravel()
data_min, data_max = float(v_data.min()), float(v_data.max())

st.subheader("Categorization thresholds")
st.caption(
    "Define how continuous values are split into categories. "
    "Changing thresholds updates the preview histogram below and the sampled category histogram after discretization."
)

method_labels = {
    "quantile": "Quantile (equal count per category)",
    "equal_width": "Equal width (even spacing on value axis)",
    "custom": "Custom thresholds (edit values below)",
}
method = st.radio(
    "Threshold method",
    options=list(method_labels.keys()),
    format_func=lambda k: method_labels[k],
    horizontal=False,
    index=list(method_labels.keys()).index(st.session_state.categorization_method),
)
st.session_state.categorization_method = method

grid_height, grid_width = slice_2d.shape
total_pixels = grid_height * grid_width

col1, col2 = st.columns(2)
with col1:
    categories = st.number_input(
        "Number of categories",
        min_value=2,
        max_value=100,
        value=st.session_state.categories,
        disabled=(method == "custom"),
    )
with col2:
    seed = st.number_input("Random seed", min_value=0, value=st.session_state.random_seed)

st.subheader("Sampling density")
sample_pct = st.slider(
    "Sample percentage of exhaustive image",
    min_value=0.1,
    max_value=100.0,
    value=float(st.session_state.sample_pct),
    step=0.1,
    help=(
        f"Fraction of the full {grid_height}×{grid_width} grid ({total_pixels:,} cells) "
        "to approximate with stratified random samples. The app picks horizontal and vertical "
        "strata counts to match this percentage while preserving the slice aspect ratio."
    ),
)
st.session_state.sample_pct = float(sample_pct)
n_h, n_v, n_samples = sample_fraction_to_strata(grid_height, grid_width, sample_pct / 100.0)
actual_pct = 100.0 * n_samples / total_pixels
st.caption(
    f"Stratified grid: **{n_h} × {n_v} = {n_samples:,}** samples "
    f"({actual_pct:.2f}% of the exhaustive image)."
)

range_col1, range_col2 = st.columns(2)
with range_col1:
    vmin = st.number_input("Range minimum", value=data_min, format="%.6f")
with range_col2:
    vmax = st.number_input("Range maximum", value=data_max, format="%.6f")

if method != "custom":
    auto_edges = compute_thresholds(
        v_data,
        int(categories),
        method=method,
        vmin=vmin,
        vmax=vmax,
    )
    config_key = (method, int(categories), float(vmin), float(vmax))
    if st.session_state.get("_threshold_config_key") != config_key:
        st.session_state.category_threshold_text = thresholds_to_text(auto_edges)
        st.session_state._threshold_config_key = config_key
elif not st.session_state.category_threshold_text:
    st.session_state.category_threshold_text = thresholds_to_text(
        compute_thresholds(v_data, int(st.session_state.categories), method="quantile")
    )

threshold_text = st.text_area(
    "Category thresholds (comma-separated bin edges)",
    value=st.session_state.category_threshold_text,
    help="Use n+1 values to define n categories. Example: 0, 0.05, 0.1, 0.2",
)
st.session_state.category_threshold_text = threshold_text

threshold_error = None
try:
    preview_edges = parse_threshold_text(threshold_text, v_data)
    n_effective = len(preview_edges) - 1
except ValueError as exc:
    threshold_error = str(exc)
    preview_edges = compute_thresholds(v_data, int(categories), method="quantile")
    n_effective = len(preview_edges) - 1

if threshold_error:
    st.error(threshold_error)
else:
    st.session_state.category_thresholds = preview_edges
    st.session_state.n_categories_effective = n_effective
    st.session_state.categories = n_effective
    st.info(f"{n_effective} categories defined by {len(preview_edges)} threshold values.")

st.plotly_chart(
    histogram_continuous_with_thresholds(
        v_data,
        preview_edges,
        title="Continuous slice with categorization thresholds",
    ),
    use_container_width=True,
)

st.subheader("Training image (optional)")
st.caption(
    "Use another Z-slice as auxiliary **DNN training labels** only. "
    "TI is categorized with the **same thresholds** as the target slice, sampled at the "
    "same density, and is **not** hard-pinned on target prediction/simulation maps. "
    "Corrected SIS ignores the training image."
)
meta = st.session_state.grid_meta
nz = int(getattr(meta, "nz", 1) or 1)
target_level = int(st.session_state.selected_level)
use_ti = st.checkbox(
    "Enable training image",
    value=bool(st.session_state.get("use_training_image", False)),
    key="use_training_image_checkbox",
)
st.session_state.use_training_image = bool(use_ti)
ti_level_value = st.session_state.get("ti_level")
if ti_level_value is None or int(ti_level_value) == target_level:
    ti_level_value = 1 if target_level == 0 and nz > 1 else 0
ti_level_value = int(np.clip(int(ti_level_value), 0, max(nz - 1, 0)))
if use_ti:
    if nz <= 1:
        st.warning("Volume has only one Z level; a separate training-image slice is unavailable.")
        st.session_state.use_training_image = False
        use_ti = False
        ti_level = None
    else:
        ti_level = st.number_input(
            "Training-image Z index",
            min_value=0,
            max_value=nz - 1,
            value=ti_level_value,
            step=1,
            help=f"Must differ from the target slice level ({target_level}).",
            key="ti_level_input",
        )
        ti_level = int(ti_level)
        if ti_level == target_level:
            st.error(
                f"Training-image level must differ from the target slice ({target_level})."
            )
        st.session_state.ti_level = ti_level
else:
    ti_level = None
    st.session_state.ti_level = None

st.session_state.random_seed = int(seed)
st.session_state._sampling_vmin = float(vmin)
st.session_state._sampling_vmax = float(vmax)

ti_enabled_for_fp = bool(st.session_state.use_training_image) and ti_level is not None
set_current_sampling_fingerprint(
    make_sampling_fingerprint(
        selected_level=int(st.session_state.selected_level),
        slice_shape=tuple(slice_2d.shape),
        source_name=str(st.session_state.source_name),
        method=method,
        categories=int(n_effective),
        sample_pct=float(sample_pct),
        seed=int(seed),
        threshold_text=threshold_text,
        vmin=float(vmin),
        vmax=float(vmax),
        use_training_image=ti_enabled_for_fp,
        ti_level=ti_level if ti_enabled_for_fp else None,
    )
)

ti_level_invalid = (
    bool(st.session_state.use_training_image)
    and (ti_level is None or int(ti_level) == target_level or nz <= 1)
)
if st.button(
    "Discretize and sample",
    type="primary",
    disabled=threshold_error is not None or ti_level_invalid,
):
    with st.spinner("Categorizing and sampling..."):
        _, categorized, edges = categorize_slice(
            continuous,
            n_categories=n_effective,
            method=method if method != "custom" else "quantile",
            bin_edges=preview_edges,
        )
        samples = stratified_sample_dataframe(categorized, int(n_h), int(n_v), seed=int(seed))
        st.session_state.categorized_2d = categorized
        st.session_state.samples_df = samples
        st.session_state.category_thresholds = edges
        st.session_state.n_categories_effective = len(edges) - 1
        st.session_state.categories = len(edges) - 1

        ti_msg = ""
        if st.session_state.use_training_image and ti_level is not None:
            volume = st.session_state.volume_3d
            ti_continuous = extract_slice(volume, int(ti_level), meta)
            _, ti_categorized, _ = categorize_slice(
                ti_continuous,
                n_categories=len(edges) - 1,
                method="quantile",
                bin_edges=edges,
            )
            ti_samples = stratified_sample_dataframe(
                ti_categorized, int(n_h), int(n_v), seed=int(seed) + 17
            )
            st.session_state.ti_categorized_2d = ti_categorized
            st.session_state.ti_samples_df = ti_samples
            st.session_state.ti_level = int(ti_level)
            ti_msg = (
                f" Training image Z={int(ti_level)}: {len(ti_samples):,} aux DNN labels."
            )
        else:
            st.session_state.use_training_image = False
            st.session_state.ti_level = None
            st.session_state.ti_samples_df = None
            st.session_state.ti_categorized_2d = None

        commit_sampling_fingerprint(
            make_sampling_fingerprint(
                selected_level=int(st.session_state.selected_level),
                slice_shape=tuple(categorized.shape),
                source_name=str(st.session_state.source_name),
                method=method,
                categories=int(len(edges) - 1),
                sample_pct=float(sample_pct),
                seed=int(seed),
                threshold_text=threshold_text,
                vmin=float(vmin),
                vmax=float(vmax),
                use_training_image=bool(st.session_state.ti_samples_df is not None),
                ti_level=st.session_state.ti_level,
            )
        )
    st.success(
        f"Created {len(samples)} target samples across {len(edges) - 1} categories."
        + ti_msg
    )
    st.rerun()

if sampling_ready():
    categorized = st.session_state.categorized_2d
    samples = st.session_state.samples_df
    n_cat = int(st.session_state.n_categories_effective or st.session_state.categories)
    vrange = (0, max(n_cat - 1, 0))
    stored_edges = st.session_state.category_thresholds or preview_edges

    tab1, tab2, tab3, tab4 = st.tabs(["Maps", "Histogram", "Variograms", "Export"])

    with tab1:
        # Side-by-side panels share width/margins/colorbar so map areas match.
        _pair_width = 420
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                heatmap_slice(
                    categorized,
                    title="Categorized slice",
                    max_width=_pair_width,
                    n_categories=n_cat,
                ),
                use_container_width=False,
            )
        with c2:
            st.plotly_chart(
                sample_scatter(
                    samples,
                    title="Sample locations",
                    grid_shape=categorized.shape,
                    max_width=_pair_width,
                    n_categories=n_cat,
                ),
                use_container_width=False,
            )
        st.plotly_chart(
            sample_overlay_on_slice(
                categorized,
                samples,
                title="Samples overlaid on categorized slice",
                n_categories=n_cat,
            ),
            use_container_width=False,
        )
        st.caption(f"Thresholds used: `{thresholds_to_text(stored_edges)}`")

        ti_cat = st.session_state.get("ti_categorized_2d")
        ti_samples = st.session_state.get("ti_samples_df")
        if ti_cat is not None and ti_samples is not None:
            st.markdown(
                f"#### Training image (Z = {st.session_state.get('ti_level')}) — "
                f"{len(ti_samples):,} aux labels"
            )
            t1, t2 = st.columns(2)
            with t1:
                st.plotly_chart(
                    heatmap_slice(
                        ti_cat,
                        title="TI categorized slice",
                        max_width=_pair_width,
                        n_categories=n_cat,
                    ),
                    use_container_width=False,
                )
            with t2:
                st.plotly_chart(
                    sample_scatter(
                        ti_samples,
                        title="TI sample locations",
                        grid_shape=ti_cat.shape,
                        max_width=_pair_width,
                        n_categories=n_cat,
                    ),
                    use_container_width=False,
                )

    with tab2:
        st.plotly_chart(
            histogram_continuous_with_thresholds(
                v_data,
                stored_edges,
                title="Continuous slice with applied thresholds",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            histogram_categorical(
                samples["V"],
                title="Sample category histogram (after discretization)",
                vrange=vrange,
                nbins=max(n_cat, 1),
            ),
            use_container_width=True,
        )

    with tab3:
        st.markdown(
            "Indicator variograms show spatial correlation of **presence** of each category "
            "(standard approach for categorical data). Point pairs are filtered by direction "
            "using scikit-gstat (East = 0°, clockwise toward +Y)."
        )
        top_cats = top_categories_by_frequency(samples, n=min(5, n_cat))
        selected = st.multiselect(
            "Categories to plot",
            options=sorted(samples["V"].unique().astype(int).tolist()),
            default=top_cats,
        )
        direction_labels = {
            "x": "X (horizontal, 0°)",
            "y": "Y (vertical, 90°)",
            "random": "Random direction",
        }
        selected_dirs = st.multiselect(
            "Variogram directions",
            options=list(direction_labels.keys()),
            default=["x", "y"],
            format_func=lambda k: direction_labels[k],
            help=(
                "Compute separate directional variograms along each orientation. "
                "**Random** draws one azimuth using the random seed (reproducible across runs)."
            ),
        )
        if not selected_dirs:
            st.info("Select at least one variogram direction.")
        elif selected and st.button("Compute variograms"):
            with st.spinner("Computing indicator variograms..."):
                directions = resolve_variogram_directions(
                    selected_dirs,
                    seed=int(st.session_state.random_seed),
                )
                variograms = indicator_variograms_by_direction(
                    samples,
                    categories=[int(c) for c in selected],
                    directions=directions,
                )
            dir_summary = ", ".join(label for label, _ in directions)
            st.plotly_chart(
                variogram_plot(
                    variograms,
                    title=f"Directional indicator variograms ({dir_summary})",
                ),
                use_container_width=True,
            )
            st.caption(f"Directions used: {dir_summary}. Angular tolerance: ±22.5°.")

    with tab4:
        project_root = Path(__file__).resolve().parents[2]
        out_dir = project_root / "2_samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"sample_all_{n_cat}_{len(samples)}.csv"
        out_path = out_dir / fname
        samples.to_csv(out_path, index=False)
        st.download_button(
            "Download sample CSV",
            data=samples.to_csv(index=False),
            file_name=fname,
            mime="text/csv",
        )
        st.caption(f"Also saved to `{out_path}`")
