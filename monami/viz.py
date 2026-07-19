"""Plotly visualization helpers for Streamlit and exports."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

SPATIAL_MAX_WIDTH = 650

# Shared colorbar geometry so side-by-side spatial plots keep equal map panels.
_SPATIAL_COLORBAR = dict(title="Value", thickness=14, len=0.8, x=1.02, xpad=4)
_SPATIAL_MARGIN = dict(l=60, r=70, t=50, b=50)

# Qualitative palette for discrete categories (cycled if n is large).
_CATEGORY_PALETTE = (
    list(px.colors.qualitative.Plotly)
    + list(px.colors.qualitative.Dark24)
    + list(px.colors.qualitative.Safe)
)


def categorical_colorscale(n_categories: int) -> list:
    """Piecewise-constant Plotly colorscale with one color per category."""
    n = max(int(n_categories), 1)
    scale: list = []
    for i in range(n):
        color = _CATEGORY_PALETTE[i % len(_CATEGORY_PALETTE)]
        lo = i / n
        hi = (i + 1) / n
        scale.append([lo, color])
        scale.append([hi, color])
    return scale


def category_color_bounds(n_categories: int) -> tuple[float, float]:
    """zmin/zmax so integer categories sit in discrete color bins."""
    n = max(int(n_categories), 1)
    return -0.5, n - 0.5


def category_colorbar(n_categories: int, title: str = "Category") -> dict:
    n = max(int(n_categories), 1)
    ticks = list(range(n))
    bar = dict(_SPATIAL_COLORBAR)
    bar.update(
        title=title,
        tickmode="array",
        tickvals=ticks,
        ticktext=[str(t) for t in ticks],
    )
    return bar


def _resolve_category_style(
    n_categories: Optional[int],
    zmin: Optional[float],
    zmax: Optional[float],
    colorscale: str = "Jet",
):
    """Return (colorscale, zmin, zmax, colorbar) for continuous or discrete maps."""
    if n_categories is not None and int(n_categories) > 0:
        n = int(n_categories)
        cmin, cmax = category_color_bounds(n)
        return categorical_colorscale(n), cmin, cmax, category_colorbar(n)
    return colorscale, zmin, zmax, dict(_SPATIAL_COLORBAR)


def _grid_layout_dims(n_rows: int, n_cols: int, max_width: int = SPATIAL_MAX_WIDTH) -> tuple[int, int, float]:
    """Return fixed width, height, and y/x scale ratio for square grid cells."""
    n_cols = max(int(n_cols), 1)
    n_rows = max(int(n_rows), 1)
    scaleratio = n_rows / n_cols
    width = max_width
    height = int(max(180, min(900, width * scaleratio)))
    return width, height, scaleratio


def _subplot_axis_ref(row: int, col: int, n_cols_panels: int) -> tuple[str, str]:
    """Plotly axis names for a subplot position in a single-row grid."""
    axis_idx = (row - 1) * n_cols_panels + col
    x_ref = "x" if axis_idx == 1 else f"x{axis_idx}"
    y_ref = "y" if axis_idx == 1 else f"y{axis_idx}"
    return x_ref, y_ref


def _apply_subplot_grid_aspect(
    fig: go.Figure,
    n_rows: int,
    n_cols: int,
    row: int,
    col: int,
    n_cols_panels: int = 1,
    panel_max_width: int = 280,
) -> tuple[int, int]:
    """Apply matching square-cell spatial axes to one subplot."""
    _, panel_height, scaleratio = _grid_layout_dims(n_rows, n_cols, max_width=panel_max_width)
    x_ref, _ = _subplot_axis_ref(row, col, n_cols_panels)
    fig.update_xaxes(
        range=[0.5, n_cols + 0.5],
        autorange=False,
        title_text="Column (X)" if col == 1 else "",
        constrain="domain",
        row=row,
        col=col,
    )
    # Fixed high→low range (no autorange="reversed"): scatter subplots otherwise
    # ignore range and pad to nice ticks, shrinking the panel vs heatmaps.
    fig.update_yaxes(
        range=[n_rows + 0.5, 0.5],
        autorange=False,
        title_text="Row (Y)" if col == 1 else "",
        scaleanchor=x_ref,
        scaleratio=scaleratio,
        constrain="domain",
        row=row,
        col=col,
    )
    return panel_max_width, panel_height


def _apply_grid_aspect(
    fig: go.Figure,
    n_rows: int,
    n_cols: int,
    row: Optional[int] = None,
    col: Optional[int] = None,
    n_cols_panels: int = 1,
    max_width: int = SPATIAL_MAX_WIDTH,
) -> go.Figure:
    """Lock heatmap/scatter spatial axes so cells stay square when rendered."""
    width, height, scaleratio = _grid_layout_dims(n_rows, n_cols, max_width=max_width)
    if row is not None and col is not None:
        _apply_subplot_grid_aspect(fig, n_rows, n_cols, row, col, n_cols_panels=n_cols_panels)
    else:
        xaxis = dict(
            constrain="domain",
            range=[0.5, n_cols + 0.5],
            autorange=False,
            domain=[0.0, 1.0],
        )
        yaxis = dict(
            autorange=False,
            scaleanchor="x",
            scaleratio=scaleratio,
            constrain="domain",
            range=[n_rows + 0.5, 0.5],
            domain=[0.0, 1.0],
        )
        fig.update_layout(
            width=width,
            height=height,
            autosize=False,
            margin=_SPATIAL_MARGIN,
            xaxis=xaxis,
            yaxis=yaxis,
        )
    return fig


def heatmap_slice(
    array_2d: np.ndarray,
    title: str = "2D slice",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    colorscale: str = "Jet",
    max_width: int = SPATIAL_MAX_WIDTH,
    n_categories: Optional[int] = None,
) -> go.Figure:
    """Interactive heatmap of a 2D array (discrete categories when ``n_categories`` set)."""
    n_rows, n_cols = array_2d.shape
    cs, z0, z1, cbar = _resolve_category_style(n_categories, zmin, zmax, colorscale)
    fig = go.Figure(
        data=go.Heatmap(
            z=array_2d,
            x=list(range(1, n_cols + 1)),
            y=list(range(1, n_rows + 1)),
            colorscale=cs,
            zmin=z0,
            zmax=z1,
            colorbar=cbar,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Column (X)",
        yaxis_title="Row (Y)",
    )
    return _apply_grid_aspect(fig, n_rows, n_cols, max_width=max_width)


def slice_gallery(volume: np.ndarray, levels: List[int], title_prefix: str = "Level") -> go.Figure:
    """Small multiples of selected levels."""
    n = len(levels)
    panel_cols = min(3, n)
    panel_rows = (n + panel_cols - 1) // panel_cols
    slice_rows, slice_cols = volume[levels[0]].shape
    _, panel_height, scaleratio = _grid_layout_dims(slice_rows, slice_cols, max_width=320)

    fig = make_subplots(
        rows=panel_rows,
        cols=panel_cols,
        subplot_titles=[f"{title_prefix} {lv}" for lv in levels],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )
    for idx, level in enumerate(levels):
        r, c = divmod(idx, panel_cols)
        fig.add_trace(
            go.Heatmap(z=volume[level], colorscale="Jet", showscale=(idx == 0)),
            row=r + 1,
            col=c + 1,
        )
        _apply_grid_aspect(fig, slice_rows, slice_cols, row=r + 1, col=c + 1)

    fig.update_layout(
        title="Slice gallery",
        width=min(SPATIAL_MAX_WIDTH, 340 * panel_cols),
        height=max(220, panel_height * panel_rows + 80),
        autosize=False,
    )
    return fig


def histogram_categorical(
    values,
    title: str = "Histogram",
    nbins: Optional[int] = None,
    vrange: Optional[tuple] = None,
) -> go.Figure:
    """Histogram for categorical or continuous values."""
    fig = px.histogram(
        x=np.asarray(values).ravel(),
        nbins=nbins,
        range_x=vrange,
        title=title,
        labels={"x": "Value"},
    )
    fig.update_layout(bargap=0.05, height=400)
    return fig


def histogram_continuous_with_thresholds(
    values,
    thresholds: Sequence[float],
    title: str = "Continuous values and category thresholds",
    nbins: int = 40,
) -> go.Figure:
    """Histogram of continuous data with vertical lines at categorization thresholds."""
    data = np.asarray(values).ravel()
    counts, _ = np.histogram(data, bins=nbins)
    ymax = float(counts.max()) if counts.size else 1.0
    if ymax <= 0:
        ymax = 1.0

    fig = px.histogram(x=data, nbins=nbins, title=title, labels={"x": "Continuous value"})
    for i, edge in enumerate(thresholds):
        fig.add_vline(
            x=float(edge),
            line_dash="dash",
            line_color="red" if i in (0, len(thresholds) - 1) else "orange",
            annotation_text=f"{float(edge):.4g}",
            annotation_position="top",
        )
    fig.update_layout(bargap=0.02, height=420, yaxis_range=[0, ymax * 1.15])
    return fig


def sample_scatter(
    samples_df: pd.DataFrame,
    value_col: str = "V",
    title: str = "Sample locations",
    size: int = 8,
    grid_shape: Optional[tuple[int, int]] = None,
    max_width: int = SPATIAL_MAX_WIDTH,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    n_categories: Optional[int] = None,
) -> go.Figure:
    """Scatter plot of sample points colored by value (matched layout to heatmaps)."""
    if grid_shape is not None:
        n_rows, n_cols = grid_shape
    else:
        n_rows = int(samples_df["Y"].max()) if len(samples_df) else 1
        n_cols = int(samples_df["X"].max()) if len(samples_df) else 1
    values = samples_df[value_col]
    cs, cmin, cmax, cbar = _resolve_category_style(
        n_categories,
        float(values.min()) if zmin is None else float(zmin),
        float(values.max()) if zmax is None else float(zmax),
    )
    fig = go.Figure(
        data=go.Scatter(
            x=samples_df["X"],
            y=samples_df["Y"],
            mode="markers",
            marker=dict(
                size=size,
                color=values,
                colorscale=cs,
                cmin=cmin,
                cmax=cmax,
                colorbar=cbar,
                line=dict(width=0.5, color="black"),
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Column (X)",
        yaxis_title="Row (Y)",
    )
    return _apply_grid_aspect(fig, n_rows, n_cols, max_width=max_width)


def sample_overlay_on_slice(
    slice_2d: np.ndarray,
    samples_df: pd.DataFrame,
    title: str = "Samples on slice",
    n_categories: Optional[int] = None,
) -> go.Figure:
    """Heatmap with sample points overlaid."""
    fig = heatmap_slice(slice_2d, title=title, n_categories=n_categories)
    fig.add_trace(
        go.Scatter(
            x=samples_df["X"] - 1,
            y=samples_df["Y"] - 1,
            mode="markers+text",
            marker=dict(color="white", size=8, line=dict(color="black", width=1)),
            text=samples_df["V"].astype(int).astype(str),
            textposition="middle center",
            name="Samples",
        )
    )
    return fig


def variogram_plot(
    variograms: Dict[int, tuple[np.ndarray, np.ndarray]] | Dict[str, Dict[int, tuple[np.ndarray, np.ndarray]]],
    title: str = "Indicator variograms",
) -> go.Figure:
    """Plot indicator variograms (single or multi-direction)."""
    fig = go.Figure()
    if variograms and isinstance(next(iter(variograms.values())), dict):
        for dir_label, by_category in variograms.items():
            for category, (lags, gamma) in by_category.items():
                fig.add_trace(
                    go.Scatter(
                        x=lags,
                        y=gamma,
                        mode="lines+markers",
                        name=f"{dir_label} · cat {category}",
                    )
                )
    else:
        for category, (lags, gamma) in variograms.items():
            fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=gamma,
                    mode="lines+markers",
                    name=f"Category {category}",
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Lag distance",
        yaxis_title="Semivariance",
        height=450,
    )
    return fig


def training_history_plot(history, title: str = "Training history") -> go.Figure:
    """Accuracy and loss curves from Keras history."""
    return training_history_live_plot(history.history, title=title)


def training_history_live_plot(
    metrics: Dict[str, List[float]],
    title: str = "Training history",
    current_epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
) -> go.Figure:
    """Plot running train/validation accuracy and loss curves."""
    acc = metrics.get("accuracy", [])
    val_acc = metrics.get("val_accuracy", [])
    loss = metrics.get("loss", [])
    val_loss = metrics.get("val_loss", [])

    if title == "Training history" and current_epoch is not None and total_epochs is not None:
        title = f"Training history — epoch {current_epoch}/{total_epochs}"

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy", "Loss"])
    if acc:
        fig.add_trace(go.Scatter(x=list(range(1, len(acc) + 1)), y=acc, name="Train acc", mode="lines+markers"), row=1, col=1)
    if val_acc:
        fig.add_trace(go.Scatter(x=list(range(1, len(val_acc) + 1)), y=val_acc, name="Val acc", mode="lines+markers"), row=1, col=1)
    if loss:
        fig.add_trace(go.Scatter(x=list(range(1, len(loss) + 1)), y=loss, name="Train loss", mode="lines+markers"), row=1, col=2)
    if val_loss:
        fig.add_trace(go.Scatter(x=list(range(1, len(val_loss) + 1)), y=val_loss, name="Val loss", mode="lines+markers"), row=1, col=2)
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_layout(title=title, height=420, showlegend=True)
    return fig


def comparison_heatmaps(
    truth: np.ndarray,
    prediction: np.ndarray,
    title: str = "Truth vs prediction",
    n_categories: Optional[int] = None,
) -> go.Figure:
    """Side-by-side truth and prediction maps."""
    if n_categories is None:
        n_categories = int(max(truth.max(), prediction.max())) + 1
    cs, vmin, vmax, cbar = _resolve_category_style(n_categories, None, None)
    n_rows, n_cols = truth.shape
    x_coords = list(range(1, n_cols + 1))
    y_coords = list(range(1, n_rows + 1))
    panel_width = 320
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Truth (categorized)", "Prediction"])
    for col, data, name in [(1, truth, "Truth"), (2, prediction, "Prediction")]:
        fig.add_trace(
            go.Heatmap(
                z=data,
                x=x_coords,
                y=y_coords,
                colorscale=cs,
                zmin=vmin,
                zmax=vmax,
                showscale=(col == 2),
                colorbar=cbar if col == 2 else None,
                name=name,
            ),
            row=1,
            col=col,
        )
        _apply_subplot_grid_aspect(fig, n_rows, n_cols, 1, col, n_cols_panels=2, panel_max_width=panel_width)
    _, panel_height, _ = _grid_layout_dims(n_rows, n_cols, max_width=panel_width)
    fig.update_layout(title=title, width=panel_width * 2 + 60, height=panel_height + 80, autosize=False)
    return fig


def exhaustive_sample_prediction_maps(
    truth: np.ndarray,
    samples_df: pd.DataFrame,
    prediction: np.ndarray,
    title: str = "Exhaustive vs samples vs prediction",
    n_categories: Optional[int] = None,
) -> go.Figure:
    """Three-panel map: exhaustive field, colored samples, and prediction.

    All panels share the same axis ranges, panel size, and discrete category colors.
    """
    if n_categories is None:
        n_categories = int(
            max(truth.max(), prediction.max(), samples_df["V"].max())
        ) + 1
    cs, vmin, vmax, cbar = _resolve_category_style(int(n_categories), None, None)
    n_rows, n_cols = truth.shape
    x_coords = list(range(1, n_cols + 1))
    y_coords = list(range(1, n_rows + 1))
    panel_width = 280
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Exhaustive (truth)", "Samples", "Prediction"],
        horizontal_spacing=0.06,
    )
    fig.add_trace(
        go.Heatmap(
            z=truth,
            x=x_coords,
            y=y_coords,
            colorscale=cs,
            zmin=vmin,
            zmax=vmax,
            showscale=False,
            name="Exhaustive",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=samples_df["X"],
            y=samples_df["Y"],
            mode="markers",
            marker=dict(
                size=9,
                color=samples_df["V"],
                colorscale=cs,
                cmin=vmin,
                cmax=vmax,
                line=dict(width=0.5, color="black"),
                showscale=False,
            ),
            name="Samples",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=prediction,
            x=x_coords,
            y=y_coords,
            colorscale=cs,
            zmin=vmin,
            zmax=vmax,
            showscale=True,
            colorbar=cbar,
            name="Prediction",
        ),
        row=1,
        col=3,
    )
    for col in (1, 2, 3):
        _apply_subplot_grid_aspect(
            fig, n_rows, n_cols, 1, col, n_cols_panels=3, panel_max_width=panel_width
        )
        fig.update_xaxes(range=[0.5, n_cols + 0.5], autorange=False, row=1, col=col)
        fig.update_yaxes(range=[n_rows + 0.5, 0.5], autorange=False, row=1, col=col)
    _, panel_height, _ = _grid_layout_dims(n_rows, n_cols, max_width=panel_width)
    fig.update_layout(title=title, width=panel_width * 3 + 90, height=panel_height + 90, autosize=False)
    return fig


def _jet_rgb(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    """Approximate Plotly Jet colormap for a normalized value."""
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (float(value) - vmin) / (vmax - vmin)))
    if t < 0.125:
        r, g, b = 0, 0, 0.5 + 4.0 * t
    elif t < 0.375:
        r, g, b = 0, 4.0 * (t - 0.125), 1.0
    elif t < 0.625:
        r, g, b = 4.0 * (t - 0.375), 1.0, 1.0 - 4.0 * (t - 0.375)
    elif t < 0.875:
        r, g, b = 1.0, 1.0 - 4.0 * (t - 0.625), 0
    else:
        r, g, b = 1.0 - 4.0 * (t - 0.875), 0, 0
    return int(r * 255), int(g * 255), int(b * 255)


def _array_to_rgb_image(array: np.ndarray, vmin: float, vmax: float, scale: int = 6) -> "Image.Image":
    from PIL import Image

    rows, cols = array.shape
    pixels = np.zeros((rows, cols, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            pixels[r, c] = _jet_rgb(array[r, c], vmin, vmax)
    img = Image.fromarray(pixels, mode="RGB")
    if scale > 1:
        try:
            resample = Image.Resampling.NEAREST
        except AttributeError:
            resample = Image.NEAREST
        img = img.resize((cols * scale, rows * scale), resample)
    return img


def preview_maps_to_png(
    truth: np.ndarray,
    samples_df: pd.DataFrame,
    prediction: np.ndarray,
    epoch: int,
    total_epochs: int,
    n_categories: Optional[int] = None,
) -> bytes:
    """Rasterize the three-panel preview figure to PNG bytes."""
    import io

    from PIL import Image, ImageDraw

    if n_categories is None:
        n_categories = int(max(truth.max(), prediction.max(), samples_df["V"].max())) + 1
    fig = exhaustive_sample_prediction_maps(
        truth,
        samples_df,
        prediction,
        title=f"Exhaustive / samples / prediction — epoch {epoch}/{total_epochs}",
        n_categories=int(n_categories),
    )
    try:
        return fig.to_image(format="png", scale=1)
    except Exception:
        vmin = float(min(truth.min(), prediction.min(), samples_df["V"].min()))
        vmax = float(max(truth.max(), prediction.max(), samples_df["V"].max()))
        scale = 5
        truth_img = _array_to_rgb_image(truth, vmin, vmax, scale=scale)
        pred_img = _array_to_rgb_image(prediction, vmin, vmax, scale=scale)
        sample_img = truth_img.copy()
        draw = ImageDraw.Draw(sample_img)
        radius = max(2, scale // 2)
        for _, row in samples_df.iterrows():
            x = int((float(row["X"]) - 1) * scale + scale // 2)
            y = int((float(row["Y"]) - 1) * scale + scale // 2)
            color = _jet_rgb(float(row["V"]), vmin, vmax)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(0, 0, 0),
            )
        panel_h = truth_img.height
        panel_w = truth_img.width
        title_h = 28
        canvas = Image.new("RGB", (panel_w * 3 + 20, panel_h + title_h), (255, 255, 255))
        canvas.paste(truth_img, (0, title_h))
        canvas.paste(sample_img, (panel_w + 10, title_h))
        canvas.paste(pred_img, (panel_w * 2 + 20, title_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f"Epoch {epoch}/{total_epochs}", fill=(0, 0, 0))
        draw.text((0, title_h - 14), "Truth", fill=(0, 0, 0))
        draw.text((panel_w + 10, title_h - 14), "Samples", fill=(0, 0, 0))
        draw.text((panel_w * 2 + 20, title_h - 14), "Prediction", fill=(0, 0, 0))
        out = io.BytesIO()
        canvas.save(out, format="PNG")
        return out.getvalue()


def assemble_gif_from_png_frames(png_frames: Sequence[bytes], duration_ms: int = 500) -> bytes:
    """Combine PNG frames into an animated GIF."""
    import io

    from PIL import Image

    if not png_frames:
        return b""
    images = [Image.open(io.BytesIO(frame)).convert("RGB") for frame in png_frames]
    out = io.BytesIO()
    images[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return out.getvalue()


def difference_map(
    truth: np.ndarray,
    prediction: np.ndarray,
    title: str = "Mismatch map",
) -> go.Figure:
    """Absolute difference between truth and prediction."""
    diff = np.abs(truth - prediction)
    fig = heatmap_slice(diff, title=title, zmin=0, zmax=diff.max() if diff.max() > 0 else 1, colorscale="Reds")
    return fig


def overlaid_histograms(
    sample_values,
    field_values,
    title: str = "Sample vs field distribution",
    vrange: Optional[tuple] = None,
) -> go.Figure:
    """Overlaid histograms for comparison."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=np.asarray(sample_values).ravel(), name="Samples", opacity=0.6, nbinsx=20))
    fig.add_trace(go.Histogram(x=np.asarray(field_values).ravel(), name="Prediction field", opacity=0.6, nbinsx=20))
    if vrange:
        fig.update_layout(xaxis_range=list(vrange))
    fig.update_layout(barmode="overlay", title=title, height=400)
    return fig


def _category_proportions(values, categories: Sequence[int]) -> list[float]:
    arr = np.asarray(values).ravel()
    n = max(len(arr), 1)
    return [float(np.sum(arr == cat)) / n for cat in categories]


def category_proportion_comparison(
    truth: np.ndarray,
    training_samples,
    prediction: np.ndarray,
    n_categories: int,
    title: str = "Category proportions",
    prediction_label: str = "Prediction",
) -> go.Figure:
    """
    Grouped bar chart of category proportions for exhaustive truth,
    training samples, and a prediction / simulation field.

    Bars use the same discrete category colors as the spatial maps; series are
    distinguished by fill pattern. Layout is forced light for reliable PDF export.
    """
    cats = list(range(int(n_categories)))
    if isinstance(training_samples, pd.DataFrame):
        train_vals = training_samples["V"].to_numpy()
    else:
        train_vals = np.asarray(training_samples).ravel()

    truth_p = _category_proportions(truth, cats)
    train_p = _category_proportions(train_vals, cats)
    pred_p = _category_proportions(prediction, cats)
    labels = [str(c) for c in cats]
    cat_colors = [_CATEGORY_PALETTE[c % len(_CATEGORY_PALETTE)] for c in cats]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Exhaustive",
            x=labels,
            y=truth_p,
            text=[f"{p:.1%}" for p in truth_p],
            textposition="auto",
            marker=dict(color=cat_colors, line=dict(width=0.5, color="#333333")),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Training samples",
            x=labels,
            y=train_p,
            text=[f"{p:.1%}" for p in train_p],
            textposition="auto",
            marker=dict(
                color=cat_colors,
                pattern_shape="/",
                line=dict(width=0.5, color="#333333"),
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            name=prediction_label,
            x=labels,
            y=pred_p,
            text=[f"{p:.1%}" for p in pred_p],
            textposition="auto",
            marker=dict(
                color=cat_colors,
                pattern_shape="x",
                line=dict(width=0.5, color="#333333"),
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#222222"),
        barmode="group",
        title=title,
        xaxis_title="Category",
        yaxis_title="Proportion",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def confusion_matrix_plot(y_true, y_pred, title: str = "Confusion matrix") -> go.Figure:
    """Heatmap confusion matrix."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        text_auto=True,
        title=title,
        labels=dict(x="Predicted", y="Actual"),
    )
    fig.update_layout(height=450)
    return fig


def observed_vs_predicted_scatter(y_true, y_pred, title: str = "Observed vs predicted") -> go.Figure:
    """Scatter of observed vs predicted at test points."""
    fig = px.scatter(x=y_true, y=y_pred, title=title, labels={"x": "Observed", "y": "Predicted"})
    min_v = min(min(y_true), min(y_pred))
    max_v = max(max(y_true), max(y_pred))
    fig.add_trace(
        go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Perfect", line=dict(dash="dash"))
    )
    fig.update_layout(height=450)
    return fig
