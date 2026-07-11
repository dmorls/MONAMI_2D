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
        title_text="Column (X)" if col == 1 else "",
        constrain="domain",
        row=row,
        col=col,
    )
    fig.update_yaxes(
        range=[n_rows + 0.5, 0.5],
        title_text="Row (Y)" if col == 1 else "",
        autorange="reversed",
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
) -> go.Figure:
    """Lock heatmap/scatter spatial axes so cells stay square when rendered."""
    width, height, scaleratio = _grid_layout_dims(n_rows, n_cols)
    if row is not None and col is not None:
        _apply_subplot_grid_aspect(fig, n_rows, n_cols, row, col, n_cols_panels=n_cols_panels)
    else:
        xaxis = dict(constrain="domain", range=[0.5, n_cols + 0.5])
        yaxis = dict(
            autorange="reversed",
            scaleanchor="x",
            scaleratio=scaleratio,
            constrain="domain",
            range=[n_rows + 0.5, 0.5],
        )
        fig.update_layout(width=width, height=height, autosize=False, xaxis=xaxis, yaxis=yaxis)
    return fig


def heatmap_slice(
    array_2d: np.ndarray,
    title: str = "2D slice",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    colorscale: str = "Jet",
) -> go.Figure:
    """Interactive heatmap of a 2D array."""
    n_rows, n_cols = array_2d.shape
    fig = go.Figure(
        data=go.Heatmap(
            z=array_2d,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Value"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Column (X)",
        yaxis_title="Row (Y)",
    )
    return _apply_grid_aspect(fig, n_rows, n_cols)


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
) -> go.Figure:
    """Scatter plot of sample points colored by value."""
    fig = px.scatter(
        samples_df,
        x="X",
        y="Y",
        color=value_col,
        color_continuous_scale="Jet",
        title=title,
        size_max=size,
    )
    fig.update_traces(marker=dict(size=size))
    if grid_shape is not None:
        n_rows, n_cols = grid_shape
    else:
        n_rows = int(samples_df["Y"].max()) if len(samples_df) else 1
        n_cols = int(samples_df["X"].max()) if len(samples_df) else 1
    fig.update_xaxes(range=[0.5, n_cols + 0.5])
    fig.update_yaxes(range=[n_rows + 0.5, 0.5])
    return _apply_grid_aspect(fig, n_rows, n_cols)


def sample_overlay_on_slice(
    slice_2d: np.ndarray,
    samples_df: pd.DataFrame,
    title: str = "Samples on slice",
) -> go.Figure:
    """Heatmap with sample points overlaid."""
    fig = heatmap_slice(slice_2d, title=title)
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
) -> go.Figure:
    """Side-by-side truth and prediction maps."""
    vmin = min(truth.min(), prediction.min())
    vmax = max(truth.max(), prediction.max())
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
                colorscale="Jet",
                zmin=vmin,
                zmax=vmax,
                showscale=(col == 2),
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
) -> go.Figure:
    """Three-panel map: exhaustive field, colored samples, and prediction."""
    vmin = min(truth.min(), prediction.min(), samples_df["V"].min())
    vmax = max(truth.max(), prediction.max(), samples_df["V"].max())
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
            colorscale="Jet",
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
                colorscale="Jet",
                cmin=vmin,
                cmax=vmax,
                line=dict(width=0.5, color="black"),
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
            colorscale="Jet",
            zmin=vmin,
            zmax=vmax,
            showscale=True,
            colorbar=dict(title="Value"),
            name="Prediction",
        ),
        row=1,
        col=3,
    )
    for col in (1, 2, 3):
        _apply_subplot_grid_aspect(
            fig, n_rows, n_cols, 1, col, n_cols_panels=3, panel_max_width=panel_width
        )
    _, panel_height, _ = _grid_layout_dims(n_rows, n_cols, max_width=panel_width)
    fig.update_layout(title=title, width=panel_width * 3 + 90, height=panel_height + 90, autosize=False)
    return fig


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
