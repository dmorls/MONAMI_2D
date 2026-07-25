"""Build shareable PDF results reports for the MONAMI workflow."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from monami.ml import ModelMeta
from monami.viz import (
    category_proportion_comparison,
    exhaustive_sample_prediction_maps,
    report_cover_overview_maps,
    training_history_live_plot,
    training_history_plot,
)


@dataclass
class ReportContext:
    """Plain inputs for PDF generation (no Streamlit dependency)."""

    version: str
    project_root: Path
    source_name: str
    property_name: str
    selected_level: int
    grid_meta: Any
    categories: int
    sample_pct: float
    random_seed: int
    samples_df: pd.DataFrame
    train_df: Optional[pd.DataFrame]
    test_df: Optional[pd.DataFrame]
    truth: np.ndarray
    algorithm_id: str
    algorithm_name: str
    algorithm_description: str
    algorithm_long_description: str
    algorithm_config: Dict[str, Any]
    meta: ModelMeta
    prediction_description: str = ""
    simulation_description: str = ""
    statistical_model: Dict[str, Any] = field(default_factory=dict)
    prediction_2d: Optional[np.ndarray] = None
    prediction_statistics: Optional[Dict[str, Any]] = None
    simulations: List[Dict[str, Any]] = field(default_factory=list)
    history: Any = None
    live_training_history: Optional[Dict[str, List[float]]] = None
    model_path: str = ""
    stop_criteria_summary: str = ""


def sanitize_version(version: str) -> str:
    """Make a version tag safe for filenames."""
    cleaned = re.sub(r"[^\w.\-]+", "_", str(version).strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("Version series must not be empty.")
    return cleaned


def report_filename(version: str, when: Optional[datetime] = None) -> str:
    ts = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"Monami_{sanitize_version(version)}_{ts}.pdf"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontSize=22,
            spaceAfter=12,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            parent=styles["Heading1"],
            fontSize=14,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#1a1a1a"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyLeft",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#444444"),
            spaceAfter=10,
            alignment=TA_CENTER,
        )
    )
    return styles


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _md_to_paragraphs(text: str, style) -> List[Any]:
    """Very light markdown → reportlab paragraphs (headings/bullets/plain)."""
    if not text:
        return []
    flow: List[Any] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flow.append(Spacer(1, 4))
            continue
        if line.startswith("### "):
            flow.append(Paragraph(f"<b>{_escape(line[4:].strip())}</b>", style))
        elif line.startswith("## "):
            flow.append(Paragraph(f"<b>{_escape(line[3:].strip())}</b>", style))
        elif line.startswith("# "):
            flow.append(Paragraph(f"<b>{_escape(line[2:].strip())}</b>", style))
        elif line.startswith("- ") or line.startswith("* "):
            flow.append(Paragraph(f"• {_escape(line[2:].strip())}", style))
        elif line.startswith("**") and line.endswith("**"):
            flow.append(Paragraph(f"<b>{_escape(line.strip('*').strip())}</b>", style))
        else:
            # Bold markers **...**
            html = _escape(line)
            html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html)
            flow.append(Paragraph(html, style))
    return flow


def _kv_table(rows: Sequence[Tuple[str, str]], col_widths=None) -> Table:
    data = [[Paragraph(f"<b>{_escape(k)}</b>", ParagraphStyle("k", fontSize=8)), Paragraph(_escape(v), ParagraphStyle("v", fontSize=8))] for k, v in rows]
    table = Table(data, colWidths=col_widths or [2.2 * inch, 4.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f3f3")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.lightgrey),
            ]
        )
    )
    return table


def _fig_to_png_bytes(fig, width: int = 900) -> Optional[bytes]:
    """Export a Plotly figure to PNG bytes; return None on failure."""
    # Force a light theme so Streamlit dark-mode defaults don't yield black PDFs.
    export_fig = fig
    try:
        import plotly.graph_objects as go

        export_fig = go.Figure(fig.to_dict())
        export_fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#222222"),
        )
    except Exception:
        export_fig = fig
    try:
        return export_fig.to_image(format="png", width=width, scale=1)
    except Exception:
        try:
            return export_fig.to_image(format="png", scale=1)
        except Exception:
            return None


def _image_flowable(
    png_bytes: Optional[bytes],
    max_width: float,
    caption: str,
    styles,
    max_height: Optional[float] = None,
) -> List[Any]:
    """Return image + caption kept together so captions are not orphaned on a blank page."""
    items: List[Any] = []
    if png_bytes is None:
        items.append(Paragraph(f"<i>[Figure unavailable: {_escape(caption)}]</i>", styles["BodyLeft"]))
        return items
    bio = io.BytesIO(png_bytes)
    img = Image(bio)
    # Scale to max width (and optional max height) preserving aspect
    iw, ih = float(img.imageWidth), float(img.imageHeight)
    scale = 1.0
    if iw > max_width:
        scale = min(scale, max_width / iw)
    draw_h = ih * scale
    if max_height is not None and draw_h > max_height:
        scale = min(scale, max_height / ih)
    if scale != 1.0:
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
    items.append(KeepTogether([img, Paragraph(_escape(caption), styles["Caption"])]))
    return items


def _history_metrics_dict(ctx: ReportContext) -> Optional[Dict[str, List[float]]]:
    if ctx.live_training_history and any(ctx.live_training_history.values()):
        return ctx.live_training_history
    hist = ctx.history
    if hist is None:
        return None
    if hasattr(hist, "history") and isinstance(hist.history, dict):
        return hist.history
    if isinstance(hist, dict):
        return hist
    return None


def _final_metrics_lines(metrics: Dict[str, List[float]]) -> List[Tuple[str, str]]:
    rows = []
    for key, label in (
        ("accuracy", "Final train accuracy"),
        ("val_accuracy", "Final val accuracy"),
        ("loss", "Final train loss"),
        ("val_loss", "Final val loss"),
    ):
        vals = metrics.get(key) or []
        if vals:
            rows.append((label, f"{float(vals[-1]):.4f} ({len(vals)} epochs)"))
    return rows


def _sample_statistics_rows(metrics: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    if not metrics:
        return []

    def _number(key: str) -> str:
        value = float(metrics.get(key, -1.0))
        return f"{value:.4f}" if value >= 0 else "n/a"

    return [
        (
            "Hard-data fidelity",
            f"{float(metrics.get('hard_data_fidelity', 0.0)):.1%}",
        ),
        ("Sample-proportion L1", _number("proportion_l1")),
        ("Sample-proportion RMSE", _number("proportion_rmse")),
        ("Indicator variogram RMSE", _number("variogram_rmse_mean")),
        (
            f"Directional transition error X (lag {int(metrics.get('transition_lag_x', 1))})",
            _number("transition_error_x"),
        ),
        (
            f"Directional transition error Y (lag {int(metrics.get('transition_lag_y', 1))})",
            _number("transition_error_y"),
        ),
    ]


def build_pdf_report(ctx: ReportContext, output_path: Path) -> Path:
    """Write an extensive PDF report to ``output_path`` and return the path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    story: List[Any] = []
    page_width = letter[0] - 1.2 * inch
    when = datetime.now()
    version = sanitize_version(ctx.version)

    has_pred = ctx.prediction_2d is not None
    n_sims = len(ctx.simulations or [])
    train_n = len(ctx.train_df) if ctx.train_df is not None else 0
    test_n = len(ctx.test_df) if ctx.test_df is not None else 0
    sample_n = len(ctx.samples_df) if ctx.samples_df is not None else 0
    grid_shape = tuple(ctx.truth.shape) if ctx.truth is not None else ()
    is_statistical = getattr(ctx.meta, "model_type", "keras") == "corrected_sis"

    # ----- Cover -----
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("MONAMI 2D — Results Report", styles["CoverTitle"]))
    story.append(Paragraph(f"Version series: <b>{_escape(version)}</b>", styles["BodyLeft"]))
    story.append(Paragraph(f"Generated: {_escape(when.strftime('%Y-%m-%d %H:%M:%S'))}", styles["BodyLeft"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        _kv_table(
            [
                ("Source file", ctx.source_name or "—"),
                ("Property", ctx.property_name or "—"),
                ("Slice / level", str(ctx.selected_level)),
                ("Grid shape (ny, nx)", str(grid_shape)),
                ("Categories", str(ctx.categories)),
                ("Algorithm", f"{ctx.algorithm_name} (`{ctx.algorithm_id}`)"),
                ("Most-likely prediction", "Yes" if has_pred else "Not available"),
                ("Simulations", str(n_sims) if n_sims else "None"),
                ("Model file", Path(ctx.model_path).name if ctx.model_path else "(session)"),
            ]
        )
    )

    cover_sim = None
    cover_sim_label = "Simulation"
    if n_sims:
        first_sim = ctx.simulations[0] or {}
        cover_sim = first_sim.get("grid")
        cover_sim_label = str(first_sim.get("label") or "Simulation 1")
    if ctx.truth is not None and ctx.samples_df is not None:
        try:
            fig_cover = report_cover_overview_maps(
                ctx.truth,
                ctx.samples_df,
                prediction=ctx.prediction_2d if has_pred else None,
                simulation=cover_sim,
                simulation_label=cover_sim_label,
                n_categories=int(ctx.categories),
                title="Overview",
            )
            story.append(Spacer(1, 0.12 * inch))
            story.extend(
                _image_flowable(
                    _fig_to_png_bytes(fig_cover, width=1000),
                    page_width,
                    "Cover overview: exhaustive / samples"
                    + (" / prediction" if has_pred else "")
                    + (f" / {cover_sim_label}" if cover_sim is not None else ""),
                    styles,
                )
            )
        except Exception:
            story.append(
                Paragraph("<i>[Cover overview maps unavailable]</i>", styles["BodyLeft"])
            )
    story.append(PageBreak())

    # ----- Data -----
    story.append(Paragraph("1. Data", styles["SectionHead"]))
    meta = ctx.grid_meta
    data_rows = [
        ("Source name", ctx.source_name or "—"),
        ("Property name", ctx.property_name or "—"),
        ("Selected level / slice", str(ctx.selected_level)),
        ("Grid shape", str(grid_shape)),
    ]
    if meta is not None:
        for attr, label in (
            ("nx", "nx"),
            ("ny", "ny"),
            ("nz", "nz"),
            ("property_name", "Meta property"),
        ):
            if hasattr(meta, attr):
                data_rows.append((label, str(getattr(meta, attr))))
    story.append(_kv_table(data_rows))
    story.append(
        Paragraph(
            "The exhaustive 2D slice is discretized into categories and used as the reference "
            "truth field for maps and proportion comparisons.",
            styles["BodyJust"],
        )
    )

    # ----- Sampling -----
    story.append(Paragraph("2. Sampling", styles["SectionHead"]))
    story.append(
        _kv_table(
            [
                ("Number of categories", str(ctx.categories)),
                ("Sample density (%)", f"{float(ctx.sample_pct):.2f}"),
                ("Total samples", str(sample_n)),
                ("Training samples", str(train_n)),
                ("Test / validation samples", str(test_n)),
                ("Random seed", str(ctx.random_seed)),
            ]
        )
    )
    story.append(
        Paragraph(
            (
                "Stratified sampling draws points from the categorized exhaustive field. "
                "For corrected SIS, all sampled points are hard conditioning data and the "
                "exhaustive field is withheld from model fitting and simulation."
                if is_statistical
                else
                "Stratified sampling draws points from the categorized exhaustive field. "
                "The training split is the hard-data / neighbor pool for prediction and "
                "sequential simulation."
            ),
            styles["BodyJust"],
        )
    )

    # ----- Algorithm -----
    story.append(Paragraph("3. Algorithm / method", styles["SectionHead"]))
    story.append(
        _kv_table(
            [
                ("Algorithm id", ctx.algorithm_id),
                ("Algorithm name", ctx.algorithm_name),
                ("Config", str(ctx.algorithm_config or {})),
                ("Feature dimension (meta)", str(getattr(ctx.meta, "feature_dim", "—"))),
                ("n_nearest (meta)", str(getattr(ctx.meta, "n_nearest", "—"))),
            ]
        )
    )
    story.append(Paragraph("<b>Description</b>", styles["BodyLeft"]))
    desc = ctx.algorithm_long_description or ctx.algorithm_description or ""
    story.extend(_md_to_paragraphs(desc, styles["BodyLeft"]))
    story.append(Paragraph("<b>Sequential simulation uncertainty</b>", styles["BodyLeft"]))
    story.append(
        Paragraph(
            ctx.simulation_description
            or (
                "Sequential simulation uses a random path and a Monte Carlo category "
                "draw at each unsampled cell while hard conditioning values remain fixed."
            ),
            styles["BodyJust"],
        )
    )

    # ----- Training -----
    story.append(
        Paragraph(
            "4. Statistical fitting" if is_statistical else "4. Training",
            styles["SectionHead"],
        )
    )
    m = ctx.meta
    if is_statistical:
        train_rows = [
            ("Model type", "Corrected Sequential Indicator Simulation"),
            ("Fitting data", "All sampled points; exhaustive truth excluded"),
            ("Fitting time (s)", f"{float(m.training_seconds):.2f}"),
            ("Hard sample count", str(m.train_sample_count)),
            ("Category map", str(m.class_to_idx)),
            ("Configuration", str(ctx.algorithm_config or {})),
        ]
    else:
        train_rows = [
            ("Hidden layers", str(list(m.nodes_per_layer))),
            ("Dropout", str(m.dropout)),
            ("Optimizer", str(m.optimizer)),
            ("Loss", str(m.loss_function)),
            ("Hidden activation", str(m.hidden_activation)),
            ("Output activation", str(m.out_activation)),
            ("Test ratio", str(m.test_ratio)),
            ("Training time (s)", f"{float(m.training_seconds):.2f}"),
            ("Train sample count (meta)", str(m.train_sample_count)),
            ("Neighbor / pool count (meta)", str(m.neighbor_sample_count)),
            ("XY scale", str(m.xy_scale)),
            ("Class map", str(m.class_to_idx)),
        ]
    if ctx.stop_criteria_summary:
        train_rows.insert(0, ("Stop criteria", ctx.stop_criteria_summary))
    story.append(_kv_table(train_rows))

    if is_statistical and ctx.statistical_model:
        story.append(Paragraph("<b>Fitted indicator variograms</b>", styles["BodyLeft"]))
        fitted_rows = []
        proportions = ctx.statistical_model.get("proportions", {})
        for category, values in ctx.statistical_model.get("variograms", {}).items():
            fitted_rows.append(
                (
                    f"Category {category}",
                    (
                        f"p={float(proportions.get(str(category), 0.0)):.4f}; "
                        f"{values.get('model', '—')}; "
                        f"nugget={float(values.get('nugget', 0.0)):.4g}; "
                        f"sill={float(values.get('nugget', 0.0)) + float(values.get('partial_sill', 0.0)):.4g}; "
                        f"range X/Y={float(values.get('range_x', 0.0)):.2f}/"
                        f"{float(values.get('range_y', 0.0)):.2f}; "
                        f"fit RMSE={float(values.get('fit_rmse', -1.0)):.4g}"
                    ),
                )
            )
        if fitted_rows:
            story.append(_kv_table(fitted_rows))

    metrics = _history_metrics_dict(ctx)
    if metrics:
        finals = _final_metrics_lines(metrics)
        if finals:
            story.append(Paragraph("<b>Final training metrics</b>", styles["BodyLeft"]))
            story.append(_kv_table(finals))
        try:
            if ctx.history is not None and hasattr(ctx.history, "history"):
                fig = training_history_plot(ctx.history, title="Training history")
            else:
                fig = training_history_live_plot(metrics, title="Training history")
            # Compact curves so section 4 fits on 1–2 pages without a trailing blank page.
            fig.update_layout(height=280, width=800, margin=dict(l=40, r=20, t=45, b=30))
            png = _fig_to_png_bytes(fig, width=800)
            story.extend(
                _image_flowable(
                    png,
                    page_width,
                    "Training accuracy and loss curves",
                    styles,
                    max_height=2.8 * inch,
                )
            )
        except Exception:
            story.append(Paragraph("<i>[Training history figure unavailable]</i>", styles["BodyLeft"]))
    elif not is_statistical:
        story.append(Paragraph("<i>No training history available in this session.</i>", styles["BodyLeft"]))

    # ----- Most-likely results -----
    story.append(Paragraph("5. Results — most-likely prediction", styles["SectionHead"]))
    if has_pred:
        story.append(
            Paragraph(
                ctx.prediction_description
                or "Deterministic full-grid category estimate from the fitted model.",
                styles["BodyJust"],
            )
        )
        try:
            fig_maps = exhaustive_sample_prediction_maps(
                ctx.truth,
                ctx.samples_df,
                ctx.prediction_2d,
                title="Exhaustive vs samples vs prediction (most likely)",
                n_categories=int(ctx.categories),
            )
            story.extend(
                _image_flowable(
                    _fig_to_png_bytes(fig_maps, width=1000),
                    page_width,
                    "Exhaustive / samples / most-likely prediction",
                    styles,
                    max_height=3.6 * inch,
                )
            )
        except Exception:
            story.append(Paragraph("<i>[Most-likely maps unavailable]</i>", styles["BodyLeft"]))
        try:
            fig_prop = category_proportion_comparison(
                ctx.truth,
                ctx.train_df if ctx.train_df is not None else ctx.samples_df,
                ctx.prediction_2d,
                n_categories=int(ctx.categories),
                title="Category proportions — Prediction (most likely)",
                prediction_label="Prediction (most likely)",
            )
            story.extend(
                _image_flowable(
                    _fig_to_png_bytes(fig_prop, width=800),
                    page_width,
                    "Category proportions: exhaustive / training / prediction",
                    styles,
                    max_height=2.8 * inch,
                )
            )
        except Exception:
            story.append(Paragraph("<i>[Proportion chart unavailable]</i>", styles["BodyLeft"]))
        prediction_stat_rows = _sample_statistics_rows(ctx.prediction_statistics)
        if prediction_stat_rows:
            story.append(
                Paragraph(
                    "<b>Sample-derived statistical fidelity</b> "
                    "(exhaustive truth not used)",
                    styles["BodyLeft"],
                )
            )
            story.append(_kv_table(prediction_stat_rows))
    else:
        story.append(
            Paragraph(
                "Most-likely prediction was not available at export time.",
                styles["BodyLeft"],
            )
        )

    # ----- Simulations -----
    story.append(Paragraph("6. Results — sequential simulations", styles["SectionHead"]))
    if n_sims == 0:
        story.append(
            Paragraph(
                "No sequential simulation realizations were available at export time.",
                styles["BodyLeft"],
            )
        )
    else:
        story.append(
            Paragraph(
                f"{n_sims} realization(s) included below. "
                + (
                    ctx.simulation_description
                    or "Each uses a random path and Monte Carlo categorical draws."
                ),
                styles["BodyJust"],
            )
        )
        for sim in ctx.simulations:
            label = str(sim.get("label", "Simulation"))
            seed = sim.get("seed", "—")
            grid = sim.get("grid")
            story.append(Paragraph(f"<b>{_escape(label)}</b> (seed = {_escape(str(seed))})", styles["BodyLeft"]))
            if grid is None:
                story.append(Paragraph("<i>[Grid missing]</i>", styles["BodyLeft"]))
                continue
            try:
                fig_maps = exhaustive_sample_prediction_maps(
                    ctx.truth,
                    ctx.samples_df,
                    grid,
                    title=f"Exhaustive vs samples vs {label}",
                    n_categories=int(ctx.categories),
                )
                story.extend(
                    _image_flowable(
                        _fig_to_png_bytes(fig_maps, width=1000),
                        page_width,
                        f"Maps — {label}",
                        styles,
                    )
                )
            except Exception:
                story.append(Paragraph(f"<i>[Maps unavailable for {_escape(label)}]</i>", styles["BodyLeft"]))
            try:
                fig_prop = category_proportion_comparison(
                    ctx.truth,
                    ctx.train_df if ctx.train_df is not None else ctx.samples_df,
                    grid,
                    n_categories=int(ctx.categories),
                    title=f"Category proportions — {label}",
                    prediction_label=label,
                )
                story.extend(
                    _image_flowable(
                        _fig_to_png_bytes(fig_prop, width=800),
                        page_width,
                        f"Proportions — {label}",
                        styles,
                    )
                )
            except Exception:
                story.append(
                    Paragraph(f"<i>[Proportions unavailable for {_escape(label)}]</i>", styles["BodyLeft"])
                )
            simulation_stat_rows = _sample_statistics_rows(sim.get("statistics"))
            if simulation_stat_rows:
                story.append(
                    Paragraph(
                        "<b>Sample-derived statistical fidelity</b>",
                        styles["BodyLeft"],
                    )
                )
                story.append(_kv_table(simulation_stat_rows))

    if is_statistical:
        story.append(PageBreak())
        story.append(Paragraph("7. Statistical validation", styles["SectionHead"]))
        story.append(
            Paragraph(
                "Corrected SIS uses every sample as hard conditioning data. "
                "Hard-data, sampled-proportion, indicator-variogram, and transition "
                "statistics are reported with each prediction/realization above.",
                styles["BodyJust"],
            )
        )

    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            f"— End of report · Monami_{_escape(version)} · {_escape(when.strftime('%Y-%m-%d %H:%M:%S'))} —",
            styles["Caption"],
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"MONAMI report {version}",
        author="MONAMI 2D",
    )
    doc.build(story)
    return output_path


def generate_report(ctx: ReportContext, reports_dir: Optional[Path] = None) -> Path:
    """
    Build ``Monami_<version>_<timestamp>.pdf`` under the reports directory.

    Returns the absolute path of the written PDF.
    """
    reports_dir = Path(reports_dir or (Path(ctx.project_root) / "4_reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / report_filename(ctx.version)
    return build_pdf_report(ctx, out)
