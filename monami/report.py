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
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from monami.ml import ModelMeta
from monami.viz import (
    category_proportion_comparison,
    confusion_matrix_plot,
    exhaustive_sample_prediction_maps,
    observed_vs_predicted_scatter,
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
    prediction_2d: Optional[np.ndarray] = None
    simulations: List[Dict[str, Any]] = field(default_factory=list)
    history: Any = None
    live_training_history: Optional[Dict[str, List[float]]] = None
    y_true_test: Optional[np.ndarray] = None
    y_pred_test: Optional[np.ndarray] = None
    classification_report_text: str = ""
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
            name="SmallMono",
            parent=styles["Code"],
            fontSize=7,
            leading=9,
            fontName="Courier",
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


def _image_flowable(png_bytes: Optional[bytes], max_width: float, caption: str, styles) -> List[Any]:
    items: List[Any] = []
    if png_bytes is None:
        items.append(Paragraph(f"<i>[Figure unavailable: {_escape(caption)}]</i>", styles["BodyLeft"]))
        return items
    bio = io.BytesIO(png_bytes)
    img = Image(bio)
    # Scale to max width preserving aspect
    iw, ih = img.imageWidth, img.imageHeight
    if iw > max_width:
        scale = max_width / float(iw)
        img.drawWidth = max_width
        img.drawHeight = ih * scale
    items.append(img)
    items.append(Paragraph(_escape(caption), styles["Caption"]))
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

    # ----- Cover -----
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("MONAMI 2D — Results Report", styles["CoverTitle"]))
    story.append(Paragraph(f"Version series: <b>{_escape(version)}</b>", styles["BodyLeft"]))
    story.append(Paragraph(f"Generated: {_escape(when.strftime('%Y-%m-%d %H:%M:%S'))}", styles["BodyLeft"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "This document summarizes the categorical geostatistical workflow: data, sampling, "
            "prediction algorithm, training parameters, full-grid prediction / sequential simulations, "
            "and evaluation metrics available at export time.",
            styles["BodyJust"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
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
            "Stratified sampling draws points from the categorized exhaustive field. "
            "The training split is the hard-data / neighbor pool for prediction and sequential simulation; "
            "the test split is used for validation metrics when available.",
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
            "When sequential simulations are run, uncertainty comes from at least two sources: "
            "(1) a <b>random path</b> over unsampled cells, and (2) a <b>Monte Carlo draw</b> from the "
            "DNN softmax distribution at each path cell (not argmax). Hard training samples remain fixed. "
            "For neighbor-based algorithms, previously simulated values enter the conditioning pool and "
            "affect later features; for coordinate-only algorithms, features are the cell’s (X, Y) only.",
            styles["BodyJust"],
        )
    )

    # ----- Training -----
    story.append(Paragraph("4. Training", styles["SectionHead"]))
    m = ctx.meta
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
            png = _fig_to_png_bytes(fig, width=800)
            story.extend(_image_flowable(png, page_width, "Training accuracy and loss curves", styles))
        except Exception:
            story.append(Paragraph("<i>[Training history figure unavailable]</i>", styles["BodyLeft"]))
    else:
        story.append(Paragraph("<i>No training history available in this session.</i>", styles["BodyLeft"]))

    story.append(PageBreak())

    # ----- Most-likely results -----
    story.append(Paragraph("5. Results — most-likely prediction", styles["SectionHead"]))
    if has_pred:
        story.append(
            Paragraph(
                "Deterministic full-grid map: argmax of the DNN softmax at every cell.",
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
                )
            )
        except Exception:
            story.append(Paragraph("<i>[Proportion chart unavailable]</i>", styles["BodyLeft"]))
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
                f"{n_sims} realization(s) included below. Each uses a random path and "
                "softmax Monte Carlo sampling as described in Section 3.",
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

    # ----- Metrics -----
    story.append(PageBreak())
    story.append(Paragraph("7. Test-set metrics", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Metrics below use the most-likely (argmax) prediction evaluated at test-sample locations, "
            "not sequential realizations.",
            styles["BodyJust"],
        )
    )
    if ctx.y_true_test is not None and ctx.y_pred_test is not None and len(ctx.y_true_test) > 0:
        try:
            fig_cm = confusion_matrix_plot(ctx.y_true_test, ctx.y_pred_test)
            story.extend(
                _image_flowable(
                    _fig_to_png_bytes(fig_cm, width=600),
                    page_width * 0.85,
                    "Confusion matrix (test set)",
                    styles,
                )
            )
        except Exception:
            story.append(Paragraph("<i>[Confusion matrix unavailable]</i>", styles["BodyLeft"]))
        try:
            fig_sc = observed_vs_predicted_scatter(ctx.y_true_test, ctx.y_pred_test)
            story.extend(
                _image_flowable(
                    _fig_to_png_bytes(fig_sc, width=600),
                    page_width * 0.85,
                    "Observed vs predicted (test set)",
                    styles,
                )
            )
        except Exception:
            story.append(Paragraph("<i>[Scatter plot unavailable]</i>", styles["BodyLeft"]))
        if ctx.classification_report_text:
            story.append(Paragraph("<b>Classification report</b>", styles["BodyLeft"]))
            story.append(Preformatted(ctx.classification_report_text, styles["SmallMono"]))
    else:
        story.append(
            Paragraph(
                "Test-set metrics were not available (no most-likely prediction and/or no test split).",
                styles["BodyLeft"],
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
