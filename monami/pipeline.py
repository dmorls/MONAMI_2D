"""End-to-end pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from monami.config import MLConfig, WorkflowConfig
from monami.io import extract_slice, load_from_path_or_bytes
from monami.ml import (
    load_model_bundle,
    predict_grid,
    save_model_bundle,
    split_samples,
    train_model,
)
from monami.sampling import stratified_sample_dataframe
from monami.transform import categorize_slice, numpy2d_to_dataframe


def run_discretize_and_sample(cfg: WorkflowConfig):
    """Load volume, extract slice, categorize, and stratified sample."""
    volume, meta = load_from_path_or_bytes(cfg.exhaustive_path())
    slice_2d = extract_slice(volume, cfg.level, meta)
    _, categorized, _ = categorize_slice(slice_2d, cfg.categories)
    samples = stratified_sample_dataframe(
        categorized,
        cfg.sample_n_h,
        cfg.sample_n_v,
        seed=cfg.random_seed,
    )
    return volume, meta, slice_2d, categorized, samples


def run_train(samples, cfg: WorkflowConfig, ml_cfg: MLConfig):
    """Train DNN from sample DataFrame."""
    train_df, test_df = split_samples(samples, ml_cfg.test_ratio, seed=cfg.random_seed)
    model, history, meta, _ = train_model(train_df, test_df, ml_cfg)
    meta.grid_shape = list(samples.attrs.get("grid_shape", [])) if hasattr(samples, "attrs") else []
    model_path = save_model_bundle(model, meta, cfg.project_root / cfg.model_folder, ml_cfg, train_df)
    return model, history, meta, train_df, test_df, model_path


def run_predict(model_path: Path, grid_shape: tuple[int, int]):
    """Predict full grid from saved model."""
    model, meta, neighbor_pool_df = load_model_bundle(model_path)
    meta.grid_shape = list(grid_shape)
    prediction = predict_grid(model, grid_shape, meta, neighbor_pool_df)
    return prediction, meta
