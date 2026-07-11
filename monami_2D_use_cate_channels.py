#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MONAMI 2D categorical workflow — refactored pipeline driver.

Runs the full pipeline using the ``monami`` package API.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import monami_2D_functions as f
from monami.config import MLConfig, WorkflowConfig
from monami.io import extract_slice, load_from_path_or_bytes, save_slice_numpy
from monami.ml import predict_grid, save_model_bundle, split_samples, train_model
from monami.pipeline import run_discretize_and_sample
from monami.transform import numpy2d_to_dataframe, numpy2d_to_easyformat


def main():
    cfg = WorkflowConfig(project_root=Path("."))
    ml_cfg = MLConfig()

    # --- Load slice (continuous) ---
    volume, meta = load_from_path_or_bytes(cfg.exhaustive_path())
    slice_2d = extract_slice(volume, level=2, meta=meta)
    easy = numpy2d_to_easyformat(slice_2d)
    df = numpy2d_to_dataframe(easy)
    out_dir = cfg.project_root / cfg.exh_folder
    stem = cfg.exh_file.rsplit(".", 1)[0]
    level = 2
    f.Histogram(easy[:, -1], cfg.hist_limits_continuous, "auto", stem, str(out_dir / f"{stem}_level{level}_hist.pdf"))
    plt.close()
    f.Show_array(
        slice_2d,
        cfg.hist_limits_continuous[0],
        cfg.hist_limits_continuous[1],
        f"{stem}, level: {level}",
        "y",
        str(out_dir / f"{stem}_level{level}_fig.pdf"),
    )
    plt.close()
    save_slice_numpy(out_dir / f"{stem}_level{level}_np.txt", slice_2d)
    df.to_csv(out_dir / f"{stem}_level{level}_df.txt", index=False)

    # --- Discretize and sample ---
    cfg.level = 0
    volume, meta, slice_2d, categorized, samples = run_discretize_and_sample(cfg)
    cat_dir = cfg.project_root / cfg.cat_folder
    cat_dir.mkdir(parents=True, exist_ok=True)
    cat_stem = f"{stem}_level{cfg.level}_cat{cfg.categories}"
    np.savetxt(cat_dir / f"{cat_stem}_np.txt", categorized, delimiter=",")
    numpy2d_to_dataframe(numpy2d_to_easyformat(categorized)).to_csv(
        cat_dir / f"{cat_stem}_df.txt", index=False
    )
    f.Show_array(categorized, categorized.min(), categorized.max(), cat_stem, "y", str(cat_dir / f"{cat_stem}_fig.pdf"))
    plt.close()
    f.Histogram(samples["V"], (0, cfg.categories - 1), "auto", cat_stem, str(cat_dir / f"{cat_stem}_hist.pdf"))
    plt.close()

    sam_dir = cfg.project_root / cfg.sample_folder
    sam_dir.mkdir(parents=True, exist_ok=True)
    sample_fn = f"sample_all_{cfg.categories}_{len(samples)}.csv"
    samples.to_csv(sam_dir / sample_fn, index=False)
    x_min, x_max = samples["X"].min(), samples["X"].max()
    y_min, y_max = samples["Y"].min(), samples["Y"].max()
    f.Plot_df_scatter(
        samples,
        "V",
        0.1,
        x_min,
        x_max,
        y_min,
        y_max,
        samples["V"].min(),
        samples["V"].max(),
        "All, Categorical:",
        str(sam_dir / sample_fn.replace(".csv", ".pdf")),
    )
    plt.close()

    # --- Train ---
    train_df, test_df = split_samples(samples, ml_cfg.test_ratio, seed=cfg.random_seed)
    model, history, meta_ml, _ = train_model(train_df, test_df, ml_cfg)
    meta_ml.grid_shape = list(categorized.shape)
    model_path = save_model_bundle(model, meta_ml, cfg.project_root / cfg.model_folder, ml_cfg, train_df)
    print(f"Saved model: {model_path}")

    # --- Predict ---
    prediction = predict_grid(model, categorized.shape, meta_ml, train_df)
    f.Histogram(
        prediction.ravel(),
        (0, cfg.categories - 1),
        "auto",
        meta_ml.model_filename,
        str(cfg.project_root / cfg.model_folder / f"{meta_ml.model_filename}_hist.pdf"),
    )
    plt.close()
    f.Show_array(
        prediction,
        prediction.min(),
        prediction.max(),
        meta_ml.model_filename,
        "y",
        str(cfg.project_root / cfg.model_folder / f"{meta_ml.model_filename}.pdf"),
    )
    plt.close()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
