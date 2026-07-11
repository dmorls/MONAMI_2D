# MONAMI 2D

Framework for 2D categorical geostatistical simulations.

Interactive workflow for loading 3D exhaustive property data, discretizing into categories, stratified sampling, training a coordinate-based DNN, and comparing full-grid predictions with truth and samples.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (typically http://localhost:8501).

On startup the app **auto-loads** the demo file (`porosity_3d.txt`), selects **slice level 0**, and runs default **sampling** (3 categories, ~5% density). The sidebar shows workflow status with checkmarks: **Data** and **Sampling** are ready immediately; **Training** and **Results** complete after you run those steps.

## Web workflow (4 pages)

| Page | Purpose |
|------|---------|
| **Data** | Load SGEMS-style 3D file, browse slices (level 0 by default) |
| **Sampling** | Discretize into categories, stratified sample, histogram, indicator variograms |
| **Training** | Configure DNN, stratified train/test split, train with early stopping |
| **Results** | Predict full grid, compare maps/histograms, confusion matrix on test set |

## Project layout

| Path | Purpose |
|------|---------|
| `app/` | Streamlit UI |
| `monami/` | Core library (I/O, sampling, geostats, ML, visualization) |
| `1_original_exhaustive/` | Demo exhaustive dataset (`porosity_3d.txt`) |
| `2_samples/` | Exported sample CSVs (created by the app) |
| `3_models/` | Saved model bundles (`.h5`, metadata, training pool) |

## Data format

Exhaustive files use an SGEMS-style header:

```
100*130*30
1
porosity
<values...>
```

Grid dimensions are parsed from the first line. Values are reshaped to `(nz, ny, nx)` for level slicing.

## Training algorithm

See [`monami/algorithm`](monami/algorithm) for the MONAMI neighbor-feature specification. Neighbor features (dX, dY, D, V) are built from the **training split only**; the test split is used for validation labels only. Saved model bundles include `{model}_samples.csv` (training pool) for prediction.

For categorical data, the app computes **indicator variograms**: for each selected category, a binary indicator (1 if cell equals category, else 0) is variogrammed in sample space. This is the standard experimental approach for categorical spatial data.

## Python version

Use Python 3.10 or 3.11 for TensorFlow compatibility.

## Troubleshooting (macOS)

If training hangs with `[mutex.cc : 452] RAW: Lock blocking` or `mutex lock failed: Invalid argument`, you likely have **TensorFlow 2.20** installed. That release conflicts with PyArrow (used by Streamlit/pandas) on Apple Silicon.

Fix:

```bash
pip install "tensorflow==2.19.1"
```

Then restart Streamlit.
