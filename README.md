# MONAMI 2D

Framework for 2D categorical geostatistical simulations.

Interactive workflow for loading 3D exhaustive property data, discretizing into categories, stratified sampling, selecting a prediction algorithm, training a model, and comparing full-grid predictions with truth and samples.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (typically http://localhost:8501).

On startup the app **auto-loads** the demo file (`porosity_3d.txt`), selects **slice level 0**, runs default **sampling** (3 categories, ~5% density), and applies the **Relative Position** (neighbor DNN) algorithm. The sidebar shows workflow status with checkmarks: **Data**, **Sampling**, and **Algorithm** are ready immediately; **Training** and **Results** complete after you run those steps.

## Web workflow (5 pages)

| Page | Purpose |
|------|---------|
| **Data** | Load SGEMS-style 3D file, browse slices (level 0 by default) |
| **Sampling** | Discretize into categories, stratified sample, histogram, indicator variograms |
| **Algorithm** | Select prediction algorithm and configure algorithm-specific parameters |
| **Training** | Configure DNN hyperparameters, stratified train/test split, train with early stopping |
| **Results** | Most-likely grid prediction, sequential simulation realizations, maps/metrics, PDF report export |

## Project layout

| Path | Purpose |
|------|---------|
| `app/` | Streamlit UI |
| `monami/` | Core library (I/O, sampling, geostats, ML, visualization) |
| `monami/algorithms/` | Pluggable prediction algorithms (registry + implementations) |
| `1_original_exhaustive/` | Demo exhaustive dataset (`porosity_3d.txt`) |
| `2_samples/` | Exported sample CSVs (created by the app) |
| `3_models/` | Saved model bundles (`.h5`, metadata, training pool) |
| `4_reports/` | Exported PDF results reports (`Monami_<version>_<timestamp>.pdf`) |

## Adding a new algorithm

1. Create `monami/algorithms/my_algo.py` subclassing `Algorithm` from `monami/algorithms/base.py`.
2. Set `id` / `name` with the next sequential prefix (`3_…`, `4_…`, …) and fill `long_description` (markdown shown under the selector).
3. Implement `render_config_ui`, `fingerprint`, `train`, `predict_grid`, `evaluate_at_points`, and `simulate_grid`.
4. Register the instance in `monami/algorithms/registry.py` **after** the existing algorithms (keep numeric order).
5. The new algorithm appears automatically on the **Algorithm** page.

For DNN-based algorithms, return `True` from `supports_dnn_training_page()` so hyperparameters appear on the **Training** page. Non-DNN algorithms can train entirely from their own UI.

Saved model bundles store `algorithm_id` and `algorithm_config` in the metadata JSON so **Results** loads the correct predictor.

## Available algorithms

| Id | Display name | Features |
|----|--------------|----------|
| `1_Absolute_Position` | Absolute Position | Normalized absolute X, Y only (input dim = 2); label is category V. |
| `2_Relative_Position` | Relative Position | Per neighbor: dX, dY, D, V (`4 × n_nearest`). App bootstrap default. |

Select either algorithm on the **Algorithm** page (Absolute Position is listed first; extended descriptions appear below the dropdown), then train on **Training**.

## Data format

Exhaustive files use an SGEMS-style header:

```
100*130*30
1
porosity
<values...>
```

Grid dimensions are parsed from the first line. Values are reshaped to `(nz, ny, nx)` for level slicing.

## Relative Position (`2_Relative_Position`)

See [`monami/algorithm`](monami/algorithm) for the MONAMI neighbor-feature specification. Neighbor features (dX, dY, D, V) are built from the **training split only**; the test split is used for validation labels only. Saved model bundles include `{model}_samples.csv` (training pool) for prediction.

## Absolute Position (`1_Absolute_Position`)

Baseline coordinate DNN: inputs are normalized absolute **X** and **Y**; the label is category **V**. No neighbor features. Useful as a simple comparison to Relative Position.

For categorical data, the app computes **indicator variograms**: for each selected category, a binary indicator (1 if cell equals category, else 0) is variogrammed in sample space. This is the standard experimental approach for categorical spatial data.

## Sequential simulation (Results)

After a model is trained, **Results** can run classic sequential simulation without retraining. Shared steps:

1. Training samples are fixed as hard data.
2. Remaining cells are visited in a random path.
3. A category is **sampled** from the DNN softmax (not argmax) and written to the grid.

Feature construction at each path cell depends on the algorithm:

- **Relative Position:** neighbor features from the **current** conditioning set (hard data + previously simulated values); each draw is added to that pool.
- **Absolute Position:** features are only that cell’s normalized **(X, Y)**; the conditioning pool is not used as model input.

Each run appends labeled maps (**Simulation 1**, **Simulation 2**, …). The separate full-grid prediction button still produces the deterministic most-likely (argmax) map used for test metrics.

## PDF results report

On **Results**, after prediction and/or simulations, use **Export report** to write a shareable PDF under `4_reports/`:

`Monami_<version>_<YYYYMMDD_HHMMSS>.pdf`

Enter a manual **version series** tag (e.g. `v0.1`). The PDF summarizes data, sampling, algorithm/method, training parameters, maps, category proportions, simulations, and test metrics available in the session.

## Python version

Use Python 3.10 or 3.11 for TensorFlow compatibility.

## Troubleshooting (macOS)

If training hangs with `[mutex.cc : 452] RAW: Lock blocking` or `mutex lock failed: Invalid argument`, you likely have **TensorFlow 2.20** installed. That release conflicts with PyArrow (used by Streamlit/pandas) on Apple Silicon.

Fix:

```bash
pip install "tensorflow==2.19.1"
```

Then restart Streamlit.
