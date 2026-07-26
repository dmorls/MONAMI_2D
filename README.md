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
| `3_models/` | Saved model bundles (`.h5` DNN or `.json` statistical model, metadata, conditioning pool) |
| `4_reports/` | Exported PDF results reports (`Monami_<version>_<timestamp>.pdf`) |

## Adding a new algorithm

1. Create `monami/algorithms/my_algo.py` subclassing `Algorithm` from `monami/algorithms/base.py`.
2. Set `id` / `name` with the next sequential prefix (`4_…`, `5_…`, …) and fill `long_description` (markdown shown under the selector).
3. Implement `render_config_ui`, `fingerprint`, `train`, `predict_grid`, `evaluate_at_points`, and `simulate_grid`.
4. Register the instance in `monami/algorithms/registry.py` **after** the existing algorithms (keep numeric order).
5. The new algorithm appears automatically on the **Algorithm** page.

For DNN-based algorithms, return `True` from `supports_dnn_training_page()` so hyperparameters appear on the **Training** page. Non-DNN algorithms return `False` and use the statistical fitting path there.

Saved model bundles store `algorithm_id` and `algorithm_config` in the metadata JSON so **Results** loads the correct predictor.

## Available algorithms

| Id | Display name | Features |
|----|--------------|----------|
| `1_Absolute_Position` | Absolute Position | Normalized absolute X, Y only (input dim = 2); label is category V. |
| `2_Relative_Position` | Relative Position | Per neighbor: dX, dY, D, V (`4 × n_nearest`). App bootstrap default. |
| `3_Corrected_SIS` | Corrected Sequential Indicator Simulation | Sample proportions + fitted category indicator variograms; local indicator kriging and proportion-corrected sequential draws. |
| `4_Hybrid_Position` | Hybrid Position | Normalized absolute X, Y prepended to relative neighbor features (`2 + 4 × n_nearest`). |

Select an algorithm on the **Algorithm** page (extended descriptions appear below the dropdown), then train or statistically fit it on **Training**.

## Data format

Exhaustive files use an SGEMS-style header:

```
100*130*30
1
porosity
<values...>
```

Grid dimensions are parsed from the first line. Values are reshaped to `(nz, ny, nx)` for level slicing.

## Optional training image (TI)

On **Sampling**, you can enable a **training image**: another Z-slice from the same 3D volume.

- TI is categorized with the **same thresholds** as the target slice and sampled at the same density.
- TI points are **auxiliary DNN training labels only** (Absolute / Relative / Hybrid). They are **not** hard-pinned on the target grid for prediction or sequential simulation.
- Relative/Hybrid neighbor features for TI rows use the TI sample pool; target rows and Results hard data use the target training split only.
- **Corrected SIS** ignores the training image.

## Relative Position (`2_Relative_Position`)

See [`monami/algorithm`](monami/algorithm) for the MONAMI neighbor-feature specification. Neighbor features (dX, dY, D, V) are built from the **training split only**; the test split is used for validation labels only. Saved model bundles include `{model}_samples.csv` (training pool) for prediction. On the Algorithm page, **Inspect nearest neighbors** walks the training pool with a slider and highlights the `n` conditioning neighbors used for features.

## Absolute Position (`1_Absolute_Position`)

Baseline coordinate DNN: inputs are normalized absolute **X** and **Y**; the label is category **V**. No neighbor features. Useful as a simple comparison to Relative Position.

## Hybrid Position (`4_Hybrid_Position`)

Combines Absolute and Relative Position: normalized target **X**, **Y** are prepended to the MONAMI neighbor block (`dX`, `dY`, `D`, `V` × `n`). Input dimension = `2 + 4 × n`. Training can learn how much weight to give absolute location vs local neighborhood. Prediction and sequential simulation use the same hybrid vector; simulation grows the conditioning pool like Relative Position. The Algorithm page includes the same nearest-neighbor inspector as Relative Position (training-pool focus sample + `n` neighbors).

## Corrected SIS (`3_Corrected_SIS`)

Corrected Sequential Indicator Simulation is a sample-only categorical geostatistical method:

1. It estimates the sampled category proportions.
2. It fits spherical, exponential, or Gaussian indicator variograms for each category, with optional X/Y directional ranges.
3. It computes local simple-indicator-kriging probabilities from nearby conditioning values.
4. During simulation, a servo correction gently steers global counts toward the sampled proportions.
5. Every sampled point is pinned exactly.

The exhaustive categorized field is **never used for fitting, prediction, or simulation**. It remains visible only for after-the-fact validation. Results and PDF reports include hard-data fidelity, sampled-proportion error, fitted-indicator-variogram error, and directional transition error.

For categorical data, the app computes **indicator variograms**: for each selected category, a binary indicator (1 if cell equals category, else 0) is variogrammed in sample space. This is the standard experimental approach for categorical spatial data.

## Sequential simulation (Results)

After a model is trained, **Results** can run classic sequential simulation without retraining. Shared steps:

1. Conditioning samples are fixed as hard data.
2. Remaining cells are visited in a random path.
3. A category is sampled from the algorithm’s local probability distribution and written to the grid.

Feature construction at each path cell depends on the algorithm:

- **Relative Position:** neighbor features from the **current** conditioning set (hard data + previously simulated values); each draw is added to that pool.
- **Absolute Position:** features are only that cell’s normalized **(X, Y)**; the conditioning pool is not used as model input.
- **Hybrid Position:** normalized **(X, Y)** plus neighbor features from the growing conditioning set (same pool growth as Relative Position).
- **Corrected SIS:** all samples are hard data; local indicator-kriging probabilities use the growing pool and are adjusted toward the sampled category histogram.

**Sample-proportion servo (DNN algorithms):** Absolute, Relative, and Hybrid simulations blend the local DNN softmax with a remaining training-sample histogram quota at each draw (same soft servo as Corrected SIS). Strength is set on **Results** (default 0.50; 0 = pure DNN, 1 = hard quota). Most-likely prediction remains pure argmax with no servo.

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
