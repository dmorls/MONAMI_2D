"""Sample-only corrected Sequential Indicator Simulation algorithm."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from monami.algorithms.base import Algorithm, TrainingResult
from monami.config import MLConfig
from monami.ml import ModelMeta
from monami.sis import (
    CorrectedSISModel,
    fit_corrected_sis_model,
    indicator_kriging_probabilities,
    predict_indicator_grid,
    simulate_corrected_sis_grid,
)


class CorrectedSISAlgorithm(Algorithm):
    id = "3_Corrected_SIS"
    name = "Corrected Sequential Indicator Simulation"
    description = (
        "Sample-only indicator kriging and sequential simulation with fitted "
        "variograms and proportion control."
    )
    long_description = """
### Corrected Sequential Indicator Simulation (ccSIS)

A classical categorical geostatistical method designed to reproduce the
statistics measured in the sampled data.

**What is fitted from samples**
- One indicator variogram for each category
- Optional X/Y directional ranges (anisotropy)
- The sampled category proportions

**Prediction / simulation**
- Deterministic map = category with the highest local indicator-kriging probability
- Simulation = random path, growing conditioning set, and categorical draws
- A servo correction gently steers each realization toward sampled proportions
- Every sampled value is pinned exactly

**No truth leakage**
- The exhaustive field is not used during fitting, prediction, or simulation
- It is used only afterward to compare and validate results
""".strip()

    def render_config_ui(
        self,
        st_module: Any,
        samples_df: pd.DataFrame,
        categorized_2d: np.ndarray,
        *,
        random_seed: int,
        default_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        defaults = dict(default_config or {})
        n_samples = max(int(len(samples_df)), 3)
        c1, c2 = st_module.columns(2)
        with c1:
            neighborhood_size = st_module.number_input(
                "Conditioning neighborhood size",
                min_value=2,
                max_value=max(2, min(128, n_samples)),
                value=min(
                    max(int(defaults.get("neighborhood_size", 24)), 2),
                    max(2, min(128, n_samples)),
                ),
                step=1,
                help="Nearest hard/simulated values used in each local indicator-kriging system.",
            )
            variogram_model = st_module.selectbox(
                "Indicator variogram model",
                ["auto", "spherical", "exponential", "gaussian"],
                index=["auto", "spherical", "exponential", "gaussian"].index(
                    str(defaults.get("variogram_model", "auto"))
                    if str(defaults.get("variogram_model", "auto"))
                    in {"auto", "spherical", "exponential", "gaussian"}
                    else "auto"
                ),
                help="Auto compares candidate models by weighted experimental-variogram RMSE.",
            )
            n_lags = st_module.number_input(
                "Variogram lag bins",
                min_value=5,
                max_value=30,
                value=min(max(int(defaults.get("n_lags", 15)), 5), 30),
                step=1,
            )
        with c2:
            correction_strength = st_module.slider(
                "Sample-proportion correction",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.get("correction_strength", 0.5)),
                step=0.05,
                help=(
                    "0 uses only local kriging probabilities; 1 uses only the remaining "
                    "sample-histogram quota. Recommended: 0.50."
                ),
            )
            directional = st_module.checkbox(
                "Fit X/Y directional ranges",
                value=bool(defaults.get("directional", True)),
                help="Uses directional variograms when supported; otherwise falls back to isotropic.",
            )
            radius_default = float(defaults.get("max_radius") or 0.0)
            max_radius = st_module.number_input(
                "Maximum conditioning radius (0 = unlimited)",
                min_value=0.0,
                value=max(radius_default, 0.0),
                step=1.0,
            )
        st_module.caption(
            "This model is fitted from **all sampled points** on the Training page. "
            "The exhaustive field is validation-only."
        )
        return {
            "neighborhood_size": int(neighborhood_size),
            "max_radius": float(max_radius) if max_radius > 0 else None,
            "correction_strength": float(correction_strength),
            "n_lags": int(n_lags),
            "variogram_model": str(variogram_model),
            "directional": bool(directional),
        }

    def fingerprint(self, config: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            self.id,
            int(config.get("neighborhood_size", 24)),
            config.get("max_radius"),
            round(float(config.get("correction_strength", 0.5)), 6),
            int(config.get("n_lags", 15)),
            str(config.get("variogram_model", "auto")),
            bool(config.get("directional", True)),
        )

    def supports_dnn_training_page(self) -> bool:
        return False

    def feature_summary(self, algo_config: Dict[str, Any]) -> str:
        direction = "directional X/Y" if algo_config.get("directional", True) else "isotropic"
        return (
            f"Indicator variograms ({direction}), "
            f"{int(algo_config.get('neighborhood_size', 24))} neighbors, "
            f"proportion correction {float(algo_config.get('correction_strength', 0.5)):.2f}"
        )

    def prediction_description(self) -> str:
        return (
            "Argmax of local indicator-kriging probabilities fitted from all sampled "
            "points. Sampled cells are pinned exactly; exhaustive truth is not used."
        )

    def simulation_description(self) -> str:
        return (
            "Corrected Sequential Indicator Simulation over a seeded random path. "
            "All sampled values are pinned, the conditioning pool grows after each "
            "draw, and a servo correction targets the sampled category proportions."
        )

    def validate_config(
        self,
        algo_config: Dict[str, Any],
        train_df: pd.DataFrame,
    ) -> Optional[str]:
        n = int(algo_config.get("neighborhood_size", 24))
        if len(train_df) < 3:
            return "Corrected SIS requires at least three samples."
        if n > len(train_df):
            return f"Neighborhood size ({n}) cannot exceed sample count ({len(train_df)})."
        observed = train_df["V"].nunique()
        if observed < 2:
            return "Corrected SIS requires at least two observed categories."
        return None

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        neighbor_pool_df: pd.DataFrame,
        grid_shape: Optional[Tuple[int, int]],
        algo_config: Dict[str, Any],
        ml_config: MLConfig,
        *,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        epoch_callback: Optional[Callable[[int, int, dict, Optional[np.ndarray]], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        warm_start: Any = None,
        epochs_to_run: Optional[int] = None,
        ti_samples_df: Optional[pd.DataFrame] = None,
    ) -> TrainingResult:
        del test_df, neighbor_pool_df, warm_start, epochs_to_run, ti_samples_df
        validation_error = self.validate_config(algo_config, train_df)
        if validation_error:
            raise ValueError(validation_error)
        start = time.time()
        if log_callback:
            log_callback(
                f"Fitting indicator variograms from all {len(train_df):,} sampled points..."
            )
        categories = sorted(int(v) for v in train_df["V"].unique())
        model = fit_corrected_sis_model(
            train_df,
            categories=categories,
            neighborhood_size=int(algo_config.get("neighborhood_size", 24)),
            max_radius=algo_config.get("max_radius"),
            correction_strength=float(algo_config.get("correction_strength", 0.5)),
            n_lags=int(algo_config.get("n_lags", 15)),
            variogram_model=str(algo_config.get("variogram_model", "auto")),
            directional=bool(algo_config.get("directional", True)),
        )
        if progress_callback:
            progress_callback(len(categories), len(categories))
        elapsed = time.time() - start
        if log_callback:
            for category in categories:
                fitted = model.variogram_for(category)
                fit_text = (
                    "fallback" if fitted.fallback else f"RMSE={fitted.fit_rmse:.4g}"
                )
                log_callback(
                    f"Category {category}: {fitted.model}, nugget={fitted.nugget:.4g}, "
                    f"sill={fitted.total_sill:.4g}, range X/Y="
                    f"{fitted.range_x:.2f}/{fitted.range_y:.2f} ({fit_text})"
                )
            log_callback(f"Statistical model fitted in {elapsed:.2f}s.")

        class_to_idx = {category: i for i, category in enumerate(categories)}
        meta = ModelMeta(
            n_classes=len(categories),
            categories=len(categories),
            grid_shape=list(grid_shape) if grid_shape else [],
            nodes_per_layer=[],
            dropout=0.0,
            optimizer="indicator_kriging",
            loss_function="variogram_fit",
            hidden_activation="",
            out_activation="probabilities",
            test_ratio=0.0,
            training_seconds=elapsed,
            model_filename="",
            n_nearest=model.neighborhood_size,
            feature_dim=0,
            train_sample_count=len(train_df),
            neighbor_sample_count=len(train_df),
            class_to_idx={str(k): int(v) for k, v in class_to_idx.items()},
            idx_to_class={str(i): int(c) for i, c in enumerate(categories)},
            algorithm_id=self.id,
            algorithm_config=dict(algo_config),
            model_type="corrected_sis",
        )
        return TrainingResult(
            model=model,
            history=None,
            meta=meta,
            classes=categories,
            incomplete=False,
            current_epoch=1,
            max_epochs=1,
        )

    @staticmethod
    def _coerce_model(model: Any) -> CorrectedSISModel:
        if isinstance(model, CorrectedSISModel):
            return model
        if isinstance(model, dict):
            return CorrectedSISModel.from_dict(model)
        raise TypeError(f"Expected CorrectedSISModel, got {type(model).__name__}")

    def predict_grid(
        self,
        model: Any,
        grid_shape: Tuple[int, int],
        meta: ModelMeta,
        neighbor_pool_df: pd.DataFrame,
    ) -> np.ndarray:
        del meta
        return predict_indicator_grid(
            self._coerce_model(model),
            neighbor_pool_df,
            grid_shape,
        )

    def evaluate_at_points(
        self,
        model: Any,
        meta: ModelMeta,
        points_df: pd.DataFrame,
        neighbor_pool_df: pd.DataFrame,
    ) -> np.ndarray:
        del meta
        fitted = self._coerce_model(model)
        coords = neighbor_pool_df[["X", "Y"]].to_numpy(dtype=float)
        values = neighbor_pool_df["V"].to_numpy(dtype=int)
        tree = cKDTree(coords)
        predictions = []
        for target in points_df[["X", "Y"]].to_numpy(dtype=float):
            probabilities = indicator_kriging_probabilities(
                target,
                coords,
                values,
                fitted,
                tree=tree,
            )
            predictions.append(fitted.categories[int(np.argmax(probabilities))])
        return np.asarray(predictions, dtype=int)

    def simulate_grid(
        self,
        model: Any,
        meta: ModelMeta,
        hard_df: pd.DataFrame,
        grid_shape: Tuple[int, int],
        *,
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        correction_strength: float = 0.5,
    ) -> np.ndarray:
        del meta, correction_strength  # use fitted model.correction_strength
        return simulate_corrected_sis_grid(
            self._coerce_model(model),
            hard_df,
            grid_shape,
            seed=seed,
            progress_callback=progress_callback,
        )
