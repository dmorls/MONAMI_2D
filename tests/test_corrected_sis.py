import numpy as np
import pandas as pd

from monami.config import MLConfig
from monami.geostats import (
    corrected_sis_validation_metrics,
    fit_indicator_variogram_model,
    variogram_values,
)
from monami.ml import load_model_bundle, save_model_bundle
from monami.sis import (
    apply_proportion_servo,
    fit_corrected_sis_model,
    indicator_kriging_probabilities,
    simulate_corrected_sis_grid,
)


def _samples() -> pd.DataFrame:
    rows = []
    for y in range(1, 19, 3):
        for x in range(1, 16, 3):
            value = 0 if x < 6 else (1 if y < 10 else 2)
            rows.append((x, y, value))
    return pd.DataFrame(rows, columns=["X", "Y", "V"])


def _model():
    return fit_corrected_sis_model(
        _samples(),
        neighborhood_size=10,
        n_lags=6,
        directional=False,
    )


def test_variogram_models_are_zero_at_origin_and_reach_sill():
    distances = np.array([0.0, 1_000.0])
    for model in ("spherical", "exponential", "gaussian"):
        values = variogram_values(distances, model, 0.05, 0.2, 10.0)
        assert values[0] == 0.0
        assert np.isclose(values[1], 0.25, atol=1e-6)


def test_sparse_variogram_fit_has_valid_isotropic_fallback():
    sparse = pd.DataFrame(
        {"X": [1, 10], "Y": [1, 10], "V": [0, 1]}
    )
    fitted = fit_indicator_variogram_model(
        sparse,
        0,
        n_lags=15,
        directional=True,
    )
    assert fitted.fallback
    assert fitted.range_x > 0
    assert fitted.range_y > 0
    assert fitted.partial_sill > 0


def test_indicator_probabilities_are_normalized_and_pin_exact_data():
    samples = _samples()
    model = _model()
    coords = samples[["X", "Y"]].to_numpy()
    values = samples["V"].to_numpy()
    probabilities = indicator_kriging_probabilities(
        np.array([8.0, 8.0]),
        coords,
        values,
        model,
    )
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.isclose(probabilities.sum(), 1.0)

    exact = indicator_kriging_probabilities(coords[0], coords, values, model)
    assert exact[model.categories.index(int(values[0]))] == 1.0
    assert np.isclose(exact.sum(), 1.0)


def test_proportion_servo_moves_probability_toward_remaining_quota():
    local = np.array([0.8, 0.1, 0.1])
    target = np.array([0.4, 0.3, 0.3])
    counts = np.array([40, 10, 10])
    corrected = apply_proportion_servo(
        local,
        target,
        counts,
        total_cells=100,
        completed_cells=60,
        strength=0.5,
    )
    assert np.isclose(corrected.sum(), 1.0)
    assert corrected[0] < local[0]
    assert corrected[1] > local[1]
    assert corrected[2] > local[2]


def test_seeded_simulation_is_reproducible_and_honors_hard_data():
    samples = _samples()
    model = _model()
    first = simulate_corrected_sis_grid(model, samples, (18, 15), seed=17)
    second = simulate_corrected_sis_grid(model, samples, (18, 15), seed=17)
    assert np.array_equal(first, second)
    assert set(np.unique(first)).issubset(set(model.categories))
    x = samples["X"].to_numpy(dtype=int) - 1
    y = samples["Y"].to_numpy(dtype=int) - 1
    assert np.array_equal(first[y, x], samples["V"].to_numpy(dtype=int))

    metrics = corrected_sis_validation_metrics(
        first,
        samples,
        model,
        seed=17,
        max_variogram_pairs=5_000,
    )
    assert metrics["hard_data_fidelity"] == 1.0
    assert metrics["proportion_l1"] >= 0.0
    assert metrics["variogram_rmse_mean"] >= 0.0


def test_statistical_bundle_round_trip(tmp_path):
    samples = _samples()
    model = _model()
    from monami.ml import ModelMeta

    meta = ModelMeta(
        n_classes=3,
        categories=3,
        grid_shape=[18, 15],
        nodes_per_layer=[],
        dropout=0.0,
        optimizer="indicator_kriging",
        loss_function="variogram_fit",
        hidden_activation="",
        test_ratio=0.0,
        training_seconds=0.1,
        model_filename="",
        algorithm_id="3_Corrected_SIS",
        algorithm_config={},
        model_type="corrected_sis",
        train_sample_count=len(samples),
    )
    path = save_model_bundle(
        model,
        meta,
        tmp_path,
        MLConfig(suffix="test"),
        samples,
        samples,
    )
    restored, restored_meta, restored_samples = load_model_bundle(path)
    assert restored.to_dict() == model.to_dict()
    assert restored_meta.model_type == "corrected_sis"
    assert len(restored_samples) == len(samples)
