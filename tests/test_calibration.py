import numpy as np
from sklearn.metrics import log_loss

from cfb_fantasy.calibration import (
    CALIBRATION_METHODS,
    calibration_metrics,
    fit_probability_calibrator,
)


def test_calibrators_return_monotonic_probabilities():
    probability = np.linspace(0.02, 0.98, 200)
    target = (np.arange(200) % 3 == 0).astype(int)
    for method in CALIBRATION_METHODS:
        calibrator = fit_probability_calibrator(method, probability, target)
        calibrated = calibrator.predict(probability)
        assert np.all(np.diff(calibrated) >= -1e-12)
        assert np.all((calibrated > 0.0) & (calibrated < 1.0))


def test_beta_calibration_can_correct_overconfident_probabilities():
    rng = np.random.default_rng(17)
    true_probability = rng.uniform(0.1, 0.9, 4000)
    target = rng.binomial(1, true_probability)
    overconfident = 1.0 / (
        1.0 + np.exp(-1.8 * np.log(true_probability / (1.0 - true_probability)))
    )
    beta = fit_probability_calibrator("beta", overconfident, target)
    assert log_loss(target, beta.predict(overconfident)) < log_loss(
        target, overconfident
    )


def test_murphy_decomposition_reconciles_to_brier_score():
    target = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    probability = np.array([0.1, 0.3, 0.55, 0.7, 0.9, 0.4, 0.8, 0.2])
    metrics = calibration_metrics(target, probability)
    assert abs(metrics["murphy_error"]) < 1e-12
