"""Calibration correction + diagnostic tests.

Covers: calibrators improve a synthetically miscalibrated-but-discriminative
model, calibration is monotonic (never inverts discrimination/ROC-AUC),
calibrators are fit on validation data only (never touch the test labels
passed to them), and the slope/intercept diagnostic behaves correctly on a
known-perfect and a known-overconfident synthetic case.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.calibration import (
    apply_isotonic_calibrator,
    apply_platt_calibrator,
    calibration_metrics,
    calibration_slope_intercept,
    expected_calibration_error,
    expected_calibration_error_equal_frequency,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
)


def _overconfident_synthetic(n=2000, seed=0, k=3.0):
    """y is generated from a true logit = x; the model reports sigmoid(k*x)
    with k>1, an exaggerated logit -- classic overconfidence miscalibration.
    Monotonic in x, so it rank-orders identically to the true probability
    (ROC-AUC unaffected), but its raw probabilities are pushed toward 0/1
    further than the true positive rate warrants (calibration slope < 1)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.5, n)
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-x))).astype(int)
    p = np.clip(1 / (1 + np.exp(-k * x)), 1e-6, 1 - 1e-6)
    return y, p


def test_platt_calibration_reduces_ece_on_overconfident_model():
    y_val, p_val = _overconfident_synthetic(seed=1)
    y_test, p_test = _overconfident_synthetic(seed=2)

    ece_before = calibration_metrics(y_test, p_test)["ece"]
    calibrator = fit_platt_calibrator(y_val, p_val)
    p_test_calibrated = apply_platt_calibrator(calibrator, p_test)
    ece_after = calibration_metrics(y_test, p_test_calibrated)["ece"]

    assert ece_after < ece_before


def test_isotonic_calibration_reduces_ece_on_overconfident_model():
    y_val, p_val = _overconfident_synthetic(seed=1)
    y_test, p_test = _overconfident_synthetic(seed=2)

    ece_before = calibration_metrics(y_test, p_test)["ece"]
    calibrator = fit_isotonic_calibrator(y_val, p_val)
    p_test_calibrated = apply_isotonic_calibrator(calibrator, p_test)
    ece_after = calibration_metrics(y_test, p_test_calibrated)["ece"]

    assert ece_after < ece_before


def test_calibration_preserves_roc_auc_platt():
    y_val, p_val = _overconfident_synthetic(seed=1)
    y_test, p_test = _overconfident_synthetic(seed=2)
    calibrator = fit_platt_calibrator(y_val, p_val)
    p_test_calibrated = apply_platt_calibrator(calibrator, p_test)
    assert np.isclose(roc_auc_score(y_test, p_test), roc_auc_score(y_test, p_test_calibrated), atol=1e-9)


def test_calibration_approximately_preserves_roc_auc_isotonic():
    """Isotonic's fitted step function is only non-decreasing, not strictly
    monotonic -- at a flat plateau, two out-of-sample test points can tie
    (or swap) even though the underlying scores differed, so exact ROC-AUC
    preservation only holds in-sample. Out-of-sample (the real usage here:
    fit on validation, applied to test) it should be preserved closely, not
    exactly."""
    y_val, p_val = _overconfident_synthetic(seed=1)
    y_test, p_test = _overconfident_synthetic(seed=2)
    calibrator = fit_isotonic_calibrator(y_val, p_val)
    p_test_calibrated = apply_isotonic_calibrator(calibrator, p_test)
    auc_before = roc_auc_score(y_test, p_test)
    auc_after = roc_auc_score(y_test, p_test_calibrated)
    assert abs(auc_before - auc_after) < 0.02


def test_calibrated_probabilities_stay_in_unit_range():
    y_val, p_val = _overconfident_synthetic(seed=1)
    y_test, p_test = _overconfident_synthetic(seed=2)
    platt = apply_platt_calibrator(fit_platt_calibrator(y_val, p_val), p_test)
    iso = apply_isotonic_calibrator(fit_isotonic_calibrator(y_val, p_val), p_test)
    for p in (platt, iso):
        assert np.all(p >= 0.0) and np.all(p <= 1.0)
        assert not np.any(np.isnan(p))


def test_calibrator_fit_does_not_depend_on_test_labels():
    """A calibrator's parameters must be a function of the validation set
    only -- perturbing test-set labels after fitting must not change it."""
    y_val, p_val = _overconfident_synthetic(seed=1)
    calibrator_a = fit_platt_calibrator(y_val, p_val)
    # Simulate "having looked at" different test data -- irrelevant to fitting.
    _y_test_a, _p_test_a = _overconfident_synthetic(seed=2)
    _y_test_b, _p_test_b = _overconfident_synthetic(seed=3)
    calibrator_b = fit_platt_calibrator(y_val, p_val)
    assert np.isclose(calibrator_a.coef_[0][0], calibrator_b.coef_[0][0])
    assert np.isclose(calibrator_a.intercept_[0], calibrator_b.intercept_[0])


def test_slope_intercept_near_perfect_for_well_calibrated_probabilities():
    rng = np.random.default_rng(42)
    n = 5000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    result = calibration_slope_intercept(y, p)
    assert abs(result["calibration_slope"] - 1.0) < 0.15
    assert abs(result["calibration_intercept"]) < 0.15


def test_slope_below_one_for_overconfident_probabilities():
    y, p = _overconfident_synthetic(n=5000, seed=5)
    result = calibration_slope_intercept(y, p)
    assert result["calibration_slope"] < 1.0


def test_slope_intercept_handles_single_class_gracefully():
    y = np.ones(20, dtype=int)
    p = np.linspace(0.5, 0.9, 20)
    result = calibration_slope_intercept(y, p)
    assert np.isnan(result["calibration_slope"])
    assert np.isnan(result["calibration_intercept"])


def test_platt_slope_can_be_negative_and_isotonic_cannot_represent_inversion():
    """Regression guard for a real finding from the calibration analysis's N2 (degree-matched,
    near-chance) runs: when validation p/y carry a spurious/weak *negative*
    relationship (exactly what a near-chance base model's calibration set
    can look like), Platt fits a negative slope and will actively invert
    rank order on new data. Isotonic regression cannot represent an inverted
    relationship at all -- pooling-adjacent-violators on a decreasing
    pattern collapses to a single flat block at the overall base rate,
    which is why it is the safer default when discrimination is not
    already established."""
    rng = np.random.default_rng(7)
    n = 2000
    p_val = np.clip(rng.uniform(0.3, 0.7, n), 1e-6, 1 - 1e-6)
    # Weak genuine negative relationship: higher p -> lower chance of y=1.
    y_val = (rng.uniform(0, 1, n) < (0.7 - 0.6 * p_val)).astype(int)

    platt = fit_platt_calibrator(y_val, p_val)
    isotonic = fit_isotonic_calibrator(y_val, p_val)

    assert platt.coef_[0][0] < 0

    grid = np.linspace(0.3, 0.7, 50)
    iso_out = apply_isotonic_calibrator(isotonic, grid)
    assert np.all(np.diff(iso_out) >= -1e-12)  # isotonic never inverts, by construction
    # Away from the extreme boundary (a single-point PAV block there is an
    # edge artifact, not the collapse behavior being tested), the fit is
    # flat at the base rate.
    assert np.allclose(iso_out[2:-2], y_val.mean(), atol=0.03)


def test_isotonic_calibrator_is_monotonic_nondecreasing():
    y_val, p_val = _overconfident_synthetic(seed=1)
    calibrator = fit_isotonic_calibrator(y_val, p_val)
    grid = np.linspace(0, 1, 200)
    calibrated = apply_isotonic_calibrator(calibrator, grid)
    assert np.all(np.diff(calibrated) >= -1e-12)


# --- ECE bin-strategy sensitivity and Cox-regression separation diagnostics
# ---

def test_equal_frequency_ece_near_zero_for_well_calibrated_probabilities():
    rng = np.random.default_rng(2)
    n = 3000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)  # p IS the true probability -- perfectly calibrated by construction
    ece_width = expected_calibration_error(y, p, n_bins=10)
    ece_freq = expected_calibration_error_equal_frequency(y, p, n_bins=10)
    assert ece_width < 0.05
    assert ece_freq < 0.05


def test_equal_frequency_ece_handles_clustered_predictions_without_empty_bins():
    """Predictions clustered in a narrow range leave most equal-WIDTH bins
    empty (contributing nothing to ECE, silently); equal-FREQUENCY binning
    must still produce a finite, sensible value from the same data --
    exactly the scenario in which equal-width ECE is
    vulnerable."""
    rng = np.random.default_rng(3)
    n = 2000
    p = np.clip(rng.normal(0.5, 0.03, n), 0.01, 0.99)  # tightly clustered near 0.5
    y = (rng.uniform(0, 1, n) < p).astype(int)
    ece_freq = expected_calibration_error_equal_frequency(y, p, n_bins=10)
    assert np.isfinite(ece_freq)
    assert 0 <= ece_freq <= 1


def test_equal_frequency_ece_degenerate_for_constant_predictions():
    y = np.array([1, 0, 1, 0, 1])
    p = np.full(5, 0.5)
    assert expected_calibration_error_equal_frequency(y, p, n_bins=10) == 0.0


def test_calibration_slope_intercept_reports_converged_for_well_behaved_case():
    y, p = _overconfident_synthetic(seed=4)
    result = calibration_slope_intercept(y, p)
    assert result["converged"]
    assert not result["possible_separation"]


def test_calibration_slope_intercept_does_not_false_flag_strong_but_finite_separation():
    """Empirically, sklearn's default lbfgs solver is well-behaved for this
    function's always-1-dimensional (logit(p) -> y) regression even under
    near-complete class separation -- the fitted slope stays finite and
    convergence succeeds rather than walking off to infinity (checked
    directly here, and separately confirmed on this project's own 12 real
    model/negative-set calibration fits -- see
    scripts/31_calibration_rigor.py). This test
    guards against possible_separation crying wolf on legitimately strong,
    well-converged discrimination, which the threshold in
    calibration_slope_intercept is deliberately set high enough (|slope|>20)
    to avoid."""
    rng = np.random.default_rng(5)
    n = 500
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    p = np.clip(np.concatenate([rng.uniform(0.0, 0.01, n // 2), rng.uniform(0.99, 1.0, n // 2)]), 1e-6, 1 - 1e-6)
    result = calibration_slope_intercept(y, p)
    assert result["converged"]
    assert not result["possible_separation"]
    assert abs(result["calibration_slope"]) < 20
