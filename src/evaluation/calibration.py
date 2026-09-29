"""Probability calibration metrics (discrimination
and calibration are separate properties — reported here alongside classification
metrics from the conventional baseline onward, never substituted for one another)."""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-WIDTH binning (bin edges fixed at 1/n_bins-wide intervals of
    [0, 1]), the more common ECE convention but one that can leave bins
    sparse or empty when predictions cluster in a narrow probability range
    -- see `expected_calibration_error_equal_frequency` for the alternative
    (ECE is bin-strategy- and bin-count-
    dependent, so both the strategy and a bin-count sensitivity check are
    reported explicitly, not just a single "ECE=10 bins" number)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def expected_calibration_error_equal_frequency(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-FREQUENCY (quantile) binning: bin edges chosen so each bin
    holds ~n/n_bins predictions, trading fixed-width interpretability for
    guaranteed-nonempty bins -- the standard alternative to equal-width ECE,
    reported alongside it rather than assuming
    one convention is self-evidently correct. Ties in y_prob can still
    leave bins uneven; duplicate quantile edges are dropped."""
    n = len(y_true)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(y_prob, quantiles))
    if len(bin_edges) < 2:
        return 0.0  # all predictions identical -- perfectly "calibrated" to a single point, degenerate
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, len(bin_edges) - 2)
    ece = 0.0
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Returns (bin_confidence, bin_accuracy, bin_counts) for a reliability diagram."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)
    bin_confidence, bin_accuracy, bin_counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        bin_counts.append(int(mask.sum()))
        if mask.any():
            bin_confidence.append(float(y_prob[mask].mean()))
            bin_accuracy.append(float(y_true[mask].mean()))
        else:
            bin_confidence.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
            bin_accuracy.append(float("nan"))
    return np.array(bin_confidence), np.array(bin_accuracy), np.array(bin_counts)


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Cox calibration regression: fit y ~ intercept + slope * logit(p) with an
    unpenalized logistic regression on the model's own test predictions (a
    diagnostic computed directly on the evaluation set, like Brier/ECE --
    NOT a correction fit on it, which would leak test information
    into the model). Slope=1, intercept=0 is perfect calibration; slope<1 indicates
    overconfident predictions (probabilities too extreme). Fit on
    logit(p) -- not raw p -- specifically because Cox's own regression is
    defined on the logit scale (a slope of 1 in logit-space, not raw-
    probability space, is what "perfect calibration" means here); extreme
    p near 0 or 1 are clipped before the logit transform (`_logit`,
    eps=1e-6) so no prediction produces +/-inf. `possible_separation` flags
    the main numerical risk directly: an unpenalized logistic
    regression's MLE diverges under (quasi-)complete separation -- a real
    risk when discrimination is strong on a small test set -- so a
    non-convergence warning or an implausibly large fitted |slope| (>20,
    the "walked off to infinity" signature) is surfaced rather than
    silently returned as a normal-looking number."""
    x = _logit(np.asarray(y_prob, dtype=float)).reshape(-1, 1)
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return {"calibration_slope": float("nan"), "calibration_intercept": float("nan"),
                "converged": False, "possible_separation": False}
    model = LogisticRegression(penalty=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    slope = float(model.coef_[0][0])
    return {
        "calibration_slope": slope,
        "calibration_intercept": float(model.intercept_[0]),
        "converged": converged,
        "possible_separation": bool(abs(slope) > 20 or not converged),
    }


def fit_platt_calibrator(y_val: np.ndarray, p_val: np.ndarray) -> LogisticRegression:
    """Platt/sigmoid calibration: fit ONLY on validation predictions (never
    the final test set).

    Caveat (observed empirically in the calibration analysis's N2/degree-matched runs, not
    hypothetical): if the validation set carries no genuine relationship
    between p_val and y_val (a near-chance base model), the fitted slope
    coefficient is noise and can come out negative -- which *inverts* the
    already-uninformative ranking on new data and can make test ROC-AUC
    worse than the raw scores, not just miscalibrated. Isotonic regression's
    monotonicity constraint (see fit_isotonic_calibrator) cannot invert in
    this way -- a near-zero true relationship makes it flatten toward the
    base rate instead. This is a real, reportable asymmetry between the two
    methods, not a defect in either implementation."""
    x = _logit(np.asarray(p_val, dtype=float)).reshape(-1, 1)
    model = LogisticRegression(penalty=None)
    model.fit(x, np.asarray(y_val))
    return model


def apply_platt_calibrator(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    x = _logit(np.asarray(p, dtype=float)).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def fit_isotonic_calibrator(y_val: np.ndarray, p_val: np.ndarray) -> IsotonicRegression:
    """Isotonic calibration: fit ONLY on validation predictions (never the
    final test set). `out_of_bounds="clip"` so test-set probabilities outside
    the validation set's observed range don't extrapolate wildly."""
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(np.asarray(p_val, dtype=float), np.asarray(y_val))
    return model


def apply_isotonic_calibrator(model: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return model.transform(np.asarray(p, dtype=float))
