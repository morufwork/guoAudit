"""Handcrafted feature extraction and pair-matrix construction."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration import expected_calibration_error
from src.evaluation.metrics import classification_metrics
from src.features.aac import compute_aac, feature_names as aac_names
from src.features.build_pair_features import build_pair_matrix
from src.features.ctd import compute_ctd, feature_names as ctd_names
from src.features.physicochemical import compute_physicochemical

SEQ_A = "MKTLLILAVLLTAAVCADASGDKSGDKSGDKSAAAAAAMKTLLILAVLLT"
SEQ_B = "GATTACAGATTACAMKTLLILAVLLTAAVCGGGGSSSSTTTTKKKKRRRR"


def test_aac_sums_to_one_and_matches_names():
    v = compute_aac(SEQ_A)
    assert len(v) == len(aac_names()) == 20
    assert v.sum() == pytest.approx(1.0)
    assert (v >= 0).all()


def test_ctd_shape_and_range():
    v = compute_ctd(SEQ_A)
    assert len(v) == len(ctd_names()) == 147
    assert (v >= 0).all() and (v <= 1).all()
    assert np.isfinite(v).all()


def test_physicochemical_finite():
    v = compute_physicochemical(SEQ_A)
    assert len(v) == 8
    assert np.isfinite(v).all()


def test_build_pair_matrix_is_symmetric_under_protein_order():
    proteins = pd.DataFrame({"protein_id": ["A", "B"], "sequence": [SEQ_A, SEQ_B]})
    feats = pd.DataFrame({"protein_id": ["A", "B"], "f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    forward = pd.DataFrame({"protein_a": ["A"], "protein_b": ["B"], "label": [1]})
    backward = pd.DataFrame({"protein_a": ["B"], "protein_b": ["A"], "label": [1]})
    x_fwd, _ = build_pair_matrix(forward, feats, fusion="combined")
    x_bwd, _ = build_pair_matrix(backward, feats, fusion="combined")
    np.testing.assert_allclose(x_fwd, x_bwd)


def test_classification_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y_true, y_prob)
    assert m["roc_auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["mcc"] == 1.0


def test_ece_zero_for_perfectly_calibrated_probabilities():
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, size=20000)
    y_true = (rng.uniform(0, 1, size=20000) < y_prob).astype(int)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.02
