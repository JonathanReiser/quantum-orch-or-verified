"""
tests/test_contestedness.py — pin the contestedness benchmark.

The important test here is the last one: it asserts that the *pooled* content
signal does not survive conditioning on the DAO. That is the finding, and it is
the one a future change could most easily and most misleadingly erase.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dao_governance"))

from benchmark_contestedness import auc, fit_logistic, predict_logistic, run  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "snapshot_dao_dataset.json")


@pytest.fixture(scope="module")
def out():
    return run(DATA)


def test_auc_is_correct_on_known_cases():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert auc([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.0)
    assert auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)  # all ties


def test_logistic_recovers_a_planted_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2))
    y = (1.5 * x[:, 0] - 1.0 * x[:, 1] + rng.normal(scale=0.3, size=500) > 0).astype(float)
    m = fit_logistic(x, y, l2=1e-3)
    assert auc(y.astype(bool), predict_logistic(m, x)) > 0.9
    assert m["w"][0] > 0 and m["w"][1] < 0


def test_constant_predictor_has_chance_auc(out):
    assert out["results"]["base_rate_constant"]["auc"] == pytest.approx(0.5, abs=1e-9)


def test_base_rate_shifts_across_the_temporal_split(out):
    """Why AUC is the headline and accuracy would mislead."""
    s = out["split"]
    assert s["train_base_rate"] > s["test_base_rate"]
    assert s["n_train"] + s["n_test"] == s["n_total"] == 905


def test_contestedness_is_predictable_above_chance(out):
    """Unlike YES-share, this task has real signal — modest but significant."""
    full = out["results"]["logistic_all_features"]
    assert full["auc"] > 0.6
    assert full["auc_ci95"][0] > 0.5          # CI excludes chance
    assert full["auc_p_gt_0.5"] < 0.05


def test_the_signal_is_venue_not_proposal(out):
    """THE FINDING. Pooled, a content-only model looks predictive (AUC ~0.65).
    Within a single DAO it collapses to chance or below, because most DAOs use a
    fixed voting window and duration_days is therefore a venue fingerprint."""
    pooled = out["results"]["logistic_content_only"]["auc"]
    assert pooled > 0.6

    within = [w["content_only_auc"] for w in out["within_dao"].values() if w]
    assert len(within) >= 4
    assert float(np.median(within)) < 0.55          # collapses towards chance
    assert float(np.median(within)) < pooled - 0.1  # and far below the pooled value


def test_fixed_voting_windows_explain_the_confound(out):
    """Several DAOs show identical median duration for contested and uncontested
    proposals — a fixed window, so duration cannot be about the proposal."""
    identical = [d for d, w in out["within_dao"].items()
                 if w and w["median_duration_contested"] == w["median_duration_uncontested"]]
    assert len(identical) >= 2
