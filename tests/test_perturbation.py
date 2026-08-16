"""Guards for src/metrics/perturbation.py (PSR/L2/Linf)."""

from __future__ import annotations

import numpy as np

from src.metrics.perturbation import l2, linf, psr


def test_zero_perturbation_gives_zero_everywhere():
    clean = np.random.default_rng(0).normal(size=(50, 20))
    assert np.allclose(psr(clean, clean), 0.0)
    assert np.allclose(l2(clean, clean), 0.0)
    assert np.allclose(linf(clean, clean), 0.0)


def test_l2_matches_hand_computed_value():
    clean = np.zeros((2, 4))
    perturbed = np.array([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    # 3-4-0-0 vector has L2 norm 5 (classic 3-4-5 triangle)
    np.testing.assert_allclose(l2(clean, perturbed), [5.0, 0.0])


def test_linf_picks_the_single_largest_deviation():
    clean = np.zeros((1, 5))
    perturbed = np.array([[1.0, -7.0, 2.0, 0.0, 3.0]])
    assert linf(clean, perturbed)[0] == 7.0


def test_psr_is_dimensionless_and_scale_invariant():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(30, 10)) + 5.0  # nonzero mean so L2 norm is well away from 0
    noise = rng.normal(scale=0.1, size=(30, 10))
    perturbed = clean + noise

    ratio_small = psr(clean, perturbed)
    ratio_big = psr(10 * clean, 10 * perturbed)  # same relative perturbation, scaled up 10x
    np.testing.assert_allclose(ratio_small, ratio_big, rtol=1e-9)


def test_psr_ranks_defenses_by_relative_cost_not_absolute_l2():
    # a defense with a bigger L2 budget isn't necessarily "more expensive" in
    # PSR terms if it's applied to a proportionally bigger signal — this is
    # exactly the comparison the defense project needs (same PSR budget,
    # compare GE degradation across defense methods).
    clean_quiet = np.full((1, 10), 1.0)
    clean_loud = np.full((1, 10), 100.0)
    same_l2_perturbation = np.full((1, 10), 0.5)  # same absolute L2 budget for both

    psr_quiet = psr(clean_quiet, clean_quiet + same_l2_perturbation)[0]
    psr_loud = psr(clean_loud, clean_loud + same_l2_perturbation)[0]
    assert psr_quiet > psr_loud  # same L2 cost is relatively far more expensive on the quiet signal


def test_all_three_return_one_value_per_trace():
    clean = np.zeros((7, 12))
    perturbed = np.ones((7, 12))
    assert psr(clean, perturbed).shape == (7,)
    assert l2(clean, perturbed).shape == (7,)
    assert linf(clean, perturbed).shape == (7,)
