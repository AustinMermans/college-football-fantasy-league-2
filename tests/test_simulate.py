import numpy as np

from cfb_fantasy.simulate import _simulate_regular_outcomes


def test_correlated_outcomes_preserve_marginals_and_share_team_form():
    outcomes, _ = _simulate_regular_outcomes(
        np.array([0, 0]),
        np.array([1, 2]),
        np.array([0.60, 0.60]),
        simulations=100_000,
        n_teams=3,
        rng=np.random.default_rng(7),
        season_strength_correlation=0.10,
    )

    assert np.allclose(outcomes.mean(axis=0), 0.60, atol=0.01)
    assert np.cov(outcomes.astype(float), rowvar=False)[0, 1] > 0.002


def test_correlated_outcomes_reject_invalid_correlation():
    try:
        _simulate_regular_outcomes(
            np.array([0]),
            np.array([1]),
            np.array([0.50]),
            simulations=2,
            n_teams=2,
            rng=np.random.default_rng(7),
            season_strength_correlation=1.0,
        )
    except ValueError as error:
        assert "correlation" in str(error)
    else:
        raise AssertionError("invalid correlation should be rejected")
