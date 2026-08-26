import numpy as np
import pandas as pd

from cfb_fantasy.market import (
    MarketConsensus,
    apply_market_consensus,
    update_fpi_probabilities,
)


def test_current_fpi_refreshes_only_fbs_matchups_and_respects_neutral_site():
    games = pd.DataFrame(
        [
            {
                "game_id": "1",
                "home_id": "a",
                "away_id": "b",
                "neutral_site": False,
                "home_win_probability": 0.5,
            },
            {
                "game_id": "2",
                "home_id": "a",
                "away_id": "fcs",
                "neutral_site": True,
                "home_win_probability": 0.8,
            },
        ]
    )
    fpi = pd.DataFrame(
        [
            {"team_id": "a", "fpi": 10.0},
            {"team_id": "b", "fpi": 5.0},
        ]
    )

    updated = update_fpi_probabilities(
        games, fpi, logistic_scale=10.0, home_advantage=2.5
    )

    assert updated.loc[0, "home_win_probability"] > 0.5
    assert np.isclose(updated.loc[1, "home_win_probability"], 0.8)
    assert bool(updated.loc[0, "current_fpi_used"])
    assert not bool(updated.loc[1, "current_fpi_used"])


def test_apply_market_consensus_uses_lines_and_falls_back_without_one():
    games = pd.DataFrame(
        [
            {"game_id": "1", "fpi_home_win_probability": 0.6, "home_win_probability": 0.6},
            {"game_id": "2", "fpi_home_win_probability": 0.7, "home_win_probability": 0.7},
        ]
    )
    betting = pd.DataFrame(
        [{"game_id": "1", "home_team_spread": -7.0, "over_under": 50.0, "odds_source": "test"}]
    )
    fitted = MarketConsensus(
        mean=(0.0, 0.0),
        scale=(1.0, 1.0),
        coefficients=(1.0, 0.1),
        intercept=0.0,
        training_games=1000,
    )

    result = apply_market_consensus(games, betting, fitted)

    assert bool(result.loc[0, "market_line_used"])
    assert result.loc[0, "home_win_probability"] > 0.6
    assert not bool(result.loc[1, "market_line_used"])
    assert np.isclose(result.loc[1, "home_win_probability"], 0.7)
    assert np.isclose(result.loc[1, "away_win_probability"], 0.3)
