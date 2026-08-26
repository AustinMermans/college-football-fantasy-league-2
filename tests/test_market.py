import numpy as np
import pandas as pd

from cfb_fantasy.market import MarketConsensus, apply_market_consensus


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
