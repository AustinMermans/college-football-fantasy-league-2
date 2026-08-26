import pandas as pd

from cfb_fantasy.features import (
    TeamState,
    build_pregame_features,
    build_preseason_features,
    matchup_features,
    regress_offseason,
    update_states,
)


def sample_games(second_home_points: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "1",
                "season": 2024,
                "start_date": pd.Timestamp("2024-09-01", tz="UTC"),
                "neutral_site": False,
                "home_id": "a",
                "away_id": "b",
                "home_team": "A",
                "away_team": "B",
                "home_points": 35,
                "away_points": 14,
            },
            {
                "game_id": "2",
                "season": 2024,
                "start_date": pd.Timestamp("2024-09-08", tz="UTC"),
                "neutral_site": False,
                "home_id": "a",
                "away_id": "c",
                "home_team": "A",
                "away_team": "C",
                "home_points": second_home_points,
                "away_points": 10,
            },
        ]
    )


def test_game_features_do_not_use_current_game_score():
    low, _ = build_pregame_features(sample_games(11))
    high, _ = build_pregame_features(sample_games(70))
    columns = [column for column in low.columns if column not in {"target"}]
    pd.testing.assert_series_equal(low.iloc[1][columns], high.iloc[1][columns])


def test_prior_score_changes_next_game_strength_features():
    features, _ = build_pregame_features(sample_games(11))
    assert features.iloc[1]["rating_diff"] > 0
    assert features.iloc[1]["margin_form_diff"] > 0


def test_offseason_resets_sample_experience_and_rest():
    state = regress_offseason(
        TeamState(games=50, last_played=pd.Timestamp("2025-12-01", tz="UTC"))
    )
    assert state.games == 0
    assert state.last_played is None


def test_offseason_preserves_history_and_records_rating_momentum():
    home, away = TeamState(), TeamState()
    update_states(
        home,
        away,
        42,
        10,
        pd.Timestamp("2025-09-01", tz="UTC"),
        neutral=False,
    )
    preseason = regress_offseason(home)
    assert preseason.total_games == 1
    assert preseason.long_margin_form > 0
    assert preseason.rating_momentum > 0


def test_matchup_features_identify_fbs_fcs_direction():
    features = matchup_features(
        TeamState(), TeamState(), home_is_fbs=True, away_is_fbs=False
    )
    assert features["fbs_status_diff"] == 1.0


def test_preseason_features_ignore_all_current_season_results():
    low, _ = build_preseason_features(sample_games(11))
    high, _ = build_preseason_features(sample_games(70))
    feature_columns = [
        "rating_diff",
        "margin_form_diff",
        "win_form_diff",
        "long_margin_diff",
        "long_win_diff",
        "fbs_status_diff",
        "history_depth_diff",
        "rating_momentum_diff",
        "experience_diff",
        "rest_diff",
        "home_field",
    ]
    pd.testing.assert_frame_equal(low[feature_columns], high[feature_columns])
