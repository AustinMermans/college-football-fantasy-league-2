from datetime import datetime, timezone

import pandas as pd

from cfb_fantasy.scoreboard import build_scoreboard, update_live_projections


def test_scoreboard_counts_points_and_builds_snake_board():
    teams = pd.DataFrame(
        [
            {"team_id": "1", "team": "Alpha", "display_name": "Alpha A", "conference": "ACC"},
            {"team_id": "2", "team": "Beta", "display_name": "Beta B", "conference": "SEC"},
        ]
    )
    games = pd.DataFrame(
        [
            {
                "start_date": "2026-08-20T00:00:00Z",
                "completed": True,
                "home_id": "1",
                "away_id": "2",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_points": 24,
                "away_points": 20,
                "season_type": "regular",
                "event_name": "Beta at Alpha",
                "notes": "",
                "neutral_site": False,
            },
            {
                "start_date": "2026-09-01T00:00:00Z",
                "completed": False,
                "home_id": "2",
                "away_id": "x",
                "home_team": "Beta",
                "away_team": "Gamma",
                "home_points": None,
                "away_points": None,
                "season_type": "regular",
                "event_name": "Gamma at Beta",
                "notes": "",
                "neutral_site": False,
            },
        ]
    )
    picks = pd.DataFrame(
        [
            {"pick_number": 1, "round": 1, "manager_slot": 1, "manager": "One", "team_id": "1", "team": "Alpha"},
            {"pick_number": 2, "round": 1, "manager_slot": 2, "manager": "Two", "team_id": "2", "team": "Beta"},
        ]
    )
    projections = pd.DataFrame(
        [
            {"team_id": "1", "overall_rank": 1, "expected_fantasy_points": 10.0, "playoff_probability": 0.5},
            {"team_id": "2", "overall_rank": 2, "expected_fantasy_points": 9.0, "playoff_probability": 0.2},
        ]
    )

    payload = build_scoreboard(
        games,
        teams,
        picks,
        projections,
        season=2026,
        manager_names=["One", "Two"],
        teams_per_manager=2,
        generated_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )

    assert payload["managers"][0]["name"] == "One"
    assert payload["managers"][0]["fantasyPoints"] == 1.0
    assert payload["managers"][0]["teams"][0]["expectedPlayoffPoints"] == 0.0
    assert payload["managers"][0]["teams"][0]["projectedExpectedPoints"] == 10.0
    assert payload["managers"][1]["teams"][0]["nextGame"]["opponent"] == "Gamma"
    assert [pick["managerSlot"] for pick in payload["draftBoard"]] == [1, 2, 2, 1]


def test_live_projections_replace_completed_games_with_outcomes():
    games = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "completed": True,
                "home_points": 17,
                "away_points": 20,
            }
        ]
    )
    probabilities = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_id": "1",
                "away_id": "2",
                "home_win_probability": 0.75,
                "away_win_probability": 0.25,
            }
        ]
    )
    projections = pd.DataFrame(
        [
            {
                "team_id": "1",
                "team": "Alpha",
                "expected_regular_wins": 0.75,
                "expected_regular_points": 0.75,
                "expected_points_before_playoff": 1.25,
                "expected_fantasy_points": 1.5,
            },
            {
                "team_id": "2",
                "team": "Beta",
                "expected_regular_wins": 0.25,
                "expected_regular_points": 0.25,
                "expected_points_before_playoff": 0.5,
                "expected_fantasy_points": 0.75,
            },
        ]
    )

    updated = update_live_projections(games, probabilities, projections).set_index(
        "team_id"
    )

    assert updated.loc["1", "expected_regular_points"] == 0.0
    assert updated.loc["1", "expected_fantasy_points"] == 0.75
    assert updated.loc["2", "expected_regular_points"] == 1.0
    assert updated.loc["2", "expected_fantasy_points"] == 1.5
