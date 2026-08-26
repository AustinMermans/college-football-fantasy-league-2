import pandas as pd

from cfb_fantasy.tracker import score_completed_games


def test_score_completed_games_honors_postseason_categories():
    teams = pd.DataFrame([{"team_id": "1", "team": "Alpha"}])
    games = pd.DataFrame(
        [
            {
                "completed": True,
                "home_points": 21,
                "away_points": 14,
                "home_id": "1",
                "away_id": "x",
                "season_type": season_type,
                "event_name": name,
                "notes": notes,
            }
            for season_type, name, notes in [
                ("regular", "Alpha vs Beta", ""),
                ("regular", "Conference Championship", ""),
                ("postseason", "Orange Bowl", "College Football Playoff Quarterfinal"),
                ("postseason", "Holiday Bowl", ""),
            ]
        ]
    )
    draft = pd.DataFrame([{"manager": 1, "team": "Alpha"}])
    scoring = {
        "regular_season_win": 1.0,
        "conference_championship_win": 1.0,
        "playoff_win": 1.0,
        "non_playoff_bowl_win": 0.0,
    }

    standings = score_completed_games(games, teams, draft, scoring)

    assert standings.iloc[0]["points"] == 3.0


def test_score_completed_games_rejects_unknown_or_duplicate_teams():
    teams = pd.DataFrame([{"team_id": "1", "team": "Alpha"}])
    games = pd.DataFrame(
        columns=["completed", "home_points", "away_points", "home_id", "away_id"]
    )
    for draft in [
        pd.DataFrame([{"manager": 1, "team": "Missing"}]),
        pd.DataFrame(
            [{"manager": 1, "team": "Alpha"}, {"manager": 2, "team": "Alpha"}]
        ),
    ]:
        try:
            score_completed_games(games, teams, draft)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid draft should be rejected")
