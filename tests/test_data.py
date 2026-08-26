import json
from pathlib import Path

import pandas as pd
import pytest

from cfb_fantasy.data import (
    attach_preseason_context,
    attach_historical_fpi,
    parse_espn_schedules,
    parse_fbs_teams,
    parse_fpi,
)


def test_attach_preseason_context_uses_current_talent_and_prior_epa():
    games = pd.DataFrame(
        [{"season": 2026, "home_id": "1", "away_id": "2"}]
    )
    talent = pd.DataFrame(
        [
            {"season": 2026, "team_id": "1", "talent_composite": 800.0, "blue_chip_ratio": 0.7},
            {"season": 2026, "team_id": "2", "talent_composite": 600.0, "blue_chip_ratio": 0.3},
        ]
    )
    summaries = pd.DataFrame(
        [
            {"season": 2025, "team_id": "1", "net_adj_epa": 0.2},
            {"season": 2025, "team_id": "2", "net_adj_epa": -0.1},
        ]
    )

    attached = attach_preseason_context(games, talent, summaries).iloc[0]

    assert attached["talent_composite_diff"] == 200.0
    assert attached["blue_chip_ratio_diff"] == pytest.approx(0.4)
    assert attached["net_adj_epa_diff"] == pytest.approx(0.3)


def test_parse_fbs_teams_recurses_through_divisions():
    payload = {
        "children": [
            {
                "shortName": "Example Conference",
                "isConference": True,
                "children": [
                    {
                        "shortName": "East",
                        "standings": {
                            "entries": [
                                {
                                    "team": {
                                        "id": str(team_id),
                                        "location": f"Team {team_id}",
                                        "displayName": f"Team {team_id} Mascot",
                                    }
                                }
                                for team_id in range(130)
                            ]
                        },
                    }
                ],
            }
        ]
    }
    teams = parse_fbs_teams(payload)
    assert len(teams) == 130
    assert set(teams["conference"]) == {"Example Conference"}


def test_parse_espn_schedule_deduplicates_and_marks_conference_game(tmp_path: Path):
    teams = pd.DataFrame(
        [
            {"team_id": "1", "team": "Alpha", "display_name": "Alpha A", "conference": "C"},
            {"team_id": "2", "team": "Beta", "display_name": "Beta B", "conference": "C"},
        ]
    )
    event = {
        "id": "g1",
        "date": "2026-09-01T00:00Z",
        "season": {"year": 2026},
        "seasonType": {"name": "Regular Season"},
        "week": {"number": 1},
        "competitions": [
            {
                "neutralSite": False,
                "notes": [{"headline": "Example note"}],
                "status": {"type": {"completed": False}},
                "competitors": [
                    {"id": "1", "homeAway": "home", "team": {"location": "Alpha"}},
                    {"id": "2", "homeAway": "away", "team": {"location": "Beta"}},
                ],
            }
        ],
    }
    paths = []
    for number in range(2):
        path = tmp_path / f"{number}.json"
        path.write_text(json.dumps({"events": [event]}))
        paths.append(path)
    games = parse_espn_schedules(paths, teams, 2026)
    assert len(games) == 1
    assert bool(games.iloc[0]["conference_game"])
    assert games.iloc[0]["home_team"] == "Alpha"
    assert games.iloc[0]["event_name"] == ""
    assert games.iloc[0]["notes"] == "Example note"


def test_parse_fpi_uses_named_metric_positions():
    names = [
        "fpi",
        "fpirank",
        "projectedw",
        "projectedl",
        "probmakeplayoffs",
        "probwinconf",
        "probwintitle",
    ]
    payload = {
        "categories": [{"name": "fpi", "names": names}],
        "teams": [
            {
                "team": {"id": str(team_id)},
                "categories": [
                    {
                        "name": "fpi",
                        "values": [10.0, team_id + 1, 8.0, 4.0, 25.0, 20.0, 5.0],
                    }
                ],
            }
            for team_id in range(130)
        ],
    }
    fpi = parse_fpi(payload)
    assert len(fpi) == 130
    assert fpi.iloc[0]["fpi_playoff_probability"] == 0.25
    assert fpi.iloc[0]["fpi_national_title_probability"] == 0.05


def test_attach_historical_fpi_uses_home_team_row(tmp_path: Path):
    path = tmp_path / "fpi.csv"
    pd.DataFrame(
        [
            {
                "game_id": "g1",
                "team_id": "1",
                "gameprojection": 72.0,
                "teampredptdiff": 6.5,
            },
            {
                "game_id": "g1",
                "team_id": "2",
                "gameprojection": 28.0,
                "teampredptdiff": -6.5,
            },
        ]
    ).to_csv(path, index=False)
    games = pd.DataFrame([{"game_id": "g1", "home_id": "1", "away_id": "2"}])

    attached = attach_historical_fpi(games, [path])

    assert attached.iloc[0]["fpi_home_probability"] == 0.72
    assert attached.iloc[0]["fpi_predicted_margin"] == 6.5
