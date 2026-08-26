import numpy as np
import pandas as pd

from cfb_fantasy.draft import (
    league_win_recommendations,
    live_recommendations,
    make_draft_board,
    recommendations_for_slot,
    validate_live_picks,
)


def test_snake_order_reverses_even_rounds():
    projections = pd.DataFrame(
        {
            "team": [f"T{i}" for i in range(1, 9)],
            "conference": ["C"] * 8,
            "expected_fantasy_points": list(range(8, 0, -1)),
            "expected_regular_points": list(range(8, 0, -1)),
            "expected_conference_title_points": [0] * 8,
            "expected_playoff_points": [0] * 8,
            "expected_postseason_points": [0] * 8,
            "pre_playoff_rank": list(range(1, 9)),
            "playoff_rank_lift": [0] * 8,
            "p10_fantasy_points": [0] * 8,
            "p90_fantasy_points": [10] * 8,
            "expected_regular_wins": list(range(8, 0, -1)),
            "playoff_probability": [0] * 8,
            "average_opponent_rating": [1500] * 8,
            "schedule_difficulty_rank": list(range(1, 9)),
        }
    )
    board, grid = make_draft_board(projections, managers=4, teams_per_manager=2)
    assert board["manager_slot"].tolist() == [1, 2, 3, 4, 4, 3, 2, 1]
    targets = recommendations_for_slot(board, 4, 2, draft_slot=1)
    assert targets["pick_number"].tolist() == [1, 8]
    assert grid.loc[1, "slot_1"] == "T8"


def test_live_recommendations_remove_picks_and_identify_turn():
    projections = pd.DataFrame(
        {
            "team": [f"T{i}" for i in range(1, 9)],
            "expected_fantasy_points": list(range(8, 0, -1)),
        }
    )
    picks = pd.DataFrame({"team": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]})
    status, available = live_recommendations(
        projections,
        picks,
        managers=4,
        teams_per_manager=2,
        draft_slot=1,
        count=5,
    )
    assert status["next_pick"] == 8
    assert status["our_turn"] is True
    assert available.iloc[0]["team"] == "T8"


def test_live_recommendations_flag_teams_before_next_turn():
    projections = pd.DataFrame(
        {
            "team": [f"T{i}" for i in range(1, 9)],
            "expected_fantasy_points": list(range(8, 0, -1)),
        }
    )
    status, available = live_recommendations(
        projections,
        pd.DataFrame({"team": ["T1"]}),
        managers=4,
        teams_per_manager=2,
        draft_slot=1,
        count=4,
    )
    assert status["next_our_pick"] == 8
    assert status["picks_until_our_turn"] == 6
    assert set(available["draft_action"]) == {
        "BEST AVAILABLE",
        "UNLIKELY TO RETURN",
    }


def test_live_recommendations_marks_full_between_turn_boundary():
    projections = pd.DataFrame(
        {"team": [f"T{i}" for i in range(1, 10)], "expected_fantasy_points": range(9, 0, -1)}
    )
    status, available = live_recommendations(
        projections,
        pd.DataFrame({"team": []}),
        managers=4,
        teams_per_manager=2,
        draft_slot=1,
        count=8,
    )
    assert status["our_turn"] is True
    assert available.iloc[6]["draft_action"] == "UNLIKELY TO RETURN"
    assert available.iloc[7]["draft_action"] == "MAY RETURN"


def test_validate_live_picks_rejects_corrupt_order():
    projections = pd.DataFrame({"team": ["T1", "T2"]})
    picks = pd.DataFrame(
        {"pick_number": [2], "manager_slot": [1], "team": ["T1"]}
    )
    try:
        validate_live_picks(projections, picks, managers=2, teams_per_manager=1)
    except ValueError as error:
        assert "sequential" in str(error)
    else:
        raise AssertionError("corrupt order should be rejected")


def test_league_win_recommendations_uses_joint_roster_outcomes():
    projections = pd.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "team_id": ["1", "2", "3", "4"],
            "expected_fantasy_points": [6.0, 6.0, 5.0, 5.0],
        }
    )
    samples = pd.DataFrame(
        {
            "1": [10, 0, 10, 0],
            "2": [6, 6, 6, 6],
            "3": [5, 5, 5, 5],
            "4": [0, 10, 0, 10],
        }
    ).to_numpy()

    values = league_win_recommendations(
        projections,
        pd.DataFrame(columns=["team"]),
        samples,
        np.array(["1", "2", "3", "4"]),
        managers=2,
        teams_per_manager=2,
        draft_slot=1,
        count=2,
    )

    assert values.iloc[0]["team"] == "B"
    assert values.iloc[0]["league_win_probability"] > values.iloc[1][
        "league_win_probability"
    ]
