from __future__ import annotations

import numpy as np
import pandas as pd


def manager_for_pick(pick_number: int, managers: int) -> int:
    if pick_number < 1:
        raise ValueError("pick number must be positive")
    round_number = (pick_number - 1) // managers + 1
    position = (pick_number - 1) % managers + 1
    return position if round_number % 2 else managers + 1 - position


def picks_for_slot(managers: int, rounds: int, draft_slot: int) -> list[int]:
    if not 1 <= draft_slot <= managers:
        raise ValueError("draft slot must be between 1 and number of managers")
    return [
        (round_number - 1) * managers
        + (draft_slot if round_number % 2 else managers + 1 - draft_slot)
        for round_number in range(1, rounds + 1)
    ]


def make_draft_board(
    projections: pd.DataFrame, managers: int, teams_per_manager: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_picks = managers * teams_per_manager
    board = projections.head(total_picks).copy()
    board["pick_number"] = np.arange(1, len(board) + 1)
    board["round"] = (board["pick_number"] - 1) // managers + 1
    position = (board["pick_number"] - 1) % managers + 1
    board["manager_slot"] = np.where(
        board["round"] % 2 == 1, position, managers + 1 - position
    )
    replacement = (
        projections.iloc[total_picks]["expected_fantasy_points"]
        if len(projections) > total_picks
        else projections.iloc[-1]["expected_fantasy_points"]
    )
    board["value_over_replacement"] = (
        board["expected_fantasy_points"] - float(replacement)
    )
    columns = [
        "pick_number",
        "round",
        "manager_slot",
        "team",
        "conference",
        "expected_fantasy_points",
        "expected_regular_points",
        "expected_conference_title_points",
        "expected_playoff_points",
        "expected_postseason_points",
        "pre_playoff_rank",
        "playoff_rank_lift",
        "p10_fantasy_points",
        "p90_fantasy_points",
        "expected_regular_wins",
        "playoff_probability",
        "average_opponent_rating",
        "schedule_difficulty_rank",
        "value_over_replacement",
    ]
    board = board[columns]
    grid = board.pivot(index="round", columns="manager_slot", values="team")
    grid.columns = [f"slot_{int(value)}" for value in grid.columns]
    return board, grid.reset_index()


def recommendations_for_slot(
    board: pd.DataFrame, managers: int, teams_per_manager: int, draft_slot: int
) -> pd.DataFrame:
    if not 1 <= draft_slot <= managers:
        raise ValueError("draft slot must be between 1 and number of managers")
    pick_numbers = picks_for_slot(managers, teams_per_manager, draft_slot)
    return board[board["pick_number"].isin(pick_numbers)].copy()


def validate_live_picks(
    projections: pd.DataFrame,
    picks: pd.DataFrame,
    *,
    managers: int,
    teams_per_manager: int,
) -> None:
    if "team" not in picks:
        raise ValueError("draft picks must contain a team column")
    teams = picks["team"]
    if teams.isna().any() or teams.astype(str).str.strip().eq("").any():
        raise ValueError("draft picks contain blank teams")
    picked = teams.astype(str)
    unknown = set(picked).difference(set(projections["team"].astype(str)))
    if unknown:
        raise ValueError(f"unknown drafted teams: {sorted(unknown)}")
    if picked.duplicated().any():
        raise ValueError("draft picks contain duplicate teams")
    total_picks = managers * teams_per_manager
    if len(picks) > total_picks:
        raise ValueError(f"draft has more than the configured {total_picks} picks")
    if "pick_number" in picks:
        expected = list(range(1, len(picks) + 1))
        if pd.to_numeric(picks["pick_number"], errors="coerce").tolist() != expected:
            raise ValueError("pick_number must be sequential starting at 1")
    if "manager_slot" in picks:
        expected = [manager_for_pick(number, managers) for number in range(1, len(picks) + 1)]
        if pd.to_numeric(picks["manager_slot"], errors="coerce").tolist() != expected:
            raise ValueError("manager_slot does not match the configured snake order")


def live_recommendations(
    projections: pd.DataFrame,
    picks: pd.DataFrame,
    *,
    managers: int,
    teams_per_manager: int,
    draft_slot: int,
    count: int = 15,
) -> tuple[dict[str, int | bool | None], pd.DataFrame]:
    validate_live_picks(
        projections,
        picks,
        managers=managers,
        teams_per_manager=teams_per_manager,
    )
    picked = set(picks["team"].astype(str))
    next_pick = len(picks) + 1
    total_picks = managers * teams_per_manager
    our_picks = picks_for_slot(managers, teams_per_manager, draft_slot)
    remaining_our_picks = [value for value in our_picks if value >= next_pick]
    next_our_pick = remaining_our_picks[0] if remaining_our_picks else None
    following_our_pick = remaining_our_picks[1] if len(remaining_our_picks) > 1 else None
    if next_our_pick is None:
        picks_between_turns = 0
    elif following_our_pick is None:
        picks_between_turns = max(0, total_picks - next_our_pick)
    else:
        picks_between_turns = following_our_pick - next_our_pick - 1
    if next_our_pick is None:
        survival_horizon = 0
    elif next_pick == next_our_pick:
        survival_horizon = picks_between_turns
    else:
        survival_horizon = next_our_pick - next_pick
    available = projections[~projections["team"].isin(picked)].copy().head(count)
    available.insert(0, "available_rank", np.arange(1, len(available) + 1))
    unavailable_before_next_turn = survival_horizon + int(next_pick == next_our_pick)
    available.insert(
        1,
        "draft_action",
        np.where(
            available["available_rank"].eq(1),
            "BEST AVAILABLE",
            np.where(
                available["available_rank"].le(unavailable_before_next_turn),
                "UNLIKELY TO RETURN",
                "MAY RETURN",
            ),
        ),
    )
    status: dict[str, int | bool | None] = {
        "next_pick": next_pick,
        "current_manager_slot": manager_for_pick(next_pick, managers)
        if next_pick <= total_picks
        else None,
        "our_turn": next_pick in our_picks,
        "next_our_pick": next_our_pick,
        "following_our_pick": following_our_pick,
        "picks_between_our_turns": picks_between_turns,
        "picks_until_our_turn": max(0, (next_our_pick or next_pick) - next_pick),
        "picks_recorded": len(picks),
    }
    return status, available


def league_win_recommendations(
    projections: pd.DataFrame,
    picks: pd.DataFrame,
    fantasy_points: np.ndarray,
    sample_team_ids: np.ndarray,
    *,
    managers: int,
    teams_per_manager: int,
    draft_slot: int,
    count: int = 15,
) -> pd.DataFrame:
    """Evaluate candidates by first-place probability under board-order completion."""
    validate_live_picks(
        projections,
        picks,
        managers=managers,
        teams_per_manager=teams_per_manager,
    )
    next_pick = len(picks) + 1
    total_picks = managers * teams_per_manager
    if next_pick > total_picks:
        return pd.DataFrame()
    if manager_for_pick(next_pick, managers) != draft_slot:
        return pd.DataFrame()
    if fantasy_points.ndim != 2 or fantasy_points.shape[1] != len(sample_team_ids):
        raise ValueError("simulation sample shape does not match team ids")

    sample_column = {
        str(team_id): column for column, team_id in enumerate(sample_team_ids)
    }
    team_id_by_name = dict(
        zip(projections["team"].astype(str), projections["team_id"].astype(str))
    )
    if set(team_id_by_name.values()).difference(sample_column):
        raise ValueError("simulation samples do not cover every projected team")

    drafted = picks["team"].astype(str).tolist()
    base_rosters: list[list[str]] = [[] for _ in range(managers)]
    for pick_number, team in enumerate(drafted, start=1):
        manager = manager_for_pick(pick_number, managers)
        base_rosters[manager - 1].append(team)
    available = [
        team for team in projections["team"].astype(str).tolist() if team not in drafted
    ]
    rows = []
    for candidate in available[:count]:
        rosters = [teams.copy() for teams in base_rosters]
        remaining = [team for team in available if team != candidate]
        rosters[draft_slot - 1].append(candidate)
        for pick_number in range(next_pick + 1, total_picks + 1):
            if not remaining:
                break
            manager = manager_for_pick(pick_number, managers)
            rosters[manager - 1].append(remaining.pop(0))

        totals = np.zeros((fantasy_points.shape[0], managers), dtype=float)
        for manager_number, roster in enumerate(rosters):
            columns = [
                sample_column[team_id_by_name[team]] for team in roster
            ]
            if columns:
                totals[:, manager_number] = fantasy_points[:, columns].sum(axis=1)
        ours = totals[:, draft_slot - 1]
        other_totals = np.delete(totals, draft_slot - 1, axis=1)
        best = totals.max(axis=1)
        ties = np.isclose(totals, best[:, None]).sum(axis=1)
        win_share = np.where(np.isclose(ours, best), 1.0 / ties, 0.0)
        finish = 1 + (other_totals > ours[:, None]).sum(axis=1)
        rows.append(
            {
                "team": candidate,
                "league_win_probability": float(win_share.mean()),
                "expected_finish": float(finish.mean()),
                "expected_roster_points": float(ours.mean()),
                "expected_margin_to_best_opponent": float(
                    (ours - other_totals.max(axis=1)).mean()
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["league_win_probability", "expected_finish", "expected_roster_points"],
        ascending=[False, True, False],
    )
    result.insert(0, "league_value_rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)
