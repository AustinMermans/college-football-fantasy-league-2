from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .tracker import DEFAULT_SCORING, game_category


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _manager_for_pick(pick_number: int, managers: int) -> int:
    round_index, offset = divmod(pick_number - 1, managers)
    return offset + 1 if round_index % 2 == 0 else managers - offset


def update_live_projections(
    games: pd.DataFrame,
    game_probabilities: pd.DataFrame,
    projections: pd.DataFrame,
) -> pd.DataFrame:
    """Replace completed-game probabilities with outcomes and re-aggregate EV."""
    probabilities = game_probabilities.copy()
    probabilities["game_id"] = probabilities["game_id"].astype(str)
    probabilities["home_id"] = probabilities["home_id"].astype(str)
    probabilities["away_id"] = probabilities["away_id"].astype(str)

    probabilities = apply_completed_results(games, probabilities)

    home = probabilities.groupby("home_id")["home_win_probability"].sum()
    away = probabilities.groupby("away_id")["away_win_probability"].sum()
    expected = home.add(away, fill_value=0.0)

    frame = projections.copy()
    frame["team_id"] = frame["team_id"].astype(str)
    frame["live_expected_regular_points"] = frame["team_id"].map(expected)
    frame["live_expected_regular_points"] = frame[
        "live_expected_regular_points"
    ].fillna(frame["expected_regular_points"])
    delta = (
        frame["live_expected_regular_points"] - frame["expected_regular_points"]
    )
    frame["expected_regular_wins"] = frame["live_expected_regular_points"]
    frame["expected_regular_points"] = frame.pop("live_expected_regular_points")
    frame["expected_points_before_playoff"] += delta
    frame["expected_fantasy_points"] += delta
    frame = frame.sort_values(
        ["expected_fantasy_points", "team"], ascending=[False, True]
    ).reset_index(drop=True)
    frame["overall_rank"] = frame.index + 1
    return frame


def apply_completed_results(
    games: pd.DataFrame, game_probabilities: pd.DataFrame
) -> pd.DataFrame:
    """Set completed games to their realized win probabilities."""
    probabilities = game_probabilities.copy()
    probabilities["game_id"] = probabilities["game_id"].astype(str)
    if "away_win_probability" not in probabilities:
        probabilities["away_win_probability"] = (
            1.0 - probabilities["home_win_probability"]
        )
    results = games.copy()
    results["game_id"] = results["game_id"].astype(str)
    results = results[
        results["completed"].astype(bool)
        & results["home_points"].notna()
        & results["away_points"].notna()
    ][["game_id", "home_points", "away_points"]]
    probabilities = probabilities.merge(
        results, on="game_id", how="left", validate="one_to_one"
    )
    completed = probabilities["home_points"].notna()
    probabilities.loc[completed, "home_win_probability"] = (
        probabilities.loc[completed, "home_points"]
        > probabilities.loc[completed, "away_points"]
    ).astype(float)
    probabilities.loc[completed, "away_win_probability"] = (
        probabilities.loc[completed, "away_points"]
        > probabilities.loc[completed, "home_points"]
    ).astype(float)
    tied = completed & probabilities["home_points"].eq(probabilities["away_points"])
    probabilities.loc[tied, "home_win_probability"] = 0.0
    probabilities.loc[tied, "away_win_probability"] = 0.0
    probabilities.loc[completed, "probability_source"] = "completed_result"
    return probabilities


def build_scoreboard(
    games: pd.DataFrame,
    teams: pd.DataFrame,
    picks: pd.DataFrame,
    projections: pd.DataFrame,
    *,
    season: int,
    manager_names: list[str],
    teams_per_manager: int,
    scoring: dict[str, float] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    required = {"pick_number", "manager_slot", "manager", "team_id", "team"}
    missing = required.difference(picks.columns)
    if missing:
        raise ValueError(f"picks file missing columns: {sorted(missing)}")
    if picks["team"].astype(str).duplicated().any():
        raise ValueError("picks file contains duplicate teams")

    weights = DEFAULT_SCORING.copy()
    if scoring:
        weights.update({key: float(value) for key, value in scoring.items()})

    team_meta = teams.copy()
    team_meta["team_id"] = team_meta["team_id"].astype(str)
    meta_by_id = team_meta.set_index("team_id").to_dict("index")
    projection_rows = projections.copy()
    projection_rows["team_id"] = projection_rows["team_id"].astype(str)
    projection_by_id = projection_rows.set_index("team_id").to_dict("index")

    schedule = games.copy()
    schedule["home_id"] = schedule["home_id"].astype(str)
    schedule["away_id"] = schedule["away_id"].astype(str)
    schedule["start_date"] = pd.to_datetime(schedule["start_date"], utc=True)
    now = generated_at or datetime.now(timezone.utc)
    now_timestamp = pd.Timestamp(now)
    if now_timestamp.tzinfo is None:
        now_timestamp = now_timestamp.tz_localize("UTC")

    roster_teams: list[dict[str, Any]] = []
    recent_results: list[dict[str, Any]] = []
    for pick in picks.sort_values("pick_number").itertuples(index=False):
        team_id = str(pick.team_id)
        team_games = schedule[
            schedule["home_id"].eq(team_id) | schedule["away_id"].eq(team_id)
        ].sort_values("start_date")
        wins = losses = ties = 0
        fantasy_points = 0.0
        categories = {key: 0 for key in DEFAULT_SCORING}

        for game in team_games[team_games["completed"].astype(bool)].itertuples(index=False):
            is_home = str(game.home_id) == team_id
            team_score = game.home_points if is_home else game.away_points
            opponent_score = game.away_points if is_home else game.home_points
            if pd.isna(team_score) or pd.isna(opponent_score):
                continue
            if float(team_score) > float(opponent_score):
                wins += 1
                category = game_category(game)
                categories[category] += 1
                fantasy_points += weights[category]
                result = "W"
            elif float(team_score) < float(opponent_score):
                losses += 1
                result = "L"
            else:
                ties += 1
                result = "T"
            recent_results.append(
                {
                    "date": _iso(game.start_date),
                    "teamId": team_id,
                    "team": str(pick.team),
                    "manager": str(pick.manager),
                    "result": result,
                    "teamScore": float(team_score),
                    "opponentScore": float(opponent_score),
                    "opponent": str(game.away_team if is_home else game.home_team),
                }
            )

        future = team_games[
            ~team_games["completed"].astype(bool)
            & team_games["start_date"].ge(now_timestamp)
        ]
        next_game: dict[str, Any] | None = None
        if not future.empty:
            game = future.iloc[0]
            is_home = str(game["home_id"]) == team_id
            next_game = {
                "date": _iso(game["start_date"]),
                "opponent": str(game["away_team"] if is_home else game["home_team"]),
                "site": "vs" if is_home else "at",
                "neutral": bool(game.get("neutral_site", False)),
            }

        meta = meta_by_id.get(team_id, {})
        projection = projection_by_id.get(team_id, {})
        roster_teams.append(
            {
                "pickNumber": int(pick.pick_number),
                "round": int(getattr(pick, "round", 0) or 0),
                "managerSlot": int(pick.manager_slot),
                "manager": str(pick.manager),
                "teamId": team_id,
                "team": str(pick.team),
                "displayName": str(meta.get("display_name") or pick.team),
                "conference": str(meta.get("conference") or projection.get("conference") or "FBS"),
                "logo": f"https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png",
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "fantasyPoints": fantasy_points,
                "scoringWins": int(sum(categories.values()) - categories["non_playoff_bowl_win"]),
                "categories": categories,
                "nextGame": next_game,
                "projectedExpectedPoints": float(projection.get("expected_fantasy_points", 0.0)),
                "expectedRegularPoints": float(projection.get("expected_regular_points", 0.0)),
                "expectedConferencePoints": float(projection.get("expected_conference_title_points", 0.0)),
                "expectedPlayoffPoints": float(projection.get("expected_playoff_points", 0.0)),
                "projectedRank": int(projection.get("overall_rank", 0) or 0),
                "playoffProbability": float(projection.get("playoff_probability", 0.0)),
            }
        )

    managers: list[dict[str, Any]] = []
    for slot, name in enumerate(manager_names, start=1):
        roster = [team for team in roster_teams if team["managerSlot"] == slot]
        managers.append(
            {
                "slot": slot,
                "name": name,
                "fantasyPoints": sum(team["fantasyPoints"] for team in roster),
                "wins": sum(team["wins"] for team in roster),
                "losses": sum(team["losses"] for team in roster),
                "ties": sum(team["ties"] for team in roster),
                "projectedExpectedPoints": sum(team["projectedExpectedPoints"] for team in roster),
                "rosterCount": len(roster),
                "teams": roster,
            }
        )
    managers.sort(
        key=lambda row: (-row["fantasyPoints"], -row["wins"], -row["projectedExpectedPoints"], row["slot"])
    )
    previous_points: float | None = None
    previous_rank = 0
    for position, manager in enumerate(managers, start=1):
        if previous_points is None or manager["fantasyPoints"] != previous_points:
            previous_rank = position
            previous_points = manager["fantasyPoints"]
        manager["rank"] = previous_rank

    pick_by_number = {
        int(team["pickNumber"]): team for team in roster_teams
    }
    total_picks = len(manager_names) * teams_per_manager
    draft_board = []
    for pick_number in range(1, total_picks + 1):
        slot = _manager_for_pick(pick_number, len(manager_names))
        team = pick_by_number.get(pick_number)
        draft_board.append(
            {
                "pickNumber": pick_number,
                "round": (pick_number - 1) // len(manager_names) + 1,
                "managerSlot": slot,
                "manager": manager_names[slot - 1],
                "teamId": team["teamId"] if team else None,
                "team": team["team"] if team else None,
            }
        )

    return {
        "season": season,
        "generatedAt": now_timestamp.isoformat(),
        "scoring": weights,
        "teamsPerManager": teams_per_manager,
        "recordedPicks": len(roster_teams),
        "totalPicks": total_picks,
        "managers": managers,
        "draftBoard": draft_board,
        "recentResults": sorted(recent_results, key=lambda row: row["date"], reverse=True)[:24],
    }


def write_scoreboard_data(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    path.write_text(f"window.CFB_SCOREBOARD_DATA = {encoded};\n", encoding="utf-8")
    return path
