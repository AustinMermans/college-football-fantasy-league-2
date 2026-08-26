from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_SCORING = {
    "regular_season_win": 1.0,
    "conference_championship_win": 1.0,
    "playoff_win": 1.0,
    "non_playoff_bowl_win": 0.0,
}


def game_category(game: object) -> str:
    text = f"{getattr(game, 'event_name', '')} {getattr(game, 'notes', '')}".casefold()
    if "college football playoff" in text or "cfp " in text:
        return "playoff_win"
    if "championship" in text and "bowl" not in text:
        return "conference_championship_win"
    if getattr(game, "season_type", "regular") == "regular":
        return "regular_season_win"
    return "non_playoff_bowl_win"


def score_completed_games(
    games: pd.DataFrame,
    teams: pd.DataFrame,
    draft: pd.DataFrame,
    scoring: dict[str, float] | None = None,
) -> pd.DataFrame:
    required = {"manager", "team"}
    missing = required.difference(draft.columns)
    if missing:
        raise ValueError(f"draft file missing columns: {sorted(missing)}")
    drafted = draft["team"]
    if drafted.isna().any() or drafted.astype(str).str.strip().eq("").any():
        raise ValueError("draft file contains blank teams")
    if drafted.astype(str).duplicated().any():
        raise ValueError("draft file contains duplicate teams")
    unknown = set(drafted.astype(str)).difference(set(teams["team"].astype(str)))
    if unknown:
        raise ValueError(f"draft file contains unknown teams: {sorted(unknown)}")
    team_name = dict(zip(teams["team_id"].astype(str), teams["team"]))
    weights = DEFAULT_SCORING.copy()
    if scoring:
        weights.update(scoring)
    points: dict[str, float] = {name: 0.0 for name in teams["team"]}
    completed = games[
        games["completed"]
        & games["home_points"].notna()
        & games["away_points"].notna()
    ]
    for game in completed.itertuples(index=False):
        if float(game.home_points) > float(game.away_points):
            winner_id = str(game.home_id)
        elif float(game.away_points) > float(game.home_points):
            winner_id = str(game.away_id)
        else:
            continue
        winner = team_name.get(winner_id)
        if winner is not None:
            category = game_category(game)
            points[winner] = points.get(winner, 0.0) + float(weights[category])
    scored = draft.copy()
    scored["points"] = scored["team"].map(points).fillna(0.0)
    standings = (
        scored.groupby("manager", as_index=False)
        .agg(points=("points", "sum"), teams=("team", lambda values: ", ".join(values)))
        .sort_values(["points", "manager"], ascending=[False, True])
        .reset_index(drop=True)
    )
    standings.insert(0, "rank", standings["points"].rank(method="min", ascending=False).astype(int))
    return standings


def write_draft_template(path: Path, board: pd.DataFrame) -> Path:
    template = board[["manager_slot", "team"]].rename(columns={"manager_slot": "manager"})
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    return path
