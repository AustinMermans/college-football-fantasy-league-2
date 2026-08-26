from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "long_margin_diff",
    "talent_composite_diff",
    "net_adj_epa_diff",
    "fbs_status_diff",
    "blue_chip_ratio_diff",
    "home_field",
]
FEATURE_CANDIDATES = [
    "rating_diff",
    "margin_form_diff",
    "win_form_diff",
    "long_margin_diff",
    "long_win_diff",
    "fbs_status_diff",
    "history_depth_diff",
    "rating_momentum_diff",
    "home_field",
    "talent_composite_diff",
    "blue_chip_ratio_diff",
    "net_adj_epa_diff",
]
ELO_COLUMNS = ["rating_diff", "home_field"]


@dataclass
class TeamState:
    rating: float = 1500.0
    margin_form: float = 0.0
    win_form: float = 0.5
    long_margin_form: float = 0.0
    long_win_form: float = 0.5
    games: int = 0
    total_games: int = 0
    last_played: pd.Timestamp | None = None
    preseason_rating: float = 1500.0
    rating_momentum: float = 0.0


def regress_offseason(state: TeamState) -> TeamState:
    rating = 1500.0 + 0.68 * (state.rating - 1500.0)
    return TeamState(
        rating=rating,
        margin_form=0.55 * state.margin_form,
        win_form=0.5 + 0.55 * (state.win_form - 0.5),
        long_margin_form=0.75 * state.long_margin_form,
        long_win_form=0.5 + 0.75 * (state.long_win_form - 0.5),
        games=0,
        total_games=state.total_games,
        last_played=None,
        preseason_rating=rating,
        rating_momentum=rating - state.preseason_rating,
    )


def _rest_days(state: TeamState, kickoff: pd.Timestamp) -> float:
    if state.last_played is None:
        return 14.0
    return float(np.clip((kickoff - state.last_played).days, 3, 30))


def matchup_features(
    home: TeamState,
    away: TeamState,
    *,
    kickoff: pd.Timestamp | None = None,
    neutral: bool = False,
    home_is_fbs: bool = True,
    away_is_fbs: bool = True,
) -> dict[str, float]:
    when = kickoff or pd.Timestamp("2000-01-15", tz="UTC")
    return {
        "rating_diff": (home.rating - away.rating) / 100.0,
        "margin_form_diff": (home.margin_form - away.margin_form) / 14.0,
        "win_form_diff": home.win_form - away.win_form,
        "long_margin_diff": (home.long_margin_form - away.long_margin_form) / 14.0,
        "long_win_diff": home.long_win_form - away.long_win_form,
        "fbs_status_diff": float(home_is_fbs) - float(away_is_fbs),
        "history_depth_diff": math.log1p(home.total_games)
        - math.log1p(away.total_games),
        "rating_momentum_diff": (home.rating_momentum - away.rating_momentum) / 100.0,
        "experience_diff": math.log1p(home.games) - math.log1p(away.games),
        "rest_diff": (_rest_days(home, when) - _rest_days(away, when)) / 7.0,
        "home_field": 0.0 if neutral else 1.0,
    }


def update_states(
    home: TeamState,
    away: TeamState,
    home_points: float,
    away_points: float,
    kickoff: pd.Timestamp,
    *,
    neutral: bool,
) -> None:
    home_advantage = 0.0 if neutral else 55.0
    expected = 1.0 / (1.0 + 10.0 ** (-(home.rating + home_advantage - away.rating) / 400.0))
    outcome = 1.0 if home_points > away_points else 0.0 if home_points < away_points else 0.5
    margin = float(home_points - away_points)
    multiplier = min(2.5, max(0.7, math.log1p(abs(margin)) / math.log(7.0)))
    change = 24.0 * multiplier * (outcome - expected)
    home.rating += change
    away.rating -= change
    alpha = 0.22
    capped_margin = float(np.clip(margin, -49.0, 49.0))
    home.margin_form = (1.0 - alpha) * home.margin_form + alpha * capped_margin
    away.margin_form = (1.0 - alpha) * away.margin_form - alpha * capped_margin
    home.win_form = (1.0 - alpha) * home.win_form + alpha * outcome
    away.win_form = (1.0 - alpha) * away.win_form + alpha * (1.0 - outcome)
    long_alpha = 0.08
    home.long_margin_form = (
        (1.0 - long_alpha) * home.long_margin_form + long_alpha * capped_margin
    )
    away.long_margin_form = (
        (1.0 - long_alpha) * away.long_margin_form - long_alpha * capped_margin
    )
    home.long_win_form = (
        (1.0 - long_alpha) * home.long_win_form + long_alpha * outcome
    )
    away.long_win_form = (
        (1.0 - long_alpha) * away.long_win_form + long_alpha * (1.0 - outcome)
    )
    home.games += 1
    away.games += 1
    home.total_games += 1
    away.total_games += 1
    home.last_played = kickoff
    away.last_played = kickoff


def build_pregame_features(
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, TeamState]]:
    ordered = games.sort_values(["start_date", "game_id"]).copy()
    states: dict[str, TeamState] = {}
    rows: list[dict[str, object]] = []
    current_season: int | None = None
    for game in ordered.itertuples(index=False):
        season = int(game.season)
        if current_season is not None and season != current_season:
            states = {team_id: regress_offseason(state) for team_id, state in states.items()}
        current_season = season
        home_id, away_id = str(game.home_id), str(game.away_id)
        home = states.setdefault(home_id, TeamState())
        away = states.setdefault(away_id, TeamState())
        features = matchup_features(
            home,
            away,
            kickoff=game.start_date,
            neutral=bool(game.neutral_site),
            home_is_fbs=str(getattr(game, "home_division", "fbs")).lower() == "fbs",
            away_is_fbs=str(getattr(game, "away_division", "fbs")).lower() == "fbs",
        )
        home_points, away_points = float(game.home_points), float(game.away_points)
        if home_points == away_points:
            target = np.nan
        else:
            target = float(home_points > away_points)
        rows.append(
            {
                "game_id": str(game.game_id),
                "season": season,
                "home_id": home_id,
                "away_id": away_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "target": target,
                "fpi_home_probability": getattr(
                    game, "fpi_home_probability", np.nan
                ),
                "fpi_predicted_margin": getattr(
                    game, "fpi_predicted_margin", np.nan
                ),
                **features,
            }
        )
        update_states(
            home,
            away,
            home_points,
            away_points,
            game.start_date,
            neutral=bool(game.neutral_site),
        )
    frame = pd.DataFrame(rows).dropna(subset=["target"]).reset_index(drop=True)
    frame["target"] = frame["target"].astype(int)
    return frame, states


def build_preseason_features(
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, TeamState]]:
    """Freeze every matchup at the prior season's information set."""
    ordered = games.sort_values(["season", "start_date", "game_id"]).copy()
    states: dict[str, TeamState] = {}
    rows: list[dict[str, object]] = []
    previous_season: int | None = None
    for season, season_games in ordered.groupby("season", sort=True):
        season = int(season)
        if previous_season is not None:
            states = {
                team_id: regress_offseason(state) for team_id, state in states.items()
            }
        season_ids = set(season_games["home_id"].astype(str)).union(
            season_games["away_id"].astype(str)
        )
        frozen = clone_states(states, season_ids)
        for game in season_games.itertuples(index=False):
            home_id, away_id = str(game.home_id), str(game.away_id)
            home = frozen.setdefault(home_id, TeamState())
            away = frozen.setdefault(away_id, TeamState())
            home_points, away_points = float(game.home_points), float(game.away_points)
            rows.append(
                {
                    "game_id": str(game.game_id),
                    "season": season,
                    "home_id": home_id,
                    "away_id": away_id,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "target": np.nan
                    if home_points == away_points
                    else float(home_points > away_points),
                    "fpi_home_probability": getattr(
                        game, "fpi_home_probability", np.nan
                    ),
                    "fpi_predicted_margin": getattr(
                        game, "fpi_predicted_margin", np.nan
                    ),
                    **matchup_features(
                        home,
                        away,
                        kickoff=game.start_date,
                        neutral=bool(game.neutral_site),
                        home_is_fbs=str(
                            getattr(game, "home_division", "fbs")
                        ).lower()
                        == "fbs",
                        away_is_fbs=str(
                            getattr(game, "away_division", "fbs")
                        ).lower()
                        == "fbs",
                    ),
                }
            )
        for game in season_games.itertuples(index=False):
            home_id, away_id = str(game.home_id), str(game.away_id)
            home = states.setdefault(home_id, TeamState())
            away = states.setdefault(away_id, TeamState())
            update_states(
                home,
                away,
                float(game.home_points),
                float(game.away_points),
                game.start_date,
                neutral=bool(game.neutral_site),
            )
        previous_season = season
    frame = pd.DataFrame(rows).dropna(subset=["target"]).reset_index(drop=True)
    frame["target"] = frame["target"].astype(int)
    return frame, states


def preseason_states(states: dict[str, TeamState]) -> dict[str, TeamState]:
    return {team_id: regress_offseason(replace(state)) for team_id, state in states.items()}


def schedule_feature_frame(
    schedule: pd.DataFrame, states: dict[str, TeamState]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game in schedule.itertuples(index=False):
        home = states.get(str(game.home_id), TeamState())
        away = states.get(str(game.away_id), TeamState())
        rows.append(
            {
                "game_id": str(game.game_id),
                "season": int(game.season),
                "home_id": str(game.home_id),
                "away_id": str(game.away_id),
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_conference": game.home_conference,
                "away_conference": game.away_conference,
                "conference_game": bool(game.conference_game),
                "neutral_site": bool(game.neutral_site),
                **matchup_features(
                    home,
                    away,
                    kickoff=game.start_date,
                    neutral=bool(game.neutral_site),
                    home_is_fbs=str(game.home_division).lower() == "fbs",
                    away_is_fbs=str(game.away_division).lower() == "fbs",
                ),
            }
        )
    return pd.DataFrame(rows)


def clone_states(states: dict[str, TeamState], team_ids: Iterable[str]) -> dict[str, TeamState]:
    return {str(team_id): replace(states.get(str(team_id), TeamState())) for team_id in team_ids}
