from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .features import TeamState, matchup_features
from .model import FittedGameModel


@dataclass(frozen=True)
class Scoring:
    regular: float = 1.0
    conference_championship: float = 1.0
    playoff: float = 1.0


@dataclass(frozen=True)
class SeasonSimulation:
    projections: pd.DataFrame
    fantasy_points: np.ndarray
    team_ids: list[str]


def _pairwise_probabilities(
    team_ids: list[str],
    states: dict[str, TeamState],
    model: FittedGameModel,
    fpi_by_id: dict[str, float],
    *,
    fpi_weight: float,
    fpi_logistic_scale: float,
    fpi_home_advantage: float,
    neutral: bool,
) -> np.ndarray:
    if fpi_logistic_scale <= 0.0:
        raise ValueError("fpi_logistic_scale must be positive")
    rows: list[dict[str, float]] = []
    for home_id in team_ids:
        for away_id in team_ids:
            rows.append(
                matchup_features(
                    states.get(home_id, TeamState()),
                    states.get(away_id, TeamState()),
                    neutral=neutral,
                )
            )
    score_probabilities = model.predict_home(pd.DataFrame(rows))
    fpi_probabilities = []
    for home_id in team_ids:
        for away_id in team_ids:
            point_edge = fpi_by_id[home_id] - fpi_by_id[away_id]
            if not neutral:
                point_edge += fpi_home_advantage
            fpi_probabilities.append(
                1.0 / (1.0 + np.exp(-point_edge / fpi_logistic_scale))
            )
    probabilities = (
        (1.0 - fpi_weight) * score_probabilities
        + fpi_weight * np.asarray(fpi_probabilities)
    )
    matrix = probabilities.reshape((len(team_ids), len(team_ids)))
    if neutral:
        matrix = (matrix + 1.0 - matrix.T) / 2.0
        np.fill_diagonal(matrix, 0.5)
    return matrix


def _play_game(
    left: int,
    right: int,
    pairwise: np.ndarray,
    rng: np.random.Generator,
    strength_shocks: np.ndarray | None = None,
    season_strength_correlation: float = 0.0,
) -> int:
    probability = float(pairwise[left, right])
    if strength_shocks is None or season_strength_correlation == 0.0:
        return left if rng.random() < probability else right
    team_component = np.sqrt(season_strength_correlation / 2.0) * (
        strength_shocks[left] - strength_shocks[right]
    )
    performance = team_component + np.sqrt(
        1.0 - season_strength_correlation
    ) * rng.normal()
    return left if performance > ndtri(1.0 - probability) else right


def _simulate_regular_outcomes(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_probability: np.ndarray,
    *,
    simulations: int,
    n_teams: int,
    rng: np.random.Generator,
    season_strength_correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= season_strength_correlation < 1.0:
        raise ValueError("season_strength_correlation must be between zero and one")
    strength_shocks = rng.normal(size=(simulations, n_teams))
    performance = np.sqrt(1.0 - season_strength_correlation) * rng.normal(
        size=(simulations, len(home_probability))
    )
    for game_number, (home, away) in enumerate(zip(home_idx, away_idx)):
        if home >= 0 and away >= 0:
            performance[:, game_number] += np.sqrt(
                season_strength_correlation / 2.0
            ) * (strength_shocks[:, home] - strength_shocks[:, away])
        elif home >= 0:
            performance[:, game_number] += np.sqrt(
                season_strength_correlation
            ) * strength_shocks[:, home]
        elif away >= 0:
            performance[:, game_number] -= np.sqrt(
                season_strength_correlation
            ) * strength_shocks[:, away]
    thresholds = ndtri(1.0 - np.clip(home_probability, 1e-6, 1.0 - 1e-6))
    return performance > thresholds, strength_shocks


def _simulate_playoff(
    seeds: list[int],
    neutral_pairwise: np.ndarray,
    home_pairwise: np.ndarray,
    rng: np.random.Generator,
    strength_shocks: np.ndarray,
    season_strength_correlation: float,
) -> tuple[np.ndarray, int]:
    wins = np.zeros(neutral_pairwise.shape[0], dtype=np.int16)

    def game(left: int, right: int, probabilities: np.ndarray) -> int:
        winner = _play_game(
            left,
            right,
            probabilities,
            rng,
            strength_shocks,
            season_strength_correlation,
        )
        wins[winner] += 1
        return winner

    first_5 = game(seeds[4], seeds[11], home_pairwise)
    first_6 = game(seeds[5], seeds[10], home_pairwise)
    first_7 = game(seeds[6], seeds[9], home_pairwise)
    first_8 = game(seeds[7], seeds[8], home_pairwise)
    quarter_1 = game(seeds[0], first_8, neutral_pairwise)
    quarter_4 = game(seeds[3], first_5, neutral_pairwise)
    quarter_2 = game(seeds[1], first_7, neutral_pairwise)
    quarter_3 = game(seeds[2], first_6, neutral_pairwise)
    semi_a = game(quarter_1, quarter_4, neutral_pairwise)
    semi_b = game(quarter_2, quarter_3, neutral_pairwise)
    champion = game(semi_a, semi_b, neutral_pairwise)
    return wins, champion


def simulate_season(
    teams: pd.DataFrame,
    games: pd.DataFrame,
    states: dict[str, TeamState],
    model: FittedGameModel,
    *,
    simulations: int,
    seed: int,
    scoring: Scoring = Scoring(),
    fpi_weight: float = 0.55,
    fpi_logistic_scale: float = 6.5,
    fpi_home_advantage: float = 2.5,
    season_strength_correlation: float = 0.10,
) -> SeasonSimulation:
    rng = np.random.default_rng(seed)
    team_ids = teams["team_id"].astype(str).tolist()
    names = dict(zip(teams["team_id"].astype(str), teams["team"]))
    conferences = dict(zip(teams["team_id"].astype(str), teams["conference"]))
    index = {team_id: position for position, team_id in enumerate(team_ids)}
    n_teams = len(team_ids)
    n_games = len(games)
    home_idx = np.array([index.get(str(value), -1) for value in games["home_id"]], dtype=int)
    away_idx = np.array([index.get(str(value), -1) for value in games["away_id"]], dtype=int)
    home_probability = games["home_win_probability"].to_numpy(float)
    outcomes, strength_shocks = _simulate_regular_outcomes(
        home_idx,
        away_idx,
        home_probability,
        simulations=simulations,
        n_teams=n_teams,
        rng=rng,
        season_strength_correlation=season_strength_correlation,
    )

    regular_wins = np.zeros((simulations, n_teams), dtype=np.int16)
    conference_wins = np.zeros((simulations, n_teams), dtype=np.int16)
    for game_number in range(n_games):
        home, away = home_idx[game_number], away_idx[game_number]
        home_won = outcomes[:, game_number]
        if home >= 0:
            regular_wins[:, home] += home_won
        if away >= 0:
            regular_wins[:, away] += ~home_won
        if bool(games.iloc[game_number]["conference_game"]) and home >= 0 and away >= 0:
            conference_wins[:, home] += home_won
            conference_wins[:, away] += ~home_won

    score_ratings = np.array(
        [states.get(team_id, TeamState()).rating for team_id in team_ids]
    )
    fpi_by_id = dict(zip(teams["team_id"].astype(str), teams["fpi"].astype(float)))
    fpi_values = np.array([fpi_by_id[team_id] for team_id in team_ids])
    fpi_elo = 1500.0 + 10.0 * fpi_values
    ratings = (1.0 - fpi_weight) * score_ratings + fpi_weight * fpi_elo
    opponent_rating_sum = np.zeros(n_teams, dtype=float)
    opponent_count = np.zeros(n_teams, dtype=int)
    for home, away in zip(home_idx, away_idx):
        home_rating = ratings[home] if home >= 0 else 1450.0
        away_rating = ratings[away] if away >= 0 else 1450.0
        if home >= 0:
            opponent_rating_sum[home] += away_rating
            opponent_count[home] += 1
        if away >= 0:
            opponent_rating_sum[away] += home_rating
            opponent_count[away] += 1
    schedule_strength = np.divide(
        opponent_rating_sum,
        opponent_count,
        out=np.full(n_teams, 1500.0),
        where=opponent_count > 0,
    )

    neutral_pairwise = _pairwise_probabilities(
        team_ids,
        states,
        model,
        fpi_by_id,
        fpi_weight=fpi_weight,
        fpi_logistic_scale=fpi_logistic_scale,
        fpi_home_advantage=fpi_home_advantage,
        neutral=True,
    )
    home_pairwise = _pairwise_probabilities(
        team_ids,
        states,
        model,
        fpi_by_id,
        fpi_weight=fpi_weight,
        fpi_logistic_scale=fpi_logistic_scale,
        fpi_home_advantage=fpi_home_advantage,
        neutral=False,
    )
    title_appearances = np.zeros((simulations, n_teams), dtype=bool)
    title_wins = np.zeros((simulations, n_teams), dtype=np.int8)
    conference_champions = np.full((simulations, 12), -1, dtype=int)
    conference_names = [
        name
        for name in sorted(set(conferences.values()))
        if "indep" not in name.lower()
        and sum(value == name for value in conferences.values()) >= 2
    ]
    conference_champions = conference_champions[:, : len(conference_names)]
    conference_number_by_name = {
        name: number for number, name in enumerate(conference_names)
    }
    conference_games_scheduled = np.zeros(n_teams, dtype=int)
    for game_number, (home, away) in enumerate(zip(home_idx, away_idx)):
        if bool(games.iloc[game_number]["conference_game"]):
            if home >= 0:
                conference_games_scheduled[home] += 1
            if away >= 0:
                conference_games_scheduled[away] += 1
    for conference_number, conference in enumerate(conference_names):
        members = np.array(
            [index[team_id] for team_id in team_ids if conferences[team_id] == conference],
            dtype=int,
        )
        strength_tiebreak = (ratings[members] - ratings[members].min()) / 100000.0
        conference_win_percentage = np.divide(
            conference_wins[:, members],
            conference_games_scheduled[members],
            out=np.zeros_like(conference_wins[:, members], dtype=float),
            where=conference_games_scheduled[members] > 0,
        )
        standings_score = (
            conference_win_percentage * 100.0
            + regular_wins[:, members]
            + strength_tiebreak
        )
        top_two_local = np.argpartition(standings_score, -2, axis=1)[:, -2:]
        for simulation in range(simulations):
            first_local, second_local = top_two_local[simulation]
            if standings_score[simulation, first_local] < standings_score[
                simulation, second_local
            ]:
                first_local, second_local = second_local, first_local
            left, right = int(members[first_local]), int(members[second_local])
            title_appearances[simulation, left] = True
            title_appearances[simulation, right] = True
            title_probabilities = (
                home_pairwise if conference == "Pac-12" else neutral_pairwise
            )
            winner = _play_game(
                left,
                right,
                title_probabilities,
                rng,
                strength_shocks[simulation],
                season_strength_correlation,
            )
            title_wins[simulation, winner] = 1
            conference_champions[simulation, conference_number] = winner

    wins_after_titles = regular_wins + title_wins
    playoff_appearances = np.zeros((simulations, n_teams), dtype=bool)
    playoff_wins = np.zeros((simulations, n_teams), dtype=np.int8)
    national_champions = np.zeros((simulations, n_teams), dtype=bool)
    notre_dame = next(
        (index[team_id] for team_id in team_ids if names[team_id] == "Notre Dame"),
        None,
    )
    for simulation in range(simulations):
        opponent_win_total = np.zeros(n_teams, dtype=float)
        opponent_games = np.zeros(n_teams, dtype=int)
        for game_number, (home, away) in enumerate(zip(home_idx, away_idx)):
            if home >= 0 and away >= 0:
                opponent_win_total[home] += regular_wins[simulation, away]
                opponent_win_total[away] += regular_wins[simulation, home]
                opponent_games[home] += 1
                opponent_games[away] += 1
        simulated_sos = np.divide(
            opponent_win_total,
            opponent_games,
            out=np.zeros(n_teams),
            where=opponent_games > 0,
        )
        committee_score = (
            wins_after_titles[simulation].astype(float)
            + 0.40 * ((ratings - 1500.0) / 100.0)
            + 0.25 * simulated_sos
            + rng.normal(0.0, 0.03, n_teams)
        )
        power_conferences = {"ACC", "Big Ten", "Big 12", "SEC"}
        power_auto = [
            int(
                conference_champions[
                    simulation, conference_number_by_name[conference]
                ]
            )
            for conference in power_conferences
        ]
        other_champions = np.array(
            [
                conference_champions[simulation, number]
                for number, conference in enumerate(conference_names)
                if conference not in power_conferences
            ],
            dtype=int,
        )
        highest_other = int(other_champions[np.argmax(committee_score[other_champions])])
        auto = np.array(power_auto + [highest_other], dtype=int)
        auto_set = set(int(value) for value in auto)
        at_large = [
            int(value)
            for value in np.argsort(committee_score)[::-1]
            if int(value) not in auto_set
        ][:7]
        overall_ranking = np.argsort(committee_score)[::-1]
        if (
            notre_dame is not None
            and int(np.where(overall_ranking == notre_dame)[0][0]) < 12
            and notre_dame not in auto_set
            and notre_dame not in at_large
        ):
            at_large[-1] = notre_dame
        field = list(auto_set) + at_large
        seeds = sorted(field, key=lambda value: committee_score[value], reverse=True)
        playoff_appearances[simulation, seeds] = True
        wins, champion = _simulate_playoff(
            seeds,
            neutral_pairwise,
            home_pairwise,
            rng,
            strength_shocks[simulation],
            season_strength_correlation,
        )
        playoff_wins[simulation] = wins
        national_champions[simulation, champion] = True

    fantasy_points = (
        scoring.regular * regular_wins
        + scoring.conference_championship * title_wins
        + scoring.playoff * playoff_wins
    )
    exact_regular_ev = np.zeros(n_teams, dtype=float)
    for game_number, (home, away) in enumerate(zip(home_idx, away_idx)):
        if home >= 0:
            exact_regular_ev[home] += home_probability[game_number]
        if away >= 0:
            exact_regular_ev[away] += 1.0 - home_probability[game_number]

    rows = []
    for team_number, team_id in enumerate(team_ids):
        expected_regular_points = scoring.regular * exact_regular_ev[team_number]
        expected_conference_title_points = (
            scoring.conference_championship
            * title_wins[:, team_number].mean()
        )
        expected_playoff_points = (
            scoring.playoff * playoff_wins[:, team_number].mean()
        )
        rows.append(
            {
                "team_id": team_id,
                "team": names[team_id],
                "conference": conferences[team_id],
                "scheduled_games": int(opponent_count[team_number]),
                "score_model_rating": score_ratings[team_number],
                "fpi": fpi_values[team_number],
                "fpi_rank": int(teams.iloc[team_number]["fpi_rank"]),
                "fpi_projected_wins": float(
                    teams.iloc[team_number]["fpi_projected_wins"]
                ),
                "consensus_rating": ratings[team_number],
                "average_opponent_rating": schedule_strength[team_number],
                "expected_regular_wins": exact_regular_ev[team_number],
                "expected_regular_points": expected_regular_points,
                "conference_title_game_probability": title_appearances[:, team_number].mean(),
                "conference_title_probability": title_wins[:, team_number].mean(),
                "expected_conference_title_points": expected_conference_title_points,
                "playoff_probability": playoff_appearances[:, team_number].mean(),
                "expected_playoff_wins": playoff_wins[:, team_number].mean(),
                "expected_playoff_points": expected_playoff_points,
                "expected_postseason_points": (
                    expected_conference_title_points + expected_playoff_points
                ),
                "expected_points_before_playoff": (
                    expected_regular_points + expected_conference_title_points
                ),
                "national_title_probability": national_champions[:, team_number].mean(),
                "expected_fantasy_points": expected_regular_points
                + expected_conference_title_points
                + expected_playoff_points,
                "median_fantasy_points": np.quantile(fantasy_points[:, team_number], 0.50),
                "p10_fantasy_points": np.quantile(fantasy_points[:, team_number], 0.10),
                "p90_fantasy_points": np.quantile(fantasy_points[:, team_number], 0.90),
            }
        )
    projections = pd.DataFrame(rows)
    projections["schedule_difficulty_rank"] = projections[
        "average_opponent_rating"
    ].rank(method="min", ascending=False).astype(int)
    projections["pre_playoff_rank"] = projections[
        "expected_points_before_playoff"
    ].rank(method="min", ascending=False).astype(int)
    projections = projections.sort_values(
        ["expected_fantasy_points", "p10_fantasy_points"], ascending=False
    ).reset_index(drop=True)
    projections.insert(0, "overall_rank", np.arange(1, len(projections) + 1))
    projections["playoff_rank_lift"] = (
        projections["pre_playoff_rank"] - projections["overall_rank"]
    )
    projections["playoff_value_share"] = np.divide(
        projections["expected_playoff_points"],
        projections["expected_fantasy_points"],
        out=np.zeros(len(projections), dtype=float),
        where=projections["expected_fantasy_points"] != 0,
    )
    return SeasonSimulation(
        projections=projections,
        fantasy_points=fantasy_points,
        team_ids=team_ids,
    )
