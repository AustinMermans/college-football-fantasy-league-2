from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import (
    walk_forward_calibration_backtest,
    walk_forward_probability_calibration_backtest,
)
from .data import (
    attach_historical_fpi,
    default_raw_dir,
    fetch_fpi,
    fetch_historical_fpi,
    fetch_historical_seasons,
    fetch_target_season,
    load_historical_games,
    write_snapshot,
)
from .draft import (
    league_win_recommendations,
    live_recommendations,
    make_draft_board,
    manager_for_pick,
    recommendations_for_slot,
    validate_live_picks,
)
from .features import (
    FEATURE_COLUMNS,
    build_preseason_features,
    preseason_states,
    schedule_feature_frame,
)
from .model import (
    attach_consensus_probabilities,
    calibrate_current_ensemble,
    calibrate_fpi_ensemble,
    candidates,
    estimate_season_strength_correlation,
    fit_selected_model,
    walk_forward_backtest,
    walk_forward_feature_study,
)
from .report import write_model_card
from .simulate import Scoring, simulate_season
from .server import serve_draft_room
from .tracker import score_completed_games, write_draft_template
from .web import write_draft_site_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def run_pipeline(args: argparse.Namespace) -> None:
    config = _config(args.config)
    league = config["league"]
    model_config = config["model"]
    scoring_config = config["scoring"]
    season = int(args.season or league["season"])
    raw_dir = default_raw_dir(PROJECT_ROOT)
    derived_dir = PROJECT_ROOT / "data_derived"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    historical_paths = fetch_historical_seasons(
        raw_dir,
        int(model_config["historical_start_season"]),
        season - 1,
        refresh=args.refresh,
    )
    historical_games = load_historical_games(historical_paths)
    historical_fpi_paths = fetch_historical_fpi(
        raw_dir,
        int(model_config["historical_start_season"]),
        season - 1,
        refresh=args.refresh,
    )
    historical_games = attach_historical_fpi(
        historical_games, historical_fpi_paths
    )
    teams, schedule = fetch_target_season(raw_dir, season, refresh=args.refresh)
    fpi = fetch_fpi(raw_dir, season, refresh=args.refresh)
    teams = teams.merge(fpi, on="team_id", how="left", validate="one_to_one")
    if teams["fpi"].isna().any():
        missing = teams.loc[teams["fpi"].isna(), "team"].tolist()
        raise ValueError(f"FPI missing for current FBS teams: {missing}")
    schedule = schedule[schedule["season_type"] == "regular"].copy()
    write_snapshot(schedule, teams, derived_dir, season)

    features, end_states = build_preseason_features(historical_games)
    features.to_csv(derived_dir / "preseason_features.csv", index=False)
    by_season, factor_study = walk_forward_backtest(
        features, int(model_config["first_backtest_season"])
    )
    by_season.to_csv(results_dir / "backtest_by_season.csv", index=False)
    factor_study.to_csv(results_dir / "factor_study.csv", index=False)
    feature_study, feature_screen = walk_forward_feature_study(
        features,
        int(model_config["first_backtest_season"]),
        random_seed=int(model_config["random_seed"]),
    )
    feature_study.to_csv(results_dir / "feature_study.csv", index=False)
    feature_screen.to_csv(results_dir / "feature_screen.csv", index=False)
    accepted_features = feature_study.loc[
        feature_study["accepted"], "feature"
    ].astype(str).tolist()
    if accepted_features != FEATURE_COLUMNS:
        raise ValueError(
            "production features do not match the random-control feature gate: "
            f"expected {accepted_features}, configured {FEATURE_COLUMNS}"
        )
    selected_name = str(factor_study.iloc[0]["model"])
    selected_candidate = next(
        candidate for candidate in candidates() if candidate.name == selected_name
    )
    (
        score_calibration_by_season,
        score_calibration_study,
        score_calibration_oof,
        score_calibrator,
    ) = walk_forward_calibration_backtest(
        features,
        selected_candidate.columns,
        selected_candidate.factory,
        int(model_config["first_backtest_season"]),
    )
    score_calibration_by_season.to_csv(
        results_dir / "calibration_backtest_by_season.csv", index=False
    )
    score_calibration_study.to_csv(
        results_dir / "calibration_study.csv", index=False
    )
    score_calibration_oof.to_csv(
        results_dir / "calibration_oof_predictions.csv", index=False
    )
    (results_dir / "score_calibration.json").write_text(
        json.dumps(score_calibrator.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fpi_calibration_by_season, fpi_calibration_study = (
        walk_forward_probability_calibration_backtest(
            features,
            "fpi_home_probability",
            int(model_config["first_backtest_season"]),
        )
    )
    fpi_calibration_by_season.to_csv(
        results_dir / "fpi_calibration_backtest_by_season.csv", index=False
    )
    fpi_calibration_study.to_csv(
        results_dir / "fpi_calibration_study.csv", index=False
    )
    ensemble_backtest, ensemble_oof, ensemble_calibration = calibrate_fpi_ensemble(
        features,
        factor_study,
        int(model_config["first_backtest_season"]),
        score_calibration_oof,
    )
    ensemble_backtest.to_csv(results_dir / "ensemble_backtest.csv", index=False)
    ensemble_oof.to_csv(results_dir / "ensemble_oof_predictions.csv", index=False)
    ensemble_calibration["season_strength_correlation"] = (
        estimate_season_strength_correlation(
            ensemble_oof, int(model_config["first_backtest_season"])
        )
    )
    fitted = fit_selected_model(features, factor_study, score_calibrator)
    states = preseason_states(end_states)
    game_features = schedule_feature_frame(schedule, states)
    current_calibration = calibrate_current_ensemble(
        game_features,
        fitted,
        teams,
        fpi_home_advantage=float(model_config["fpi_home_advantage"]),
    )
    ensemble_calibration.update(current_calibration)
    (results_dir / "ensemble_calibration.json").write_text(
        json.dumps(ensemble_calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fpi_weight = float(ensemble_calibration["fpi_weight"])
    fpi_logistic_scale = float(ensemble_calibration["fpi_logistic_scale"])

    game_probabilities = attach_consensus_probabilities(
        game_features,
        fitted,
        teams,
        fpi_weight=fpi_weight,
        fpi_logistic_scale=fpi_logistic_scale,
        fpi_home_advantage=float(model_config["fpi_home_advantage"]),
    )
    game_probabilities.to_csv(
        results_dir / f"game_probabilities_{season}.csv", index=False
    )
    simulation = simulate_season(
        teams,
        game_probabilities,
        states,
        fitted,
        simulations=int(args.simulations or model_config["simulations"]),
        seed=int(model_config["random_seed"]),
        scoring=Scoring(
            regular=float(scoring_config["regular_season_win"]),
            conference_championship=float(
                scoring_config["conference_championship_win"]
            ),
            playoff=float(scoring_config["playoff_win"]),
        ),
        fpi_weight=fpi_weight,
        fpi_logistic_scale=fpi_logistic_scale,
        fpi_home_advantage=float(model_config["fpi_home_advantage"]),
        season_strength_correlation=float(
            ensemble_calibration["season_strength_correlation"]
        ),
    )
    projections = simulation.projections
    np.savez_compressed(
        results_dir / f"simulation_samples_{season}.npz",
        fantasy_points=simulation.fantasy_points,
        team_ids=np.asarray(simulation.team_ids),
    )
    write_draft_site_data(
        PROJECT_ROOT / "web" / "draft-data.js",
        projections,
        simulation.fantasy_points,
        np.asarray(simulation.team_ids),
        season=season,
        managers=int(league["managers"]),
        teams_per_manager=int(league["teams_per_manager"]),
        draft_slot=int(league["draft_slot"]),
        manager_names=[str(name) for name in league.get("manager_names", [])] or None,
    )
    projections.to_csv(results_dir / f"team_projections_{season}.csv", index=False)
    board, grid = make_draft_board(
        projections,
        int(league["managers"]),
        int(league["teams_per_manager"]),
    )
    board.to_csv(results_dir / f"draft_board_{season}.csv", index=False)
    grid.to_csv(results_dir / f"snake_grid_{season}.csv", index=False)
    slot = int(league["draft_slot"])
    recommendations_for_slot(
        board, int(league["managers"]), int(league["teams_per_manager"]), slot
    ).to_csv(results_dir / f"slot_{slot}_targets_{season}.csv", index=False)
    write_draft_template(results_dir / "draft_template.csv", board)
    live_path = results_dir / "live_picks.csv"
    if not live_path.exists():
        pd.DataFrame(columns=["pick_number", "manager_slot", "team"]).to_csv(
            live_path, index=False
        )
    write_model_card(
        results_dir / "MODEL_CARD.md",
        projections,
        factor_study,
        feature_study,
        score_calibration_study,
        score_calibrator.summary(),
        fpi_calibration_study,
        ensemble_backtest,
        ensemble_calibration,
        season=season,
        simulations=int(args.simulations or model_config["simulations"]),
        schedule_games=len(schedule),
    )
    print(f"Selected model: {fitted.name}")
    print(f"Selected score calibration: {score_calibrator.method}")
    print(
        f"Calibrated FPI weight: {fpi_weight:.0%}; "
        f"logistic scale: {fpi_logistic_scale:.2f}"
    )
    print(f"Historical games: {len(historical_games):,}")
    print(f"FBS teams: {len(teams)}; known regular-season games: {len(schedule)}")
    print(projections.head(20).to_string(index=False))
    print(f"Outputs: {results_dir}")


def score_draft(args: argparse.Namespace) -> None:
    config = _config(args.config)
    season = int(args.season or config["league"]["season"])
    teams, games = fetch_target_season(
        default_raw_dir(PROJECT_ROOT), season, refresh=args.refresh
    )
    draft = pd.read_csv(args.draft)
    standings = score_completed_games(games, teams, draft, config["scoring"])
    destination = PROJECT_ROOT / "results" / f"standings_{season}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    standings.to_csv(destination, index=False)
    print(standings.to_string(index=False))
    print(f"Wrote {destination}")


def _live_inputs(args: argparse.Namespace) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    config = _config(args.config)
    season = int(config["league"]["season"])
    projection_path = PROJECT_ROOT / "results" / f"team_projections_{season}.csv"
    if not projection_path.exists():
        raise FileNotFoundError(f"run the model first: missing {projection_path}")
    projections = pd.read_csv(projection_path)
    if args.picks.exists():
        picks = pd.read_csv(args.picks)
    else:
        picks = pd.DataFrame(columns=["pick_number", "manager_slot", "team"])
    return config, projections, picks


def draft_status(args: argparse.Namespace) -> None:
    config, projections, picks = _live_inputs(args)
    league = config["league"]
    status, available = live_recommendations(
        projections,
        picks,
        managers=int(league["managers"]),
        teams_per_manager=int(league["teams_per_manager"]),
        draft_slot=int(league["draft_slot"]),
        count=args.count,
    )
    print(
        f"Pick {status['next_pick']} | current slot {status['current_manager_slot']} | "
        f"our turn: {status['our_turn']} | next our pick: {status['next_our_pick']}"
    )
    league_columns: list[str] = []
    if status["our_turn"]:
        season = int(config["league"]["season"])
        sample_path = PROJECT_ROOT / "results" / f"simulation_samples_{season}.npz"
        if sample_path.exists():
            with np.load(sample_path, allow_pickle=False) as samples:
                league_values = league_win_recommendations(
                    projections,
                    picks,
                    samples["fantasy_points"],
                    samples["team_ids"],
                    managers=int(league["managers"]),
                    teams_per_manager=int(league["teams_per_manager"]),
                    draft_slot=int(league["draft_slot"]),
                    count=args.count,
                )
            available = available.merge(league_values, on="team", how="left")
            available.loc[
                available["league_value_rank"].eq(1), "draft_action"
            ] = "DRAFT THIS"
            available = available.sort_values("league_value_rank")
            league_columns = [
                "league_value_rank",
                "league_win_probability",
                "expected_finish",
                "expected_margin_to_best_opponent",
            ]
    columns = [
        "available_rank",
        "draft_action",
        "team",
        "conference",
        "expected_fantasy_points",
        "expected_regular_points",
        "expected_conference_title_points",
        "expected_playoff_points",
        "playoff_rank_lift",
        *league_columns,
        "p10_fantasy_points",
        "p90_fantasy_points",
        "expected_regular_wins",
        "playoff_probability",
        "schedule_difficulty_rank",
    ]
    print(available[columns].to_string(index=False))
    destination = PROJECT_ROOT / "results" / "live_recommendations.csv"
    available.to_csv(destination, index=False)


def _resolve_team(value: str, projections: pd.DataFrame) -> str:
    folded = value.strip().casefold()
    exact = projections[projections["team"].str.casefold().eq(folded)]["team"].tolist()
    if exact:
        return str(exact[0])
    partial = projections[
        projections["team"].str.casefold().str.contains(folded, regex=False)
    ]["team"].tolist()
    if len(partial) == 1:
        return str(partial[0])
    raise ValueError(f"team must match exactly or uniquely; candidates: {partial[:10]}")


def record_pick(args: argparse.Namespace) -> None:
    config, projections, picks = _live_inputs(args)
    managers = int(config["league"]["managers"])
    teams_per_manager = int(config["league"]["teams_per_manager"])
    validate_live_picks(
        projections,
        picks,
        managers=managers,
        teams_per_manager=teams_per_manager,
    )
    if len(picks) >= managers * teams_per_manager:
        raise ValueError("the configured draft is already complete")
    team = _resolve_team(args.team, projections)
    if team in set(picks["team"].astype(str)):
        raise ValueError(f"{team} is already drafted")
    pick_number = len(picks) + 1
    row = pd.DataFrame(
        [
            {
                "pick_number": pick_number,
                "manager_slot": manager_for_pick(pick_number, managers),
                "team": team,
            }
        ]
    )
    picks = pd.concat([picks, row], ignore_index=True)
    args.picks.parent.mkdir(parents=True, exist_ok=True)
    picks.to_csv(args.picks, index=False)
    print(f"Recorded pick {pick_number}: {team}")
    draft_status(args)


def undo_pick(args: argparse.Namespace) -> None:
    _, _, picks = _live_inputs(args)
    if picks.empty:
        raise ValueError("there are no picks to undo")
    removed = picks.iloc[-1]["team"]
    picks.iloc[:-1].to_csv(args.picks, index=False)
    print(f"Removed last pick: {removed}")
    draft_status(args)


def serve_site(args: argparse.Namespace) -> None:
    config = _config(args.config)
    league = config["league"]
    serve_draft_room(
        PROJECT_ROOT,
        host=args.host,
        port=args.port,
        season=int(league["season"]),
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="College football fantasy model")
    command.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "league.toml"
    )
    subcommands = command.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="refresh data, backtest, and project season")
    run.add_argument("--season", type=int)
    run.add_argument("--simulations", type=int)
    run.add_argument("--refresh", action="store_true")
    run.set_defaults(func=run_pipeline)
    score = subcommands.add_parser("score", help="score completed games for a draft")
    score.add_argument("--draft", type=Path, required=True)
    score.add_argument("--season", type=int)
    score.add_argument("--refresh", action="store_true")
    score.set_defaults(func=score_draft)
    serve = subcommands.add_parser(
        "serve", help="run the persistent local browser draft room"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=serve_site)
    for name, help_text, handler in [
        ("draft-status", "show the best teams still available", draft_status),
        ("record-pick", "record one pick and refresh recommendations", record_pick),
        ("undo-pick", "remove the most recent pick", undo_pick),
    ]:
        live = subcommands.add_parser(name, help=help_text)
        live.add_argument(
            "--picks", type=Path, default=PROJECT_ROOT / "results" / "live_picks.csv"
        )
        live.add_argument("--count", type=int, default=15)
        if name == "record-pick":
            live.add_argument("--team", required=True)
        live.set_defaults(func=handler)
    return command


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
