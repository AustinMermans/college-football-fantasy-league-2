from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cfb_fantasy.data import (  # noqa: E402
    fetch_betting,
    fetch_fpi,
    fetch_target_season,
    load_betting,
)
from cfb_fantasy.market import (  # noqa: E402
    MarketConsensus,
    apply_market_consensus,
    update_fpi_probabilities,
)
from cfb_fantasy.scoreboard import (  # noqa: E402
    apply_completed_results,
    build_scoreboard,
    write_scoreboard_data,
)
from cfb_fantasy.simulate import Scoring, simulate_season  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the public fantasy scoreboard")
    parser.add_argument("--refresh", action="store_true", help="download current ESPN schedules")
    args = parser.parse_args()

    with (PROJECT_ROOT / "config" / "league.toml").open("rb") as handle:
        config = tomllib.load(handle)
    league = config["league"]
    season = int(league["season"])
    model_config = config["model"]
    cache_dir = PROJECT_ROOT / ".cache" / "cfb_fantasy"

    if args.refresh:
        teams, games = fetch_target_season(
            cache_dir,
            season,
            refresh=True,
        )
        current_fpi = fetch_fpi(cache_dir, season, refresh=True)
        teams = teams.merge(current_fpi, on="team_id", how="left", validate="one_to_one")
    else:
        teams = pd.read_csv(PROJECT_ROOT / "data_derived" / f"teams_{season}.csv")
        games = pd.read_csv(PROJECT_ROOT / "data_derived" / f"schedule_{season}.csv")

    picks = pd.read_csv(PROJECT_ROOT / "results" / "live_picks.csv")
    probabilities = pd.read_csv(
        PROJECT_ROOT / "results" / f"game_probabilities_{season}.csv"
    )
    market_summary = json.loads(
        (PROJECT_ROOT / "results" / "market_model.json").read_text()
    )
    if args.refresh:
        probabilities = update_fpi_probabilities(
            probabilities,
            teams,
            logistic_scale=float(market_summary["fpi_logistic_scale"]),
            home_advantage=float(market_summary["fpi_home_advantage"]),
        )
        betting_paths = fetch_betting(
            cache_dir,
            season,
            season,
            refresh=True,
        )
        betting = load_betting(betting_paths)
        probabilities = apply_market_consensus(
            probabilities,
            betting,
            MarketConsensus.from_summary(market_summary),
        )
    probabilities = apply_completed_results(games, probabilities)
    scoring = Scoring(
        regular=float(config["scoring"]["regular_season_win"]),
        conference_championship=float(
            config["scoring"]["conference_championship_win"]
        ),
        playoff=float(config["scoring"]["playoff_win"]),
    )
    simulation = simulate_season(
        teams,
        probabilities,
        {},
        None,
        simulations=int(model_config["simulations"]),
        seed=int(model_config["random_seed"]),
        scoring=scoring,
        fpi_weight=1.0,
        fpi_logistic_scale=float(market_summary["fpi_logistic_scale"]),
        fpi_home_advantage=float(market_summary["fpi_home_advantage"]),
        season_strength_correlation=float(
            market_summary["season_strength_correlation"]
        ),
    )
    projections = simulation.projections
    payload = build_scoreboard(
        games,
        teams,
        picks,
        projections,
        season=season,
        manager_names=[str(name) for name in league["manager_names"]],
        teams_per_manager=int(league["teams_per_manager"]),
        scoring=config["scoring"],
    )
    destination = write_scoreboard_data(
        PROJECT_ROOT / "web" / "scoreboard-data.js", payload
    )
    print(
        f"Wrote {destination} with {payload['recordedPicks']} picks and "
        f"{sum(len(manager['teams']) for manager in payload['managers'])} roster teams"
    )


if __name__ == "__main__":
    main()
