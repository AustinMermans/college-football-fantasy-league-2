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
    fetch_target_season,
    load_betting,
)
from cfb_fantasy.market import MarketConsensus, apply_market_consensus  # noqa: E402
from cfb_fantasy.scoreboard import (  # noqa: E402
    build_scoreboard,
    update_live_projections,
    write_scoreboard_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the public fantasy scoreboard")
    parser.add_argument("--refresh", action="store_true", help="download current ESPN schedules")
    args = parser.parse_args()

    with (PROJECT_ROOT / "config" / "league.toml").open("rb") as handle:
        config = tomllib.load(handle)
    league = config["league"]
    season = int(league["season"])

    if args.refresh:
        teams, games = fetch_target_season(
            PROJECT_ROOT / ".cache" / "cfb_fantasy",
            season,
            refresh=True,
        )
    else:
        teams = pd.read_csv(PROJECT_ROOT / "data_derived" / f"teams_{season}.csv")
        games = pd.read_csv(PROJECT_ROOT / "data_derived" / f"schedule_{season}.csv")

    picks = pd.read_csv(PROJECT_ROOT / "results" / "live_picks.csv")
    projections = pd.read_csv(
        PROJECT_ROOT / "results" / f"team_projections_{season}.csv"
    )
    probabilities = pd.read_csv(
        PROJECT_ROOT / "results" / f"game_probabilities_{season}.csv"
    )
    if args.refresh:
        betting_paths = fetch_betting(
            PROJECT_ROOT / ".cache" / "cfb_fantasy",
            season,
            season,
            refresh=True,
        )
        betting = load_betting(betting_paths)
        market_summary = json.loads(
            (PROJECT_ROOT / "results" / "market_model.json").read_text()
        )
        probabilities = apply_market_consensus(
            probabilities,
            betting,
            MarketConsensus.from_summary(market_summary),
        )
    projections = update_live_projections(games, probabilities, projections)
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
