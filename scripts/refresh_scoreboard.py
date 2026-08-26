from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cfb_fantasy.data import fetch_target_season  # noqa: E402
from cfb_fantasy.scoreboard import build_scoreboard, write_scoreboard_data  # noqa: E402


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
