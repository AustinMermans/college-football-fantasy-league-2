from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from cfb_fantasy.draft import make_draft_board, recommendations_for_slot
from cfb_fantasy.server import DraftStateStore, load_web_metadata
from cfb_fantasy.tracker import write_draft_template
from cfb_fantasy.web import write_draft_site_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def import_roster_grid(source_path: Path) -> dict:
    with (PROJECT_ROOT / "config" / "league.toml").open("rb") as handle:
        league = tomllib.load(handle)["league"]
    season = int(league["season"])
    managers = int(league["managers"])
    rounds = int(league["teams_per_manager"])
    draft_slot = int(league["draft_slot"])
    manager_names = [str(name) for name in league["manager_names"]]

    projections = pd.read_csv(
        PROJECT_ROOT / "results" / f"team_projections_{season}.csv"
    )
    with np.load(
        PROJECT_ROOT / "results" / f"simulation_samples_{season}.npz",
        allow_pickle=False,
    ) as samples:
        write_draft_site_data(
            PROJECT_ROOT / "web" / "draft-data.js",
            projections,
            samples["fantasy_points"],
            samples["team_ids"],
            season=season,
            managers=managers,
            teams_per_manager=rounds,
            draft_slot=draft_slot,
            manager_names=manager_names,
        )

    board, grid = make_draft_board(projections, managers, rounds)
    board.to_csv(PROJECT_ROOT / "results" / f"draft_board_{season}.csv", index=False)
    grid.to_csv(PROJECT_ROOT / "results" / f"snake_grid_{season}.csv", index=False)
    recommendations_for_slot(board, managers, rounds, draft_slot).to_csv(
        PROJECT_ROOT / "results" / f"slot_{draft_slot}_targets_{season}.csv",
        index=False,
    )
    write_draft_template(PROJECT_ROOT / "results" / "draft_template.csv", board)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    roster_rows = source["managers"]
    if len(roster_rows) != managers:
        raise ValueError("roster import must contain every manager")
    for offset, row in enumerate(roster_rows):
        if int(row["slot"]) != offset + 1 or row["manager"] != manager_names[offset]:
            raise ValueError(f"manager mismatch at slot {offset + 1}")
        if len(row["teams"]) > rounds:
            raise ValueError(f"too many teams for {row['manager']}")

    metadata = load_web_metadata(PROJECT_ROOT / "web" / "draft-data.js")
    team_index = {team["name"]: index for index, team in enumerate(metadata["teams"])}
    picks: list[int] = []
    gap_found = False
    for round_index in range(rounds):
        slot_indexes = (
            range(managers) if round_index % 2 == 0 else range(managers - 1, -1, -1)
        )
        for slot_index in slot_indexes:
            teams = roster_rows[slot_index]["teams"]
            if round_index >= len(teams):
                gap_found = True
                continue
            if gap_found:
                raise ValueError("roster grid contains a pick after an unfilled snake slot")
            name = str(teams[round_index])
            if name not in team_index:
                raise ValueError(f"unknown team in roster import: {name}")
            index = team_index[name]
            if index in picks:
                raise ValueError(f"team appears more than once: {name}")
            picks.append(index)

    store = DraftStateStore(PROJECT_ROOT / "results", metadata, season)
    current = store.load()
    state = {
        **store.default_state(),
        "draftSlot": draft_slot,
        "picks": picks,
        "mode": current["state"].get("mode", "win"),
        "detailView": "log",
    }
    return store.write(
        state,
        action="roster-grid-recovery",
        expected_revision=int(current["revision"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a named snake-draft roster grid")
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    payload = import_roster_grid(args.source)
    print(f"Imported {len(payload['pickLog'])} picks; next pick {len(payload['pickLog']) + 1}")


if __name__ == "__main__":
    main()
