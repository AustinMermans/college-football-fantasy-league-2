from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WEB_TEAM_COLUMNS = {
    "team_id": "id",
    "team": "name",
    "conference": "conference",
    "overall_rank": "rank",
    "expected_fantasy_points": "expectedPoints",
    "expected_regular_points": "regularPoints",
    "expected_conference_title_points": "conferenceTitlePoints",
    "expected_playoff_points": "playoffPoints",
    "playoff_probability": "playoffProbability",
    "national_title_probability": "titleProbability",
    "p10_fantasy_points": "p10",
    "p90_fantasy_points": "p90",
    "schedule_difficulty_rank": "scheduleRank",
}


def _base64_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def write_draft_site_data(
    path: Path,
    projections: pd.DataFrame,
    fantasy_points: np.ndarray,
    sample_team_ids: np.ndarray,
    *,
    season: int,
    managers: int,
    teams_per_manager: int,
    draft_slot: int,
    manager_names: list[str] | None = None,
) -> Path:
    if fantasy_points.ndim != 2 or fantasy_points.shape[1] != len(sample_team_ids):
        raise ValueError("simulation samples do not match team ids")
    if not np.allclose(fantasy_points, np.rint(fantasy_points)):
        raise ValueError("web export requires integer fantasy scoring samples")
    if fantasy_points.min() < 0 or fantasy_points.max() > 255:
        raise ValueError("fantasy scoring samples exceed byte storage")
    names = manager_names or [f"Slot {slot}" for slot in range(1, managers + 1)]
    if len(names) != managers or any(not str(name).strip() for name in names):
        raise ValueError("manager names must contain one nonempty name per slot")

    sample_ids = [str(value) for value in sample_team_ids]
    sample_index = {team_id: column for column, team_id in enumerate(sample_ids)}
    ordered = projections.sort_values("overall_rank").copy()
    missing = set(ordered["team_id"].astype(str)).difference(sample_index)
    if missing:
        raise ValueError(f"web export samples missing team ids: {sorted(missing)}")

    teams = []
    for row in ordered.itertuples(index=False):
        record = {
            target: getattr(row, source)
            for source, target in WEB_TEAM_COLUMNS.items()
        }
        record["id"] = str(record["id"])
        for key, value in list(record.items()):
            if isinstance(value, np.integer):
                record[key] = int(value)
            elif isinstance(value, np.floating):
                record[key] = float(value)
        record["sampleIndex"] = sample_index[record["id"]]
        record["logo"] = (
            f"https://a.espncdn.com/i/teamlogos/ncaa/500/{record['id']}.png"
        )
        teams.append(record)

    byte_samples = np.rint(fantasy_points).astype(np.uint8)
    correlation = np.corrcoef(fantasy_points, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    correlation = correlation.astype("<f4")
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "managers": managers,
        "teamsPerManager": teams_per_manager,
        "draftSlot": draft_slot,
        "managerNames": [str(name).strip() for name in names],
        "simulations": int(fantasy_points.shape[0]),
        "teamCount": int(fantasy_points.shape[1]),
        "teams": teams,
        "samplesBase64": _base64_array(byte_samples),
        "correlationsBase64": _base64_array(correlation),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "window.CFB_DRAFT_DATA = "
        + json.dumps(payload, separators=(",", ":"), allow_nan=False)
        + ";\n",
        encoding="utf-8",
    )
    return path
