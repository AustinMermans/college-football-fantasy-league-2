from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


CFBFASTR_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/"
    "main/schedules/csv/cfb_schedules_{season}.csv"
)
HISTORICAL_FPI_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_cfb_power_index/power_index_{season}.csv"
)
ESPN_STANDINGS_URL = (
    "https://site.api.espn.com/apis/v2/sports/football/college-football/standings"
    "?region=us&lang=en&contentorigin=espn&isqualified=true&type=0&level=3"
    "&sort=winpercent%3Adesc&season={season}&seasontype=2"
)
ESPN_SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/"
    "teams/{team_id}/schedule?season={season}"
)
ESPN_FPI_URL = (
    "https://site.web.api.espn.com/apis/fitt/v3/sports/football/"
    "college-football/powerindex?region=us&lang=en&season={season}"
    "&sort=fpi.fpi%3Adesc"
)
USER_AGENT = "college-football-fantasy/0.1 research"


def default_raw_dir(project_root: Path) -> Path:
    return project_root.parent / "Data" / "Raw" / "cfb_fantasy"


def _download(url: str, destination: Path, retries: int = 3) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            return destination
        except Exception as exc:  # urllib raises several transport exceptions
            error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                "60",
                url,
                "-o",
                str(temporary),
            ],
            check=True,
        )
        os.replace(temporary, destination)
        return destination
    except (OSError, subprocess.CalledProcessError) as curl_error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}") from curl_error


def fetch_historical_seasons(
    raw_dir: Path,
    start_season: int,
    end_season: int,
    *,
    refresh: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for season in range(start_season, end_season + 1):
        destination = raw_dir / "cfbfastR" / f"cfb_schedules_{season}.csv"
        if refresh or not destination.exists():
            _download(CFBFASTR_URL.format(season=season), destination)
        paths.append(destination)
    return paths


def fetch_historical_fpi(
    raw_dir: Path,
    start_season: int,
    end_season: int,
    *,
    refresh: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for season in range(max(2015, start_season), end_season + 1):
        destination = raw_dir / "sportsdataverse" / f"power_index_{season}.csv"
        if refresh or not destination.exists():
            _download(HISTORICAL_FPI_URL.format(season=season), destination)
        paths.append(destination)
    return paths


def attach_historical_fpi(
    games: pd.DataFrame, paths: Iterable[Path]
) -> pd.DataFrame:
    frames = [
        pd.read_csv(path, dtype={"game_id": str, "team_id": str}) for path in paths
    ]
    if not frames:
        frame = games.copy()
        frame["fpi_home_probability"] = float("nan")
        frame["fpi_predicted_margin"] = float("nan")
        return frame
    ratings = pd.concat(frames, ignore_index=True)
    required = {"game_id", "team_id", "gameprojection", "teampredptdiff"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"historical FPI data missing columns: {sorted(missing)}")
    if ratings.duplicated(["game_id", "team_id"]).any():
        raise ValueError("historical FPI contains duplicate team-game rows")
    home = ratings[list(required)].rename(
        columns={
            "team_id": "home_id",
            "gameprojection": "fpi_home_probability",
            "teampredptdiff": "fpi_predicted_margin",
        }
    )
    home["fpi_home_probability"] = pd.to_numeric(
        home["fpi_home_probability"], errors="coerce"
    ) / 100.0
    home["fpi_predicted_margin"] = pd.to_numeric(
        home["fpi_predicted_margin"], errors="coerce"
    )
    frame = games.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["home_id"] = frame["home_id"].astype(str)
    frame = frame.merge(
        home, on=["game_id", "home_id"], how="left", validate="one_to_one"
    )
    valid = frame["fpi_home_probability"].dropna().between(0.0, 1.0)
    if not valid.all():
        raise ValueError("historical FPI probabilities must be between zero and one")
    return frame


def load_historical_games(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    if not frames:
        raise ValueError("no historical schedule files supplied")
    games = pd.concat(frames, ignore_index=True)
    required = {
        "game_id",
        "season",
        "week",
        "season_type",
        "start_date",
        "completed",
        "neutral_site",
        "home_id",
        "home_team",
        "home_division",
        "home_conference",
        "home_points",
        "away_id",
        "away_team",
        "away_division",
        "away_conference",
        "away_points",
    }
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"historical data missing columns: {sorted(missing)}")
    games["start_date"] = pd.to_datetime(games["start_date"], utc=True)
    games["completed"] = games["completed"].astype(str).str.lower().eq("true")
    games = games[
        games["completed"]
        & games["home_points"].notna()
        & games["away_points"].notna()
        & games["season_type"].isin(["regular", "postseason"])
        & (
            games["home_division"].astype(str).str.lower().eq("fbs")
            | games["away_division"].astype(str).str.lower().eq("fbs")
        )
    ].copy()
    for column in ("home_id", "away_id"):
        games[column] = games[column].astype("Int64").astype(str)
    games["neutral_site"] = games["neutral_site"].fillna(False).astype(bool)
    return games.sort_values(["start_date", "game_id"]).reset_index(drop=True)


def _standings_entries(node: dict[str, Any], conference: str | None = None):
    label = node.get("shortName") or node.get("name")
    current = label if node.get("isConference", conference is None) else conference
    for entry in node.get("standings", {}).get("entries", []):
        yield entry, current
    for child in node.get("children", []):
        yield from _standings_entries(child, conference=current)


def parse_fbs_teams(payload: dict[str, Any]) -> pd.DataFrame:
    rows: dict[str, dict[str, str]] = {}
    for entry, conference in _standings_entries(payload):
        team = entry.get("team", {})
        team_id = str(team.get("id", ""))
        if not team_id:
            continue
        rows[team_id] = {
            "team_id": team_id,
            "team": team.get("location") or team.get("shortDisplayName"),
            "display_name": team.get("displayName"),
            "conference": conference or "FBS Independents",
        }
    teams = pd.DataFrame(rows.values()).sort_values("team").reset_index(drop=True)
    if len(teams) < 130:
        raise ValueError(f"expected at least 130 FBS teams, found {len(teams)}")
    return teams


def fetch_target_season(
    raw_dir: Path,
    season: int,
    *,
    refresh: bool = False,
    workers: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    espn_dir = raw_dir / "espn" / str(season)
    standings_path = espn_dir / "standings.json"
    if refresh or not standings_path.exists():
        _download(ESPN_STANDINGS_URL.format(season=season), standings_path)
    teams = parse_fbs_teams(json.loads(standings_path.read_text(encoding="utf-8")))

    def fetch_one(team_id: str) -> Path:
        path = espn_dir / "schedules" / f"{team_id}.json"
        if refresh or not path.exists():
            _download(ESPN_SCHEDULE_URL.format(team_id=team_id, season=season), path)
        return path

    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, team_id): team_id for team_id in teams["team_id"]}
        for future in as_completed(futures):
            paths.append(future.result())
    return teams, parse_espn_schedules(paths, teams, season)


def parse_fpi(payload: dict[str, Any]) -> pd.DataFrame:
    category = next(
        item for item in payload.get("categories", []) if item.get("name") == "fpi"
    )
    names = category["names"]
    rows = []
    for item in payload.get("teams", []):
        values_by_category = {
            block.get("name"): block.get("values", [])
            for block in item.get("categories", [])
        }
        values = values_by_category.get("fpi", [])
        if len(values) != len(names):
            continue
        metrics = dict(zip(names, values))
        team = item.get("team", {})
        rows.append(
            {
                "team_id": str(team.get("id")),
                "fpi": float(metrics["fpi"]),
                "fpi_rank": int(metrics["fpirank"]),
                "fpi_projected_wins": float(metrics["projectedw"]),
                "fpi_projected_losses": float(metrics["projectedl"]),
                "fpi_playoff_probability": float(metrics["probmakeplayoffs"]) / 100.0,
                "fpi_conference_title_probability": float(metrics["probwinconf"]) / 100.0,
                "fpi_national_title_probability": float(metrics["probwintitle"]) / 100.0,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) < 130 or frame["team_id"].duplicated().any():
        raise ValueError(f"invalid FPI coverage: {len(frame)} unique rows expected >= 130")
    return frame.sort_values("fpi_rank").reset_index(drop=True)


def fetch_fpi(
    raw_dir: Path, season: int, *, refresh: bool = False
) -> pd.DataFrame:
    path = raw_dir / "espn" / str(season) / "fpi.json"
    if refresh or not path.exists():
        _download(ESPN_FPI_URL.format(season=season), path)
    return parse_fpi(json.loads(path.read_text(encoding="utf-8")))


def _score_value(competitor: dict[str, Any]) -> float | None:
    score = competitor.get("score")
    if isinstance(score, dict):
        score = score.get("value", score.get("displayValue"))
    if score in (None, ""):
        return None
    return float(score)


def parse_espn_schedules(
    paths: Iterable[Path], teams: pd.DataFrame, season: int
) -> pd.DataFrame:
    fbs_ids = set(teams["team_id"].astype(str))
    conference = dict(zip(teams["team_id"].astype(str), teams["conference"]))
    events: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for event in payload.get("events", []):
            if str(event.get("season", {}).get("year", season)) != str(season):
                continue
            competition = (event.get("competitions") or [{}])[0]
            competitors = competition.get("competitors", [])
            sides = {item.get("homeAway"): item for item in competitors}
            if "home" not in sides or "away" not in sides:
                continue
            home, away = sides["home"], sides["away"]
            home_id, away_id = str(home.get("id")), str(away.get("id"))
            if home_id not in fbs_ids and away_id not in fbs_ids:
                continue
            status = competition.get("status", event.get("status", {})).get("type", {})
            home_conf, away_conf = conference.get(home_id), conference.get(away_id)
            season_type = event.get("seasonType", {}).get("name", "regular season")
            notes = " ".join(
                str(note.get("headline") or note.get("text") or "")
                for note in competition.get("notes", [])
            ).strip()
            home_name = home.get("team", {}).get("location")
            away_name = away.get("team", {}).get("location")
            same_conference = bool(home_conf and home_conf == away_conf)
            is_army_navy = {home_name, away_name} == {"Army", "Navy"}
            row = {
                "game_id": str(event.get("id")),
                "season": season,
                "week": event.get("week", {}).get("number", 0),
                "season_type": "postseason" if "post" in season_type.lower() else "regular",
                "event_name": event.get("name", ""),
                "notes": notes,
                "start_date": event.get("date"),
                "completed": bool(status.get("completed", False)),
                "neutral_site": bool(competition.get("neutralSite", False)),
                "conference_game": same_conference and not is_army_navy,
                "home_id": home_id,
                "home_team": home_name,
                "home_division": "fbs" if home_id in fbs_ids else "fcs",
                "home_conference": home_conf,
                "home_points": _score_value(home),
                "away_id": away_id,
                "away_team": away_name,
                "away_division": "fbs" if away_id in fbs_ids else "fcs",
                "away_conference": away_conf,
                "away_points": _score_value(away),
            }
            events[row["game_id"]] = row
    games = pd.DataFrame(events.values())
    if games.empty:
        raise ValueError(f"no ESPN schedule events parsed for {season}")
    games["start_date"] = pd.to_datetime(games["start_date"], utc=True)
    return games.sort_values(["start_date", "game_id"]).reset_index(drop=True)


def write_snapshot(
    games: pd.DataFrame, teams: pd.DataFrame, derived_dir: Path, season: int
) -> tuple[Path, Path]:
    derived_dir.mkdir(parents=True, exist_ok=True)
    games_path = derived_dir / f"schedule_{season}.csv"
    teams_path = derived_dir / f"teams_{season}.csv"
    games.to_csv(games_path, index=False)
    teams.to_csv(teams_path, index=False)
    return games_path, teams_path
