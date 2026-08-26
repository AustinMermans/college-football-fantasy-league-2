from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .draft import manager_for_pick


STATE_ENDPOINT = "/api/draft-state"


class StateConflictError(ValueError):
    pass


def load_web_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    prefix = "window.CFB_DRAFT_DATA = "
    if not text.startswith(prefix) or not text.endswith(";\n"):
        raise ValueError(f"invalid web data file: {path}")
    return json.loads(text.removeprefix(prefix).removesuffix(";\n"))


class DraftStateStore:
    def __init__(self, results_dir: Path, metadata: dict[str, Any], season: int):
        self.results_dir = results_dir
        self.metadata = metadata
        self.season = season
        self.state_path = results_dir / f"draft_state_{season}.json"
        self.journal_path = results_dir / f"draft_events_{season}.jsonl"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def default_state(self) -> dict[str, Any]:
        return {
            "draftSlot": int(self.metadata["draftSlot"]),
            "picks": [],
            "mode": "win",
            "detailView": "log",
            "search": "",
            "conference": "all",
        }

    def sanitize(self, candidate: Any) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            raise ValueError("state must be an object")
        fallback = self.default_state()
        managers = int(self.metadata["managers"])
        total_picks = managers * int(self.metadata["teamsPerManager"])
        team_count = int(self.metadata["teamCount"])

        raw_picks = candidate.get("picks", fallback["picks"])
        if not isinstance(raw_picks, list):
            raise ValueError("picks must be a list")
        picks: list[int] = []
        for value in raw_picks:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("every pick must be an integer team index")
            if not 0 <= value < team_count:
                raise ValueError(f"team index out of range: {value}")
            if value in picks:
                raise ValueError(f"team index drafted twice: {value}")
            picks.append(value)
        if len(picks) > total_picks:
            raise ValueError("draft contains more picks than configured roster spots")

        draft_slot = candidate.get("draftSlot", fallback["draftSlot"])
        if isinstance(draft_slot, bool) or not isinstance(draft_slot, int):
            raise ValueError("draftSlot must be an integer")
        if not 1 <= draft_slot <= managers:
            raise ValueError("draftSlot is outside the configured league")

        mode = candidate.get("mode", fallback["mode"])
        detail_view = candidate.get("detailView", fallback["detailView"])
        if mode not in {"win", "points"}:
            raise ValueError("mode must be win or points")
        if detail_view not in {"log", "rosters"}:
            raise ValueError("detailView must be log or rosters")

        search = candidate.get("search", fallback["search"])
        conference = candidate.get("conference", fallback["conference"])
        if not isinstance(search, str) or len(search) > 100:
            raise ValueError("search must be a string of at most 100 characters")
        if not isinstance(conference, str) or len(conference) > 100:
            raise ValueError("conference must be a string of at most 100 characters")
        return {
            "draftSlot": draft_slot,
            "picks": picks,
            "mode": mode,
            "detailView": detail_view,
            "search": search,
            "conference": conference,
        }

    def load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self.write(self.default_state(), action="initialize")
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = self.sanitize(payload.get("state", payload))
        normalized = self._payload(
            state,
            revision=int(payload.get("revision", 0)),
            updated_at=payload.get("updatedAt"),
        )
        self._write_csv_exports(normalized)
        return normalized

    def write(
        self,
        candidate: Any,
        *,
        action: str = "update",
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        state = self.sanitize(candidate)
        revision = 1
        if self.state_path.exists():
            current = json.loads(self.state_path.read_text(encoding="utf-8"))
            current_revision = int(current.get("revision", 0))
            if expected_revision is not None and expected_revision != current_revision:
                raise StateConflictError(
                    "draft changed in another tab; reload before recording more picks"
                )
            revision = current_revision + 1
        payload = self._payload(state, revision=revision)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)
        self._write_csv_exports(payload)
        if action != "view":
            event = {
                "recordedAt": payload["updatedAt"],
                "revision": revision,
                "action": str(action)[:40],
                "state": state,
            }
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return payload

    def _payload(
        self, state: dict[str, Any], *, revision: int, updated_at: str | None = None
    ) -> dict[str, Any]:
        manager_names = list(self.metadata.get("managerNames", []))
        teams = list(self.metadata["teams"])
        managers = int(self.metadata["managers"])
        pick_log = []
        for offset, team_index in enumerate(state["picks"]):
            pick_number = offset + 1
            slot = manager_for_pick(pick_number, managers)
            team = teams[team_index]
            pick_log.append(
                {
                    "pickNumber": pick_number,
                    "managerSlot": slot,
                    "manager": manager_names[slot - 1] if manager_names else f"Slot {slot}",
                    "teamIndex": team_index,
                    "teamId": team["id"],
                    "team": team["name"],
                }
            )
        return {
            "version": 1,
            "season": self.season,
            "revision": revision,
            "updatedAt": updated_at or datetime.now(timezone.utc).isoformat(),
            "managerNames": manager_names,
            "state": state,
            "pickLog": pick_log,
        }

    def _write_csv_exports(self, payload: dict[str, Any]) -> None:
        live_path = self.results_dir / "live_picks.csv"
        live_temp = live_path.with_suffix(".csv.tmp")
        live_fields = [
            "pick_number",
            "round",
            "manager_slot",
            "manager",
            "team_id",
            "team",
        ]
        with live_temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=live_fields)
            writer.writeheader()
            for pick in payload["pickLog"]:
                writer.writerow(
                    {
                        "pick_number": pick["pickNumber"],
                        "round": (int(pick["pickNumber"]) - 1)
                        // int(self.metadata["managers"])
                        + 1,
                        "manager_slot": pick["managerSlot"],
                        "manager": pick["manager"],
                        "team_id": pick["teamId"],
                        "team": pick["team"],
                    }
                )
        os.replace(live_temp, live_path)

        roster_path = self.results_dir / f"draft_rosters_{self.season}.csv"
        roster_temp = roster_path.with_suffix(".csv.tmp")
        rounds = int(self.metadata["teamsPerManager"])
        roster_fields = ["manager_slot", "manager"] + [
            f"round_{round_number}" for round_number in range(1, rounds + 1)
        ]
        roster_rows = []
        names = list(self.metadata.get("managerNames", []))
        for slot in range(1, int(self.metadata["managers"]) + 1):
            row: dict[str, Any] = {
                "manager_slot": slot,
                "manager": names[slot - 1] if names else f"Slot {slot}",
            }
            for round_number in range(1, rounds + 1):
                row[f"round_{round_number}"] = ""
            roster_rows.append(row)
        for pick in payload["pickLog"]:
            round_number = (
                (int(pick["pickNumber"]) - 1) // int(self.metadata["managers"])
            ) + 1
            roster_rows[int(pick["managerSlot"]) - 1][f"round_{round_number}"] = pick[
                "team"
            ]
        with roster_temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=roster_fields)
            writer.writeheader()
            writer.writerows(roster_rows)
        os.replace(roster_temp, roster_path)


def make_handler(web_root: Path, store: DraftStateStore) -> type[SimpleHTTPRequestHandler]:
    class DraftRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, directory=str(web_root), **kwargs)

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] == STATE_ENDPOINT:
                self._send_json(store.load())
                return
            super().do_GET()

        def do_PUT(self) -> None:
            if self.path.split("?", 1)[0] != STATE_ENDPOINT:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 100_000:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if "revision" not in request:
                    raise StateConflictError(
                        "this draft tab is stale; reload before recording more picks"
                    )
                state = request.get("state", request)
                payload = store.write(
                    state,
                    action=request.get("action", "update"),
                    expected_revision=int(request["revision"]),
                )
            except StateConflictError as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.CONFLICT)
                return
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)

        def _send_json(
            self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            body = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return DraftRequestHandler


def serve_draft_room(
    project_root: Path, *, host: str = "127.0.0.1", port: int = 8765, season: int
) -> None:
    web_root = project_root / "web"
    metadata = load_web_metadata(web_root / "draft-data.js")
    if int(metadata["season"]) != season:
        raise ValueError("web data season does not match league configuration")
    store = DraftStateStore(project_root / "results", metadata, season)
    store.load()
    server = ThreadingHTTPServer((host, port), make_handler(web_root, store))
    print(f"Persistent draft room: http://{host}:{port}")
    print(f"Draft state: {store.state_path}")
    print(f"Event journal: {store.journal_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
