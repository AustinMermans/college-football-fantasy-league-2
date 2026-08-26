import json

import pytest

from cfb_fantasy.server import DraftStateStore, StateConflictError


def metadata():
    return {
        "draftSlot": 2,
        "managers": 3,
        "teamsPerManager": 2,
        "teamCount": 3,
        "managerNames": ["One", "Two", "Three"],
        "teams": [
            {"id": "a", "name": "Alpha"},
            {"id": "b", "name": "Beta"},
            {"id": "c", "name": "Gamma"},
        ],
    }


def test_state_store_writes_human_readable_pick_log_and_journal(tmp_path):
    store = DraftStateStore(tmp_path, metadata(), 2026)
    payload = store.write(
        {
            "draftSlot": 2,
            "picks": [1, 0],
            "mode": "win",
            "detailView": "log",
            "search": "",
            "conference": "all",
        },
        action="record",
    )

    assert payload["pickLog"][0] == {
        "pickNumber": 1,
        "managerSlot": 1,
        "manager": "One",
        "teamIndex": 1,
        "teamId": "b",
        "team": "Beta",
    }
    saved = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert saved["state"]["picks"] == [1, 0]
    assert store.load()["updatedAt"] == saved["updatedAt"]
    live_rows = (tmp_path / "live_picks.csv").read_text(encoding="utf-8").splitlines()
    assert live_rows[1].endswith(",One,b,Beta")
    roster_rows = (tmp_path / "draft_rosters_2026.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert roster_rows[1].endswith(",Beta,")
    event = json.loads(store.journal_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["action"] == "record"


def test_state_store_rejects_duplicate_or_invalid_picks(tmp_path):
    store = DraftStateStore(tmp_path, metadata(), 2026)
    with pytest.raises(ValueError, match="drafted twice"):
        store.write({**store.default_state(), "picks": [1, 1]})
    with pytest.raises(ValueError, match="out of range"):
        store.write({**store.default_state(), "picks": [9]})


def test_state_store_rejects_stale_revision(tmp_path):
    store = DraftStateStore(tmp_path, metadata(), 2026)
    initial = store.write(store.default_state(), action="initialize")
    store.write(
        {**store.default_state(), "picks": [0]},
        action="record",
        expected_revision=initial["revision"],
    )
    with pytest.raises(StateConflictError, match="another tab"):
        store.write(
            {**store.default_state(), "picks": [1]},
            action="record",
            expected_revision=initial["revision"],
        )
