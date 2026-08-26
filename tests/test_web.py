import json

import numpy as np
import pandas as pd

from cfb_fantasy.web import write_draft_site_data


def test_web_export_contains_ranked_teams_and_byte_samples(tmp_path):
    projections = pd.DataFrame(
        [
            {
                "team_id": "a",
                "team": "Alpha",
                "conference": "ACC",
                "overall_rank": 1,
                "expected_fantasy_points": 10.0,
                "expected_regular_points": 9.0,
                "expected_conference_title_points": 0.4,
                "expected_playoff_points": 0.6,
                "playoff_probability": 0.5,
                "national_title_probability": 0.1,
                "p10_fantasy_points": 7.0,
                "p90_fantasy_points": 13.0,
                "schedule_difficulty_rank": 5,
            },
            {
                "team_id": "b",
                "team": "Beta",
                "conference": "SEC",
                "overall_rank": 2,
                "expected_fantasy_points": 8.0,
                "expected_regular_points": 7.5,
                "expected_conference_title_points": 0.2,
                "expected_playoff_points": 0.3,
                "playoff_probability": 0.2,
                "national_title_probability": 0.02,
                "p10_fantasy_points": 5.0,
                "p90_fantasy_points": 11.0,
                "schedule_difficulty_rank": 10,
            },
        ]
    )
    samples = np.array([[10.0, 7.0], [8.0, 9.0], [12.0, 6.0]])
    path = write_draft_site_data(
        tmp_path / "draft-data.js",
        projections,
        samples,
        np.array(["a", "b"]),
        season=2026,
        managers=12,
        teams_per_manager=4,
        draft_slot=3,
        manager_names=[f"Manager {slot}" for slot in range(1, 13)],
    )
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text.removeprefix("window.CFB_DRAFT_DATA = ").removesuffix(";\n"))
    assert payload["simulations"] == 3
    assert payload["draftSlot"] == 3
    assert payload["managerNames"][2] == "Manager 3"
    assert payload["teams"][0]["name"] == "Alpha"
    assert payload["teams"][0]["sampleIndex"] == 0
    assert payload["samplesBase64"]
