# College Football Fantasy

A public roster board and live scoreboard for a 12-manager fantasy league that
drafts college football teams instead of players.

**Live site:** [austinmermans.github.io/college-football-fantasy](https://austinmermans.github.io/college-football-fantasy/)

## League scoring

- 1 point per regular-season win
- 1 point for a conference championship win
- 1 point for each College Football Playoff win
- 0 points for losses or ties
- Ordinary bowl wins do not count

The site includes a manager-relative roster view, league standings, a drafted
team EV leaderboard with postseason decomposition, each team's next scheduled
game, and the complete ten-round snake draft ledger. It is read-only; the
recorded roster file is the source of truth.

## Daily score refresh

GitHub Actions downloads the current season schedules from ESPN every day at
4:00 AM America/Los_Angeles, rebuilds `web/scoreboard-data.js`, and deploys the
static `web/` directory to GitHub Pages. The workflow can also be run manually
from the Actions tab.

The refresh command is:

```bash
python scripts/refresh_scoreboard.py --refresh
```

A failed upstream refresh stops the deployment, leaving the most recent working
Pages build online.

## Run locally

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python scripts/refresh_scoreboard.py
.venv/bin/python -m http.server 8765 --directory web
```

Open `http://127.0.0.1:8765`.

## Data and model

`results/live_picks.csv` contains the recorded draft. The daily scoreboard uses
ESPN's public team schedule endpoints for completed results and upcoming games.
`results/team_projections_2026.csv` contains the preseason schedule-adjusted
expected values used as context in the roster and standings views.

The underlying forecast prices every known matchup using both teams' pregame
strength and venue, then simulates conference championships and the 12-team
playoff. See `results/MODEL_CARD.md`, `docs/METHODOLOGY.md`, and
`docs/SOURCES.md` for its validation, calibration, and source details.

## Update rosters

Edit `results/live_picks.csv` while preserving the columns and chronological
pick order, regenerate the scoreboard data, and push to `main`. A push triggers
an immediate Pages deployment in addition to the daily schedule.
