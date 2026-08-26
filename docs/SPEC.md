# College Football Fantasy Forecast - Model Contract

## Objective

Forecast total fantasy points for every FBS team before the 2026 season and
turn those forecasts into a 12-manager, ten-round snake-draft board. One
point is awarded for every counted win. Losses and ties are worth zero.

## Counted games

- Every scheduled regular-season win counts.
- A conference championship win counts.
- Every College Football Playoff win counts, including the national title game.
- Non-CFP bowl wins do not count in the default league configuration.
- Cancelled games and games without a final score do not count.

The last two choices are explicit configuration rather than hidden modeling
assumptions. If the league votes to count ordinary bowls, change
`non_playoff_bowl_win` in `config/league.toml`; forecasting unknown bowl
matchups is intentionally deferred.

## Decision outputs

The model must produce:

1. A win probability for every known 2026 regular-season game.
2. Expected regular-season, conference-title, and CFP wins and points for each
   FBS team, including the draft-rank effect attributable to CFP wins.
3. Simulation percentiles and playoff/conference-title probabilities.
4. A value-ranked draft board and a 12-manager snake grid.
5. Walk-forward backtests and a factor-ablation report.
6. A live recommendation that maximizes first-place probability from the joint
   season simulation when it is our turn.
7. A static browser draft room that records the snake draft in order, separates
   our roster from opponent picks, persists state locally, and recalculates the
   best available team for our next pick.
8. When served through the local application, an atomic JSON state file and an
   append-only action journal that survive browser-data deletion and retain the
   manager assigned to every active pick.

## Information boundary

For a historical game, every feature must be computed from games completed
before kickoff. Backtests train only on seasons earlier than the test season.
The 2026 forecast uses results through the end of 2025 plus the known 2026
schedule. It may not use 2026 game outcomes or end-of-season ratings.

## Known approximations

- Conference tiebreakers are approximated by conference winning percentage,
  total wins, then model strength.
- The CFP committee is approximated with record, schedule strength, and model
  strength. The ACC, Big Ten, Big 12, and SEC champions plus the highest-ranked
  champion from the other six FBS conferences receive automatic bids. Notre
  Dame is guaranteed a bid when the committee proxy ranks it in the top 12.
- Future injuries, transfers after the data snapshot, and ordinary bowl
  assignments are outside the current free-data model.
- Game simulations use an estimated Gaussian-copula team-season shock. It
  preserves each matchup's marginal probability but remains an approximation
  of injuries, development, and other correlated season effects.
- The live optimizer assumes that unrecorded opponent picks follow the model
  board order. It evaluates the top 30 candidates expected to reach our next
  pick and completes the rest of the ten-round draft in that same order.
