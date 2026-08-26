# Methodology

## Data

Historical schedules and scores come from the public `cfbfastR-data` season
CSVs. The target-season FBS membership, conference map, and schedules come from
ESPN's public standings and team-schedule JSON feeds. ESPN's current preseason
FPI supplies an all-team roster/recruiting-aware strength prior. SportsDataverse
adds team-talent composites, blue-chip ratios, prior-season opponent-adjusted
EPA, and consensus betting lines. Raw responses are cached under the
monorepo's shared `Data/Raw/cfb_fantasy/` directory.

## Preseason state

Historical games are processed in kickoff order to update each team's state.
For model selection, however, every game in a test season uses one frozen state
derived before that season begins. This matches the information available at a
fantasy draft and prevents September results from improving predictions for
October games in the same backtest. The production features are longer-horizon
scoring-margin difference, current-season team-talent composite difference,
prior-season opponent-adjusted EPA difference, FBS-versus-FCS status,
current-season blue-chip-ratio difference, and home-field/neutral-site status.

Only after those features are recorded does the state update with the result.
At each offseason, Elo and short-run form are regressed toward the national
mean. This preserves useful prior-season information while recognizing roster
turnover.

## Model selection

The factor study compares a home-field baseline, Elo-only logistic regression,
an Elo-plus-score logistic model, and a nonlinear histogram gradient-boosting
model. Each candidate is evaluated with expanding-window, season-level
backtests. The final estimator is selected by mean out-of-sample log loss,
because expected wins depend on calibrated probabilities rather than only
picking winners.

The report also includes Brier score and winner accuracy. These are diagnostics,
not the selection objective.

Feature admission is audited separately with pooled expanding-window out-of-
fold ROC AUC. Greedy forward selection compares the best remaining real feature
with 100 independent random controls at each step. A feature is retained only
when its marginal AUC gain exceeds the random controls' 95th percentile. This
study is a development diagnostic; log loss remains the probability-model
selection objective.

## Probability calibration

Identity, intercept-only adjustment, temperature scaling, Platt scaling, beta
calibration, and isotonic regression are compared in a nested expanding-window
backtest. For each outer test season, the calibrator sees only base-model
predictions that were themselves generated out of sample in earlier seasons.
The method with the lowest mean outer-fold log loss is refit on all historical
out-of-fold predictions and applied to the final score model. Parametric maps
are constrained to remain monotonic so they do not change ranking; isotonic may
create ties.

Diagnostics include log loss, Brier score, ROC AUC, calibration intercept and
slope, ICI, E50, E90, Emax, 15-bin ECE, and the isotonic CORP form of the Murphy
Brier decomposition into miscalibration, discrimination, and uncertainty. The
ICI family uses a LOWESS-style local-linear calibration curve with span 0.75.

## Preseason consensus

The selected score model and current FPI each produce a probability for FBS-vs-
FBS games. Historical game-level FPI projections provide an external benchmark,
but they update during the season and are not mislabeled as a frozen-preseason
backtest. For 2026, the FPI weight and point-to-probability scale are fitted to
current ESPN projected wins for teams whose totals do not imply postseason
games. Games against opponents without FPI coverage use the score model alone.
The component probabilities and effective weight are retained in the output.

This consensus addresses the main limitation of a prior-score model: coaching,
recruiting, and roster turnover between seasons. Calibration parameters and
benchmark metrics are written to `results/ensemble_calibration.json` and
`results/ensemble_backtest.csv`.

## Live market consensus

When a line is available, a regularized logistic model combines FPI log odds
with the home-team point spread. It is tested with an expanding window using
2023 as the first training season and 2024 as the first test season. This is a
live forecasting layer rather than a frozen-preseason input. Games without a
published line use FPI, and completed games are replaced with their realized
0/1 result.

## Incremental in-season update

The daily workflow refreshes ESPN FPI, which incorporates the season's new team
performance data, and converts the new ratings into matchup probabilities with
the preseason-validated logistic scale and home-field adjustment. A current
market line overrides that FPI-only probability through the validated market
consensus model. Completed games are fixed as realized outcomes. The workflow
then reruns all 20,000 joint season simulations, including conference title
games and the CFP, so every regular-season and postseason component of EV can
change after each daily update. The fixed random seed prevents Monte Carlo
noise from looking like a model change when the inputs have not changed.

## Season simulation

Regular-season games are simulated from model probabilities with a Gaussian
copula that preserves each game's marginal probability while sharing a latent
team-season strength shock across the schedule. The correlation is estimated
from within-team residuals in the frozen-preseason walk-forward predictions.
Conference title
participants are selected by conference winning percentage, total wins, and a
strength tiebreak. Title games are neutral except the home-hosted Pac-12 game,
where the higher finisher hosts. A committee proxy ranks teams using record, simulated
strength of schedule, and preseason model strength. The ACC, Big Ten, Big 12,
and SEC champions plus the highest-ranked champion from the other six FBS
conferences receive automatic bids; seven at-large teams complete the current
12-team CFP format, with Notre Dame protected when ranked in the top 12. Seeds
1-4 receive byes, seeds 5-8 host first-round games, and later rounds are neutral;
the fixed bracket is simulated through the championship with no reseeding.

Fantasy points are the sum of wins in the counted stages. Outputs decompose
regular-season, conference-title, and CFP expected points. `playoff_rank_lift`
shows how many board positions a team gains specifically from expected CFP
wins. During our turn, the live optimizer completes the remaining snake draft
in board order and uses the joint season samples to rank candidates by our
probability of finishing first, including ties fractionally.

## Live draft optimization

The browser consumes the complete 20,000-by-team fantasy-point sample matrix,
not independent team means. For each of the 30 highest-ranked candidates likely
to be available at our target pick, it inserts that candidate into our roster,
fills every later snake pick in board order, and scores all 12 completed rosters
in every simulated season. The primary objective is fractional first-place
probability; expected finish, expected roster points, and expected margin are
also computed. The optional expected-points mode ranks the same candidate
completions by mean roster points.

This portfolio calculation is where schedule interactions enter the draft
decision. If two teams play each other, compete for the same conference title,
or collide in the simulated CFP bracket, that dependence is already present in
their joint outcome samples. An additional fixed same-conference penalty would
double-count some of that information, so conference concentration and average
pair correlation are displayed as diagnostics rather than imposed as ad hoc
penalties. Before our turn, intervening opponent picks are projected in board
order; live recorded picks replace those assumptions immediately.
