from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def write_model_card(
    path: Path,
    projections: pd.DataFrame,
    factor_study: pd.DataFrame,
    feature_study: pd.DataFrame,
    score_calibration_study: pd.DataFrame,
    score_calibration: dict[str, float | str],
    fpi_calibration_study: pd.DataFrame,
    ensemble_backtest: pd.DataFrame,
    ensemble_calibration: dict[str, float],
    market_backtest: pd.DataFrame,
    market_summary: dict[str, float],
    *,
    season: int,
    simulations: int,
    schedule_games: int,
) -> Path:
    selected = factor_study.iloc[0]
    top = projections.head(25).copy()
    for column in [
        "expected_fantasy_points",
        "expected_regular_wins",
        "expected_playoff_points",
        "playoff_probability",
        "average_opponent_rating",
    ]:
        top[column] = top[column].map(lambda value: f"{value:.3f}")
    table = top[
        [
            "overall_rank",
            "team",
            "conference",
            "expected_fantasy_points",
            "expected_regular_wins",
            "expected_playoff_points",
            "playoff_rank_lift",
            "playoff_probability",
            "schedule_difficulty_rank",
        ]
    ].to_markdown(index=False)
    factor = factor_study.copy()
    for column in [
        "mean_log_loss",
        "mean_brier",
        "mean_accuracy",
        "mean_roc_auc",
        "worst_log_loss",
    ]:
        factor[column] = factor[column].map(lambda value: f"{value:.4f}")
    factor_table = factor.to_markdown(index=False)
    feature = feature_study.copy()
    for column in [
        "baseline_auc",
        "auc_with_feature",
        "delta_auc",
        "random_delta_auc_mean",
        "random_delta_auc_p95",
    ]:
        feature[column] = feature[column].map(lambda value: f"{value:.5f}")
    feature_table = feature.to_markdown(index=False)
    calibration_columns = [
        "rank",
        "method",
        "mean_log_loss",
        "mean_brier",
        "mean_roc_auc",
        "mean_calibration_intercept",
        "mean_calibration_slope",
        "mean_ici",
        "mean_e50",
        "mean_e90",
        "mean_emax",
        "mean_murphy_miscalibration",
        "mean_murphy_discrimination",
    ]
    calibration = score_calibration_study[calibration_columns].copy()
    for column in calibration_columns[2:]:
        calibration[column] = calibration[column].map(lambda value: f"{value:.5f}")
    calibration_table = calibration.to_markdown(index=False)
    fpi_calibration_columns = [
        "rank",
        "method",
        "mean_log_loss",
        "mean_brier",
        "mean_roc_auc",
        "mean_ici",
        "mean_e90",
        "mean_emax",
    ]
    fpi_calibration = fpi_calibration_study[fpi_calibration_columns].copy()
    for column in fpi_calibration_columns[2:]:
        fpi_calibration[column] = fpi_calibration[column].map(
            lambda value: f"{value:.5f}"
        )
    fpi_calibration_table = fpi_calibration.to_markdown(index=False)
    ensemble = (
        ensemble_backtest.groupby("model", as_index=False)
        .agg(
            seasons=("test_season", "count"),
            games=("games", "sum"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        .sort_values("mean_log_loss")
    )
    for column in ["mean_log_loss", "mean_brier", "mean_accuracy", "mean_roc_auc"]:
        ensemble[column] = ensemble[column].map(lambda value: f"{value:.4f}")
    ensemble_table = ensemble.to_markdown(index=False)
    market = (
        market_backtest.groupby("model", as_index=False)
        .agg(
            seasons=("test_season", "count"),
            games=("games", "sum"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        .sort_values("mean_log_loss")
    )
    for column in ["mean_log_loss", "mean_brier", "mean_accuracy", "mean_roc_auc"]:
        market[column] = market[column].map(lambda value: f"{value:.4f}")
    market_table = market.to_markdown(index=False)
    text = f"""# {season} College Football Fantasy Draft Report

Generated {datetime.now(timezone.utc).isoformat()} from {schedule_games} known
regular-season games and {simulations:,} season simulations.

## Scoring used

One point per regular-season, conference-championship, or CFP win. Losses and
ties score zero. Non-CFP bowl wins are excluded.

## Selected game model

`{selected['model']}` was selected by expanding-window preseason mean log loss
(`{selected['mean_log_loss']:.4f}`). Lower log loss is better because the draft
decision uses win probabilities, not only winner accuracy. Every matchup in a
test season is frozen at information available before that season begins.

`{score_calibration['method']}` calibration is applied to the score model. It was
selected by nested expanding-window log loss: each test season's calibrator was
fit only on prior-season out-of-fold predictions. Calibration mappings are
constrained to be monotonic except isotonic's permitted flat steps.

Archived game-level FPI, which updates during each season, provides an external
performance benchmark but is not presented as a frozen-preseason backtest.

For the final forecast, the score probability receives weight
`{1.0 - ensemble_calibration['fpi_weight']:.0%}` and current ESPN FPI receives
weight `{ensemble_calibration['fpi_weight']:.0%}`. The weight and
`{ensemble_calibration['fpi_logistic_scale']:.2f}`-point logistic scale minimize
RMSE against current ESPN projected wins for
`{int(ensemble_calibration['projected_wins_calibration_teams'])}` teams whose FPI
totals do not imply postseason games. The resulting win-total RMSE is
`{ensemble_calibration['projected_wins_rmse']:.3f}`. Both component probabilities
remain in the game-level CSV.

## Factor study

{factor_table}

## Marginal feature study

Features are added greedily using pooled expanding-window out-of-fold ROC AUC.
At every step, the selected feature must beat the 95th percentile gain from 100
independent random control features. The first rejected row is the stopping
point; unselected candidates at that step are retained in `feature_screen.csv`.

{feature_table}

## Probability calibration study

Lower log loss, Brier, ICI, E50, E90, Emax, and Murphy miscalibration are better;
higher AUC and Murphy discrimination are better. A perfect calibration
intercept and slope are 0 and 1. ICI-family metrics use a local-linear smooth
calibration curve with span 0.75. The Murphy columns use the exact isotonic CORP
Brier decomposition.

{calibration_table}

Archived updated-pregame FPI was run through the same calibration contest.
Identity wins on log loss, so no beta, Platt, or isotonic map is layered onto
FPI. This is an external diagnostic rather than a frozen-preseason validation.

{fpi_calibration_table}

## Updated-FPI benchmark

{ensemble_table}

## Betting-market consensus

For games with a published line, the deployed probability combines the FPI
logit and home-team point spread in a regularized logistic regression. The
market layer uses an expanding window: the 2024 test model trains on 2023, and
the 2025 test model trains on 2023-24. Games without a line fall back to FPI.
Market lines are not frozen-preseason information; this layer is for the daily
live forecast.

Across {int(market_summary['evaluation_games']):,} covered 2024-25 games, log
loss improves from `{market_summary['score_log_loss']:.4f}` for the enhanced
frozen-preseason score model and `{market_summary['fpi_log_loss']:.4f}` for FPI
to `{market_summary['market_log_loss']:.4f}`. Relative to the 50/50 log-loss
baseline, that is a `{market_summary['skill_improvement_vs_score']:.1%}` increase
in predictive skill versus the score model. This percentage is skill lift, not
a claim of the same percentage reduction in raw log loss.

{market_table}

## Incremental in-season update

The public leaderboard is rebuilt daily from current ESPN results and FPI plus
available market lines. Completed games become fixed 0/1 outcomes; current FPI
updates uncovered future games; the market model updates covered future games.
The complete `{simulations:,}`-season simulation is then rerun with a fixed
random seed, so conference-title, CFP, national-title, and total expected value
all respond to the latest season state without introducing avoidable Monte
Carlo drift.

## Live draft optimizer

The browser draft room evaluates candidate teams against the full joint
`{simulations:,}`-season fantasy-point sample matrix. For each of the 30
highest-ranked candidates expected to be available at our next snake pick, it
completes all 12 rosters in board order and reports fractional first-place
probability, expected finish, expected roster points, and expected margin. The
default recommendation maximizes first-place probability; expected points is an
optional alternative objective.

Head-to-head games, conference-title competition, and CFP bracket collisions
are already represented in the joint samples. Same-conference count and average
pair correlation are therefore diagnostics, not a second arbitrary penalty.
Before our turn, intervening opponent picks are projected in board order and are
replaced as actual selections are recorded.

## Top 25 draft board

{table}

## Reading schedule difficulty

Schedule difficulty rank 1 is the hardest known regular-season schedule by
average preseason opponent rating. Expected regular-season wins already price
each actual opponent and venue separately; the schedule column is a diagnostic,
not an extra adjustment layered on afterward.

## Limitations

The conference-title and CFP fields are simulations of unknown postseason
participants. Committee selection and conference tiebreakers are transparent
proxies, not claims about official decisions. Injuries, late transfers, and
future schedule changes are not present unless the data snapshot is refreshed.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
