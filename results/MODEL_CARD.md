# 2026 College Football Fantasy Draft Report

Generated 2026-08-26T04:37:18.156035+00:00 from 892 known
regular-season games and 20,000 season simulations.

## Scoring used

One point per regular-season, conference-championship, or CFP win. Losses and
ties score zero. Non-CFP bowl wins are excluded.

## Selected game model

`score_logit` was selected by expanding-window preseason mean log loss
(`0.5424`). Lower log loss is better because the draft
decision uses win probabilities, not only winner accuracy. Every matchup in a
test season is frozen at information available before that season begins.

`beta` calibration is applied to the score model. It was
selected by nested expanding-window log loss: each test season's calibrator was
fit only on prior-season out-of-fold predictions. Calibration mappings are
constrained to be monotonic except isotonic's permitted flat steps.

Archived game-level FPI, which updates during each season, provides an external
performance benchmark but is not presented as a frozen-preseason backtest.

For the final forecast, the score probability receives weight
`0%` and current ESPN FPI receives
weight `100%`. The weight and
`9.44`-point logistic scale minimize
RMSE against current ESPN projected wins for
`108` teams whose FPI
totals do not imply postseason games. The resulting win-total RMSE is
`0.093`. Both component probabilities
remain in the game-level CSV.

## Factor study

|   rank | model         |   seasons |   games |   mean_log_loss |   mean_brier |   mean_accuracy |   mean_roc_auc |   worst_log_loss |
|-------:|:--------------|----------:|--------:|----------------:|-------------:|----------------:|---------------:|-----------------:|
|      1 | score_logit   |         8 |    6701 |          0.5424 |       0.1853 |          0.7052 |         0.77   |           0.5732 |
|      2 | score_boost   |         8 |    6701 |          0.5446 |       0.1862 |          0.7038 |         0.7682 |           0.5845 |
|      3 | elo_logit     |         8 |    6701 |          0.5864 |       0.2009 |          0.6885 |         0.7183 |           0.6144 |
|      4 | home_baseline |         8 |    6701 |          0.6594 |       0.2333 |          0.6294 |         0.5    |           0.68   |

## Marginal feature study

Features are added greedily using pooled expanding-window out-of-fold ROC AUC.
At every step, the selected feature must beat the 95th percentile gain from 100
independent random control features. The first rejected row is the stopping
point; unselected candidates at that step are retained in `feature_screen.csv`.

|   step | feature               |   baseline_auc |   auc_with_feature |   delta_auc |   random_delta_auc_mean |   random_delta_auc_p95 | accepted   |
|-------:|:----------------------|---------------:|-------------------:|------------:|------------------------:|-----------------------:|:-----------|
|      1 | long_margin_diff      |        0.5     |            0.72143 |     0.22143 |                -0.00383 |                0.00745 | True       |
|      2 | talent_composite_diff |        0.72143 |            0.75882 |     0.03739 |                -8e-05   |                0.0002  | True       |
|      3 | net_adj_epa_diff      |        0.75882 |            0.76482 |     0.006   |                -6e-05   |                0.00014 | True       |
|      4 | fbs_status_diff       |        0.76482 |            0.7701  |     0.00528 |                -5e-05   |                0.00014 | True       |
|      5 | blue_chip_ratio_diff  |        0.7701  |            0.77059 |     0.00049 |                -5e-05   |                0.00022 | True       |
|      6 | home_field            |        0.77059 |            0.77103 |     0.00044 |                -5e-05   |                0.00014 | True       |
|      7 | margin_form_diff      |        0.77103 |            0.77115 |     0.00012 |                -6e-05   |                0.00016 | False      |

## Additional factor research

A second screen tested 16 deployable lagged-efficiency and recruiting variables
plus historical returning-production measures. Candidates had to beat the 95th
percentile random-feature gain and retain one-sided significance after Holm
family-wise correction. None passed both gates. Prior EPA/play margin was the
best deployable score-layer addition (AUC `+0.00060`, log loss `-0.00061`, Holm
`p=1.000`). Offensive returning production was larger (AUC `+0.00810`, log loss
`-0.00601`) but failed correction (`p=0.156`) and has no 2026 public file.

All candidates were also tested against the deployed FPI-plus-market model on
1,830 covered 2024-25 games. Defensive returning production was best (AUC
`+0.00083`, log loss `-0.00103`) but was not significant before correction
(`p=0.122` using season-week blocks) and had Holm `p=1.000`. Production
therefore remains unchanged.
Full results and deferred-factor notes are in `docs/FACTOR_RESEARCH.md`.

## Probability calibration study

Lower log loss, Brier, ICI, E50, E90, Emax, and Murphy miscalibration are better;
higher AUC and Murphy discrimination are better. A perfect calibration
intercept and slope are 0 and 1. ICI-family metrics use a local-linear smooth
calibration curve with span 0.75. The Murphy columns use the exact isotonic CORP
Brier decomposition.

|   rank | method      |   mean_log_loss |   mean_brier |   mean_roc_auc |   mean_calibration_intercept |   mean_calibration_slope |   mean_ici |   mean_e50 |   mean_e90 |   mean_emax |   mean_murphy_miscalibration |   mean_murphy_discrimination |
|-------:|:------------|----------------:|-------------:|---------------:|-----------------------------:|-------------------------:|-----------:|-----------:|-----------:|------------:|-----------------------------:|-----------------------------:|
|      1 | beta        |         0.54185 |      0.18498 |         0.77   |                      0.0512  |                  1.01881 |    0.02306 |    0.02153 |    0.04269 |     0.07028 |                      0.00625 |                      0.05421 |
|      2 | platt       |         0.54234 |      0.18505 |         0.77   |                      0.04781 |                  1.01678 |    0.02389 |    0.02174 |    0.04463 |     0.07369 |                      0.00632 |                      0.05421 |
|      3 | identity    |         0.54243 |      0.18527 |         0.77   |                      0.07251 |                  0.9543  |    0.02526 |    0.02157 |    0.05161 |     0.08402 |                      0.00655 |                      0.05421 |
|      4 | intercept   |         0.54247 |      0.18529 |         0.77   |                      0.06908 |                  0.95429 |    0.02605 |    0.0225  |    0.05086 |     0.08367 |                      0.00657 |                      0.05421 |
|      5 | temperature |         0.54256 |      0.18514 |         0.77   |                      0.07252 |                  1.00947 |    0.02459 |    0.02142 |    0.04959 |     0.0773  |                      0.00641 |                      0.05421 |
|      6 | isotonic    |         0.54353 |      0.18521 |         0.7698 |                      0.06476 |                  0.98207 |    0.02309 |    0.02137 |    0.04175 |     0.07291 |                      0.00401 |                      0.05173 |

Archived updated-pregame FPI was run through the same calibration contest.
Identity wins on log loss, so no beta, Platt, or isotonic map is layered onto
FPI. This is an external diagnostic rather than a frozen-preseason validation.

|   rank | method      |   mean_log_loss |   mean_brier |   mean_roc_auc |   mean_ici |   mean_e90 |   mean_emax |
|-------:|:------------|----------------:|-------------:|---------------:|-----------:|-----------:|------------:|
|      1 | identity    |         0.47958 |      0.16009 |        0.83143 |    0.01586 |    0.02795 |     0.04079 |
|      2 | temperature |         0.47962 |      0.16004 |        0.83143 |    0.01578 |    0.02778 |     0.03993 |
|      3 | platt       |         0.47998 |      0.1602  |        0.83143 |    0.01973 |    0.03169 |     0.04327 |
|      4 | intercept   |         0.47998 |      0.16026 |        0.83143 |    0.02023 |    0.0348  |     0.04512 |
|      5 | beta        |         0.48002 |      0.16027 |        0.83143 |    0.01996 |    0.03321 |     0.04454 |
|      6 | isotonic    |         0.48375 |      0.1612  |        0.83013 |    0.02022 |    0.03562 |     0.04838 |

## Updated-FPI benchmark

| model                             |   seasons |   games |   mean_log_loss |   mean_brier |   mean_accuracy |   mean_roc_auc |
|:----------------------------------|----------:|--------:|----------------:|-------------:|----------------:|---------------:|
| nested_updated_fpi_blend          |         8 |    6697 |          0.4794 |       0.16   |          0.7544 |         0.8316 |
| updated_pregame_fpi               |         8 |    6697 |          0.4796 |       0.1601 |          0.7553 |         0.8314 |
| calibrated_frozen_preseason_score |         8 |    6697 |          0.5419 |       0.185  |          0.7041 |         0.7699 |

## Betting-market consensus

For games with a published line, the deployed probability combines the FPI
logit and home-team point spread in a regularized logistic regression. The
market layer uses an expanding window: the 2024 test model trains on 2023, and
the 2025 test model trains on 2023-24. Games without a line fall back to FPI.
Market lines are not frozen-preseason information; this layer is for the daily
live forecast.

Across 1,830 covered 2024-25 games, log
loss improves from `0.5571` for the enhanced
frozen-preseason score model and `0.4864` for FPI
to `0.4747`. Relative to the 50/50 log-loss
baseline, that is a `60.5%` increase
in predictive skill versus the score model. This percentage is skill lift, not
a claim of the same percentage reduction in raw log loss.

| model                           |   seasons |   games |   mean_log_loss |   mean_brier |   mean_accuracy |   mean_roc_auc |
|:--------------------------------|----------:|--------:|----------------:|-------------:|----------------:|---------------:|
| fpi_market_consensus            |         2 |    1830 |          0.4747 |       0.1597 |          0.7551 |         0.833  |
| updated_pregame_fpi             |         2 |    1830 |          0.4864 |       0.164  |          0.7442 |         0.8231 |
| enhanced_frozen_preseason_score |         2 |    1830 |          0.5571 |       0.192  |          0.6906 |         0.7485 |

## Incremental in-season update

The public leaderboard is rebuilt daily from current ESPN results and FPI plus
available market lines. Completed games become fixed 0/1 outcomes; current FPI
updates uncovered future games; the market model updates covered future games.
The complete `20,000`-season simulation is then rerun with a fixed random seed,
so conference-title, CFP, national-title, and total expected value all respond
to the latest season state without introducing avoidable Monte Carlo drift.

## Live draft optimizer

The browser draft room evaluates candidate teams against the full joint
`20,000`-season fantasy-point sample matrix. For each of the 30
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

|   overall_rank | team          | conference    |   expected_fantasy_points |   expected_regular_wins |   expected_playoff_points |   playoff_rank_lift |   playoff_probability |   schedule_difficulty_rank |
|---------------:|:--------------|:--------------|--------------------------:|------------------------:|--------------------------:|--------------------:|----------------------:|---------------------------:|
|              1 | Notre Dame    | FBS Indep.    |                    12.143 |                  10.871 |                     1.271 |                   1 |                 0.912 |                         61 |
|              2 | Texas Tech    | Big 12        |                    11.828 |                  10.464 |                     0.76  |                  -1 |                 0.823 |                         68 |
|              3 | Miami         | ACC           |                    11.505 |                  10.044 |                     0.855 |                   0 |                 0.795 |                         48 |
|              4 | Oregon        | Big Ten       |                    11.44  |                  10.089 |                     1.044 |                   0 |                 0.781 |                         20 |
|              5 | Ohio State    | Big Ten       |                    11.347 |                   9.787 |                     1.201 |                   2 |                 0.741 |                         17 |
|              6 | Indiana       | Big Ten       |                    11.125 |                  10.042 |                     0.855 |                  -1 |                 0.727 |                         35 |
|              7 | Georgia       | SEC           |                    11.11  |                   9.877 |                     0.9   |                  -1 |                 0.7   |                         15 |
|              8 | Texas         | SEC           |                    10.54  |                   9.323 |                     0.908 |                   0 |                 0.596 |                          2 |
|              9 | UNLV          | Mountain West |                     9.225 |                   8.711 |                     0.089 |                   0 |                 0.305 |                        120 |
|             10 | Penn State    | Big Ten       |                     9.222 |                   8.982 |                     0.201 |                   0 |                 0.306 |                         67 |
|             11 | Alabama       | SEC           |                     9.042 |                   8.597 |                     0.345 |                   0 |                 0.331 |                          7 |
|             12 | LSU           | SEC           |                     8.848 |                   8.454 |                     0.309 |                   1 |                 0.288 |                         11 |
|             13 | BYU           | Big 12        |                     8.727 |                   8.397 |                     0.178 |                  -1 |                 0.28  |                         49 |
|             14 | Texas A&M     | SEC           |                     8.594 |                   8.277 |                     0.253 |                   2 |                 0.241 |                         10 |
|             15 | South Florida | American      |                     8.517 |                   8.336 |                     0.038 |                  -1 |                 0.152 |                        115 |
|             16 | Toledo        | MAC           |                     8.467 |                   8.11  |                     0.028 |                  -1 |                 0.14  |                        132 |
|             17 | USC           | Big Ten       |                     8.366 |                   8.172 |                     0.164 |                   3 |                 0.191 |                         26 |
|             18 | SMU           | ACC           |                     8.356 |                   8.15  |                     0.107 |                   0 |                 0.189 |                         56 |
|             19 | James Madison | Sun Belt      |                     8.312 |                   8.01  |                     0.031 |                  -2 |                 0.148 |                        113 |
|             20 | Clemson       | ACC           |                     8.295 |                   8.072 |                     0.132 |                   1 |                 0.195 |                         33 |
|             21 | Tulane        | American      |                     8.288 |                   7.974 |                     0.049 |                  -2 |                 0.166 |                         89 |
|             22 | East Carolina | American      |                     8.098 |                   7.885 |                     0.024 |                   0 |                 0.104 |                         96 |
|             23 | Utah          | Big 12        |                     8.088 |                   7.941 |                     0.076 |                   0 |                 0.154 |                         59 |
|             24 | Michigan      | Big Ten       |                     8.04  |                   7.905 |                     0.114 |                   2 |                 0.136 |                         18 |
|             25 | Hawai'i       | Mountain West |                     7.989 |                   7.743 |                     0.019 |                  -1 |                 0.091 |                        125 |

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
