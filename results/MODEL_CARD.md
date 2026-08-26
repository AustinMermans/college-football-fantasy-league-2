# 2026 College Football Fantasy Draft Report

Generated 2026-08-26T00:49:07.802970+00:00 from 892 known
regular-season games and 20,000 season simulations.

## Scoring used

One point per regular-season, conference-championship, or CFP win. Losses and
ties score zero. Non-CFP bowl wins are excluded.

## Selected game model

`score_logit` was selected by expanding-window preseason mean log loss
(`0.5490`). Lower log loss is better because the draft
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
`9.68`-point logistic scale minimize
RMSE against current ESPN projected wins for
`108` teams whose FPI
totals do not imply postseason games. The resulting win-total RMSE is
`0.106`. Both component probabilities
remain in the game-level CSV.

## Factor study

|   rank | model         |   seasons |   games |   mean_log_loss |   mean_brier |   mean_accuracy |   mean_roc_auc |   worst_log_loss |
|-------:|:--------------|----------:|--------:|----------------:|-------------:|----------------:|---------------:|-----------------:|
|      1 | score_logit   |         8 |    6701 |          0.549  |       0.1878 |          0.7022 |         0.7633 |           0.5865 |
|      2 | score_boost   |         8 |    6701 |          0.5587 |       0.1917 |          0.6931 |         0.7515 |           0.5933 |
|      3 | elo_logit     |         8 |    6701 |          0.5864 |       0.2009 |          0.6885 |         0.7183 |           0.6144 |
|      4 | home_baseline |         8 |    6701 |          0.6594 |       0.2333 |          0.6294 |         0.5    |           0.68   |

## Marginal feature study

Features are added greedily using pooled expanding-window out-of-fold ROC AUC.
At every step, the selected feature must beat the 95th percentile gain from 100
independent random control features. The first rejected row is the stopping
point; unselected candidates at that step are retained in `feature_screen.csv`.

|   step | feature              |   baseline_auc |   auc_with_feature |   delta_auc |   random_delta_auc_mean |   random_delta_auc_p95 | accepted   |
|-------:|:---------------------|---------------:|-------------------:|------------:|------------------------:|-----------------------:|:-----------|
|      1 | long_margin_diff     |        0.5     |            0.72143 |     0.22143 |                -0.00383 |                0.00745 | True       |
|      2 | fbs_status_diff      |        0.72143 |            0.74957 |     0.02814 |                -8e-05   |                0.0002  | True       |
|      3 | rating_diff          |        0.74957 |            0.75346 |     0.00389 |                -8e-05   |                0.00015 | True       |
|      4 | long_win_diff        |        0.75346 |            0.75829 |     0.00483 |                -7e-05   |                0.00015 | True       |
|      5 | rating_momentum_diff |        0.75829 |            0.76316 |     0.00487 |                -6e-05   |                0.00025 | True       |
|      6 | home_field           |        0.76316 |            0.76374 |     0.00057 |                -5e-05   |                0.00015 | True       |
|      7 | win_form_diff        |        0.76374 |            0.76398 |     0.00024 |                -4e-05   |                0.00018 | True       |
|      8 | margin_form_diff     |        0.76398 |            0.76428 |     0.0003  |                -7e-05   |                0.00015 | True       |
|      9 | history_depth_diff   |        0.76428 |            0.76276 |    -0.00152 |                -7e-05   |                0.0002  | False      |

## Probability calibration study

Lower log loss, Brier, ICI, E50, E90, Emax, and Murphy miscalibration are better;
higher AUC and Murphy discrimination are better. A perfect calibration
intercept and slope are 0 and 1. ICI-family metrics use a local-linear smooth
calibration curve with span 0.75. The Murphy columns use the exact isotonic CORP
Brier decomposition.

|   rank | method      |   mean_log_loss |   mean_brier |   mean_roc_auc |   mean_calibration_intercept |   mean_calibration_slope |   mean_ici |   mean_e50 |   mean_e90 |   mean_emax |   mean_murphy_miscalibration |   mean_murphy_discrimination |
|-------:|:------------|----------------:|-------------:|---------------:|-----------------------------:|-------------------------:|-----------:|-----------:|-----------:|------------:|-----------------------------:|-----------------------------:|
|      1 | beta        |         0.5487  |      0.18767 |        0.76335 |                      0.04672 |                  1.00718 |    0.02387 |    0.02041 |    0.04654 |     0.07195 |                      0.00657 |                      0.05184 |
|      2 | identity    |         0.54898 |      0.18776 |        0.76335 |                      0.05508 |                  0.97305 |    0.02373 |    0.01953 |    0.04928 |     0.07575 |                      0.00667 |                      0.05184 |
|      3 | intercept   |         0.54907 |      0.18779 |        0.76335 |                      0.05718 |                  0.97306 |    0.02416 |    0.02    |    0.04941 |     0.0759  |                      0.0067  |                      0.05184 |
|      4 | platt       |         0.54914 |      0.18772 |        0.76335 |                      0.04552 |                  1.0058  |    0.02436 |    0.0211  |    0.04812 |     0.07428 |                      0.00662 |                      0.05184 |
|      5 | temperature |         0.54921 |      0.18775 |        0.76335 |                      0.05509 |                  1.00253 |    0.02495 |    0.02153 |    0.04977 |     0.07561 |                      0.00666 |                      0.05184 |
|      6 | isotonic    |         0.55239 |      0.18849 |        0.76207 |                      0.06421 |                  0.95162 |    0.02389 |    0.02123 |    0.04534 |     0.08584 |                      0.00485 |                      0.04929 |

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
| nested_updated_fpi_blend          |         8 |    6697 |          0.4796 |       0.16   |          0.7557 |         0.8315 |
| updated_pregame_fpi               |         8 |    6697 |          0.4796 |       0.1601 |          0.7553 |         0.8314 |
| calibrated_frozen_preseason_score |         8 |    6697 |          0.5487 |       0.1877 |          0.7019 |         0.7632 |

## Live draft optimizer

The browser draft room evaluates candidate teams against the full joint
`20,000`-season fantasy-point sample matrix. For each of the 30 highest-ranked
candidates expected to be available at our next snake pick, it completes all 12
rosters in board order and reports fractional first-place probability, expected
finish, expected roster points, and expected margin. The default recommendation
maximizes first-place probability; expected points is an optional alternative
objective.

Head-to-head games, conference-title competition, and CFP bracket collisions
are already represented in the joint samples. Same-conference count and average
pair correlation are therefore diagnostics, not a second arbitrary penalty.
Before our turn, intervening opponent picks are projected in board order and are
replaced as actual selections are recorded.

## Top 25 draft board

|   overall_rank | team          | conference    |   expected_fantasy_points |   expected_regular_wins |   expected_playoff_points |   playoff_rank_lift |   playoff_probability |   schedule_difficulty_rank |
|---------------:|:--------------|:--------------|--------------------------:|------------------------:|--------------------------:|--------------------:|----------------------:|---------------------------:|
|              1 | Notre Dame    | FBS Indep.    |                    11.916 |                  10.676 |                     1.24  |                   1 |                 0.875 |                         61 |
|              2 | Texas Tech    | Big 12        |                    11.665 |                  10.341 |                     0.747 |                  -1 |                 0.793 |                         68 |
|              3 | Ohio State    | Big Ten       |                    11.338 |                   9.777 |                     1.194 |                   2 |                 0.742 |                         17 |
|              4 | Miami         | ACC           |                    11.283 |                   9.903 |                     0.812 |                  -1 |                 0.752 |                         48 |
|              5 | Oregon        | Big Ten       |                    11.237 |                   9.946 |                     0.997 |                  -1 |                 0.744 |                         20 |
|              6 | Indiana       | Big Ten       |                    10.911 |                   9.886 |                     0.812 |                   0 |                 0.68  |                         35 |
|              7 | Georgia       | SEC           |                    10.883 |                   9.719 |                     0.856 |                   0 |                 0.66  |                         15 |
|              8 | Texas         | SEC           |                    10.688 |                   9.407 |                     0.952 |                   0 |                 0.624 |                          2 |
|              9 | Penn State    | Big Ten       |                     9.176 |                   8.926 |                     0.209 |                   1 |                 0.303 |                         67 |
|             10 | UNLV          | Mountain West |                     9.166 |                   8.658 |                     0.089 |                  -1 |                 0.297 |                        120 |
|             11 | Alabama       | SEC           |                     8.949 |                   8.512 |                     0.336 |                   1 |                 0.321 |                          7 |
|             12 | BYU           | Big 12        |                     8.877 |                   8.505 |                     0.205 |                  -1 |                 0.309 |                         49 |
|             13 | LSU           | SEC           |                     8.748 |                   8.364 |                     0.297 |                   1 |                 0.278 |                         11 |
|             14 | Texas A&M     | SEC           |                     8.572 |                   8.243 |                     0.262 |                   3 |                 0.242 |                         10 |
|             15 | Toledo        | MAC           |                     8.555 |                   8.198 |                     0.034 |                  -2 |                 0.158 |                        132 |
|             16 | South Florida | American      |                     8.479 |                   8.298 |                     0.038 |                  -1 |                 0.15  |                        115 |
|             17 | USC           | Big Ten       |                     8.457 |                   8.226 |                     0.195 |                   2 |                 0.212 |                         26 |
|             18 | SMU           | ACC           |                     8.413 |                   8.187 |                     0.12  |                   0 |                 0.201 |                         56 |
|             19 | Tulane        | American      |                     8.388 |                   8.065 |                     0.058 |                  -3 |                 0.181 |                         89 |
|             20 | Clemson       | ACC           |                     8.367 |                   8.117 |                     0.149 |                   1 |                 0.212 |                         33 |
|             21 | James Madison | Sun Belt      |                     8.288 |                   7.993 |                     0.031 |                  -1 |                 0.145 |                        113 |
|             22 | East Carolina | American      |                     8.105 |                   7.892 |                     0.027 |                   0 |                 0.108 |                         96 |
|             23 | Navy          | American      |                     8.037 |                   7.827 |                     0.028 |                   0 |                 0.111 |                         88 |
|             24 | Hawai'i       | Mountain West |                     8.013 |                   7.763 |                     0.023 |                   0 |                 0.097 |                        125 |
|             25 | Michigan      | Big Ten       |                     7.949 |                   7.806 |                     0.117 |                   1 |                 0.134 |                         18 |

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
