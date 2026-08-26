# Additional Factor Research

## Decision rule

Candidate variables must be known before the forecasted season, improve
expanding-window out-of-fold performance, beat the 95th percentile gain from
100 random features, and retain one-sided block-level statistical significance
after Holm correction within the candidate family. Season blocks are used for
the long score backtest and season-week blocks for the two-season market test.
The deployed FPI-plus-market layer is
tested separately because improving a weaker score model is not sufficient to
improve the live forecast.

## Candidate families

The August 2026 study screened 16 deployable variables derived from lagged team
summaries and current preseason recruiting data: EPA/play margin, success rate,
yards/play, available-yards rate, explosiveness, havoc, line yards, red-zone
success, late-down success, recruiting-class size, talent per recruit, and
year-over-year talent and blue-chip changes. The study also examined offensive,
defensive, and total returning production where historical data existed.

## Frozen-score results

The production baseline has pooled expanding-window AUC `0.77103` and log loss
`0.54110`. The 95th-percentile random-feature AUC gain was `0.00009`.

| Candidate | AUC gain | Log-loss gain | Unadjusted p | Holm p | Decision |
|:--|--:|--:|--:|--:|:--|
| Prior EPA/play margin | 0.00060 | 0.00061 | 0.106 | 1.000 | Reject |
| Prior line-yards balance | 0.00058 | 0.00040 | 0.125 | 1.000 | Reject |
| Prior red-zone balance | 0.00058 | 0.00036 | 0.339 | 1.000 | Reject |
| Offensive returning production | 0.00810 | 0.00601 | 0.039 | 0.156 | Reject |

Offensive returning production is the only economically meaningful score-layer
candidate. It does not survive family-wise correction, and the public source
does not publish a 2026 file, so it cannot be used in the current forecast.

## Deployed-model results

The live FPI-plus-market baseline has 2024-25 AUC `0.83307` and log loss
`0.47464` over 1,830 line-covered games. All 20 candidate additions failed Holm
correction. The best two were:

| Candidate | AUC gain | Log-loss gain | Unadjusted p | Holm p | Decision |
|:--|--:|--:|--:|--:|:--|
| Defensive returning production | 0.00083 | 0.00103 | 0.122 | 1.000 | Reject |
| Prior red-zone balance | 0.00088 | 0.00088 | 0.136 | 1.000 | Reject |

The remaining candidates were flat or harmful. No production feature or model
weight changed as a result of this study.

## Deferred factors

Travel distance and time-zone changes have published evidence of affecting
college-football home advantage, but the free historical schedule assets expose
venue city/state rather than stable team and venue coordinates. They remain a
candidate only after a reproducible, season-correct geographic source is added.
Injury data is too sparse for a historical walk-forward study. Returning
production should be retested when a current-season file becomes available.
