# Sources

- Historical schedules/results: https://github.com/sportsdataverse/cfbfastR-data/tree/main/schedules/csv
- Current schedules/team metadata: https://site.api.espn.com/apis/site/v2/sports/football/college-football/
- Current all-team FPI: https://site.web.api.espn.com/apis/fitt/v3/sports/football/college-football/powerindex
- Team talent and blue-chip ratio: https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_team_talent
- Opponent-adjusted team EPA: https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_team_summaries
- Historical and current betting lines: https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_betting
- Optional authenticated API: https://api.collegefootballdata.com/getting-started
- CFP format: https://collegefootballplayoff.com/sports/2024/5/29/12-team-format.aspx
- South and Egros (2020), game forecasting methods: https://doi.org/10.3233/JSA-190314
- Coleman (2025), predictive metamodel: https://doi.org/10.1177/22150218251365223
- Kull, Silva Filho, and Flach (2017), beta calibration: https://proceedings.mlr.press/v54/kull17a.html
- Austin and Steyerberg (2019), ICI/E50/E90/Emax: https://doi.org/10.1002/sim.8281
- Dimitriadis, Gneiting, and Jordan (2021), CORP calibration and score decomposition: https://doi.org/10.1073/pnas.2016191118

The ESPN endpoints are public but unofficial and may change without notice.
Betting-line coverage varies by game and sportsbook. The raw cache makes each
model run reproducible even if an upstream response later changes.
