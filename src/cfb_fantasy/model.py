from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from .calibration import ProbabilityCalibrator
from .features import ELO_COLUMNS, FEATURE_CANDIDATES, FEATURE_COLUMNS


@dataclass(frozen=True)
class Candidate:
    name: str
    columns: list[str]
    factory: Callable[[], ClassifierMixin]


@dataclass
class FittedGameModel:
    name: str
    columns: list[str]
    estimator: ClassifierMixin
    calibrator: ProbabilityCalibrator | None = None

    def predict_home(self, frame: pd.DataFrame) -> np.ndarray:
        probability = self.estimator.predict_proba(frame[self.columns])[:, 1]
        if self.calibrator is not None:
            probability = self.calibrator.predict(probability)
        return probability


def candidates() -> list[Candidate]:
    return [
        Candidate(
            "home_baseline",
            ["home_field"],
            lambda: DummyClassifier(strategy="prior"),
        ),
        Candidate(
            "elo_logit",
            ELO_COLUMNS,
            lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=2.0, max_iter=2000),
            ),
        ),
        Candidate(
            "score_logit",
            FEATURE_COLUMNS,
            lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000),
            ),
        ),
        Candidate(
            "score_boost",
            FEATURE_COLUMNS,
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=8.0,
                random_state=17,
            ),
        ),
    ]


def _metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y, clipped)),
        "accuracy": float(accuracy_score(y, clipped >= 0.5)),
        "roc_auc": float(roc_auc_score(y, clipped)),
    }


def walk_forward_backtest(
    features: pd.DataFrame, first_test_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    seasons = sorted(int(value) for value in features["season"].unique())
    for test_season in [season for season in seasons if season >= first_test_season]:
        train = features[features["season"] < test_season]
        test = features[features["season"] == test_season]
        if train.empty or test.empty or train["target"].nunique() < 2:
            continue
        for candidate in candidates():
            estimator = candidate.factory()
            estimator.fit(train[candidate.columns], train["target"])
            probability = estimator.predict_proba(test[candidate.columns])[:, 1]
            rows.append(
                {
                    "model": candidate.name,
                    "test_season": test_season,
                    "train_games": len(train),
                    "test_games": len(test),
                    **_metrics(test["target"], probability),
                }
            )
    by_season = pd.DataFrame(rows)
    if by_season.empty:
        raise ValueError("backtest produced no evaluation folds")
    aggregate = (
        by_season.groupby("model", as_index=False)
        .agg(
            seasons=("test_season", "count"),
            games=("test_games", "sum"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            worst_log_loss=("log_loss", "max"),
        )
        .sort_values(["mean_log_loss", "mean_brier"])
        .reset_index(drop=True)
    )
    aggregate.insert(0, "rank", np.arange(1, len(aggregate) + 1))
    return by_season, aggregate


def _walk_forward_predictions(
    features: pd.DataFrame,
    columns: list[str],
    first_test_season: int,
) -> tuple[np.ndarray, np.ndarray]:
    target: list[np.ndarray] = []
    probability: list[np.ndarray] = []
    seasons = sorted(int(value) for value in features["season"].unique())
    for test_season in [season for season in seasons if season >= first_test_season]:
        train = features[features["season"] < test_season]
        test = features[features["season"] == test_season]
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000),
        )
        estimator.fit(train[columns], train["target"])
        target.append(test["target"].to_numpy(int))
        probability.append(estimator.predict_proba(test[columns])[:, 1])
    return np.concatenate(target), np.concatenate(probability)


def walk_forward_feature_study(
    features: pd.DataFrame,
    first_test_season: int,
    *,
    random_repeats: int = 100,
    random_seed: int = 20260825,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy OOF feature study with repeated random-feature controls."""
    frame = features.copy()
    rng = np.random.default_rng(random_seed)
    selected: list[str] = []
    remaining = FEATURE_CANDIDATES.copy()
    selection_rows: list[dict[str, object]] = []
    screen_rows: list[dict[str, object]] = []
    baseline_auc = 0.5
    step = 1

    while remaining:
        candidates_at_step = []
        for feature in remaining:
            target, probability = _walk_forward_predictions(
                frame, selected + [feature], first_test_season
            )
            auc = float(roc_auc_score(target, probability))
            candidates_at_step.append((auc, feature))
        best_auc, best_feature = max(candidates_at_step)

        random_deltas = []
        for repeat in range(random_repeats):
            random_column = f"__random_control_{repeat}"
            frame[random_column] = rng.normal(size=len(frame))
            target, probability = _walk_forward_predictions(
                frame, selected + [random_column], first_test_season
            )
            random_deltas.append(float(roc_auc_score(target, probability)) - baseline_auc)
            frame.drop(columns=random_column, inplace=True)
        null_p95 = float(np.quantile(random_deltas, 0.95))
        best_delta = best_auc - baseline_auc
        accepted = bool(best_delta > null_p95)

        for auc, feature in sorted(candidates_at_step, reverse=True):
            screen_rows.append(
                {
                    "step": step,
                    "feature": feature,
                    "auc_with_feature": auc,
                    "delta_auc": auc - baseline_auc,
                    "random_delta_auc_p95": null_p95,
                    "best_at_step": feature == best_feature,
                    "accepted": feature == best_feature and accepted,
                }
            )
        selection_rows.append(
            {
                "step": step,
                "feature": best_feature,
                "baseline_auc": baseline_auc,
                "auc_with_feature": best_auc,
                "delta_auc": best_delta,
                "random_delta_auc_mean": float(np.mean(random_deltas)),
                "random_delta_auc_p95": null_p95,
                "accepted": accepted,
            }
        )
        if not accepted:
            break
        selected.append(best_feature)
        remaining.remove(best_feature)
        baseline_auc = best_auc
        step += 1

    return pd.DataFrame(selection_rows), pd.DataFrame(screen_rows)


def fit_selected_model(
    features: pd.DataFrame,
    factor_study: pd.DataFrame,
    calibrator: ProbabilityCalibrator | None = None,
) -> FittedGameModel:
    selected_name = str(factor_study.iloc[0]["model"])
    candidate = next(item for item in candidates() if item.name == selected_name)
    estimator = candidate.factory()
    estimator.fit(features[candidate.columns], features["target"])
    return FittedGameModel(selected_name, candidate.columns, estimator, calibrator)


def _candidate(name: str) -> Candidate:
    return next(item for item in candidates() if item.name == name)


def _best_blend_weight(frame: pd.DataFrame) -> float:
    weights = np.linspace(0.0, 1.0, 21)
    losses = []
    for weight in weights:
        probability = (
            (1.0 - weight) * frame["score_probability"].to_numpy(float)
            + weight * frame["fpi_home_probability"].to_numpy(float)
        )
        losses.append(_metrics(frame["target"], probability)["log_loss"])
    return float(weights[int(np.argmin(losses))])


def calibrate_fpi_ensemble(
    features: pd.DataFrame,
    factor_study: pd.DataFrame,
    first_test_season: int,
    calibrated_score_oof: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Nested benchmark of frozen score forecasts against updated pregame FPI."""
    selected_name = str(factor_study.iloc[0]["model"])
    candidate = _candidate(selected_name)
    covered = features.dropna(subset=["fpi_home_probability"]).copy()
    if covered.empty:
        raise ValueError("historical pregame FPI coverage is required for calibration")
    prediction_rows = []
    for test_season in sorted(int(value) for value in covered["season"].unique()):
        train = features[features["season"] < test_season]
        test = covered[covered["season"] == test_season]
        if train.empty or test.empty or train["target"].nunique() < 2:
            continue
        estimator = candidate.factory()
        estimator.fit(train[candidate.columns], train["target"])
        fold = test[
            [
                "game_id",
                "season",
                "home_id",
                "away_id",
                "target",
                "fpi_home_probability",
                "fpi_predicted_margin",
            ]
        ].copy()
        fold["score_probability"] = estimator.predict_proba(
            test[candidate.columns]
        )[:, 1]
        prediction_rows.append(fold)
    oof = pd.concat(prediction_rows, ignore_index=True)
    if calibrated_score_oof is not None:
        selected_calibration = calibrated_score_oof[
            calibrated_score_oof["selected"]
        ][["game_id", "calibrated_probability"]].copy()
        selected_calibration["game_id"] = selected_calibration["game_id"].astype(str)
        oof = oof.merge(
            selected_calibration,
            on="game_id",
            how="left",
            validate="one_to_one",
        )
        covered_calibration = oof["calibrated_probability"].notna()
        oof.loc[covered_calibration, "score_probability"] = oof.loc[
            covered_calibration, "calibrated_probability"
        ]
        oof.drop(columns="calibrated_probability", inplace=True)

    backtest_rows = []
    oof["ensemble_probability"] = np.nan
    oof["nested_fpi_weight"] = np.nan
    for test_season in sorted(
        int(value) for value in oof.loc[oof["season"] >= first_test_season, "season"].unique()
    ):
        history = oof[oof["season"] < test_season]
        test = oof[oof["season"] == test_season]
        if history.empty or test.empty:
            continue
        weight = _best_blend_weight(history)
        score_probability = test["score_probability"].to_numpy(float)
        fpi_probability = test["fpi_home_probability"].to_numpy(float)
        blend_probability = (1.0 - weight) * score_probability + weight * fpi_probability
        oof.loc[test.index, "ensemble_probability"] = blend_probability
        oof.loc[test.index, "nested_fpi_weight"] = weight
        for model_name, probability in [
                ("calibrated_frozen_preseason_score", score_probability),
            ("updated_pregame_fpi", fpi_probability),
            ("nested_updated_fpi_blend", blend_probability),
        ]:
            backtest_rows.append(
                {
                    "model": model_name,
                    "test_season": test_season,
                    "games": len(test),
                    "fpi_weight": weight if model_name == "calibrated_blend" else np.nan,
                    **_metrics(test["target"], probability),
                }
            )
    backtest = pd.DataFrame(backtest_rows)
    evaluation = oof[oof["season"] >= first_test_season]
    final_weight = _best_blend_weight(evaluation)

    margin_sample = covered.dropna(subset=["fpi_predicted_margin"])
    margin = margin_sample["fpi_predicted_margin"].to_numpy(float)
    target = margin_sample["target"].to_numpy(int)

    def margin_loss(scale: float) -> float:
        return _metrics(pd.Series(target), expit(margin / scale))["log_loss"]

    scale_result = minimize_scalar(margin_loss, bounds=(3.0, 15.0), method="bounded")
    summary = {
        "fpi_weight": final_weight,
        "historical_fpi_logistic_scale": float(scale_result.x),
        "backtest_games": float(
            backtest.loc[
                backtest["model"] == "nested_updated_fpi_blend", "games"
            ].sum()
        ),
        "blend_mean_log_loss": float(
            backtest.loc[
                backtest["model"] == "nested_updated_fpi_blend", "log_loss"
            ].mean()
        ),
        "score_mean_log_loss": float(
            backtest.loc[
                backtest["model"] == "calibrated_frozen_preseason_score", "log_loss"
            ].mean()
        ),
        "fpi_mean_log_loss": float(
            backtest.loc[
                backtest["model"] == "updated_pregame_fpi", "log_loss"
            ].mean()
        ),
    }
    return backtest, oof, summary


def calibrate_current_ensemble(
    schedule_features: pd.DataFrame,
    fitted: FittedGameModel,
    teams: pd.DataFrame,
    *,
    fpi_home_advantage: float,
) -> dict[str, float]:
    """Fit the current blend to ESPN win totals for teams without postseason load."""
    score_probability = fitted.predict_home(schedule_features)
    fpi = dict(zip(teams["team_id"].astype(str), teams["fpi"].astype(float)))
    team_ids = teams["team_id"].astype(str).tolist()
    team_index = {team_id: position for position, team_id in enumerate(team_ids)}
    scheduled = np.zeros(len(teams), dtype=int)
    for game in schedule_features.itertuples(index=False):
        if str(game.home_id) in team_index:
            scheduled[team_index[str(game.home_id)]] += 1
        if str(game.away_id) in team_index:
            scheduled[team_index[str(game.away_id)]] += 1
    projected_games = (
        teams["fpi_projected_wins"].to_numpy(float)
        + teams["fpi_projected_losses"].to_numpy(float)
    )
    calibration_mask = np.abs(projected_games - scheduled) <= 0.25
    if calibration_mask.sum() < 60:
        raise ValueError("too few current teams without projected postseason games")

    def expected_wins(weight: float, scale: float) -> np.ndarray:
        wins = np.zeros(len(teams), dtype=float)
        for row_number, game in enumerate(schedule_features.itertuples(index=False)):
            home_id, away_id = str(game.home_id), str(game.away_id)
            probability = float(score_probability[row_number])
            if home_id in fpi and away_id in fpi:
                edge = fpi[home_id] - fpi[away_id]
                if not bool(game.neutral_site):
                    edge += fpi_home_advantage
                fpi_probability = float(expit(edge / scale))
                probability = (1.0 - weight) * probability + weight * fpi_probability
            if home_id in team_index:
                wins[team_index[home_id]] += probability
            if away_id in team_index:
                wins[team_index[away_id]] += 1.0 - probability
        return wins

    target = teams["fpi_projected_wins"].to_numpy(float)

    def objective(parameters: np.ndarray) -> float:
        wins = expected_wins(float(parameters[0]), float(parameters[1]))
        return float(np.mean((wins[calibration_mask] - target[calibration_mask]) ** 2))

    result = minimize(
        objective,
        x0=np.array([0.8, 9.0]),
        bounds=[(0.0, 1.0), (3.0, 15.0)],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"current FPI calibration failed: {result.message}")
    return {
        "fpi_weight": float(result.x[0]),
        "fpi_logistic_scale": float(result.x[1]),
        "projected_wins_rmse": float(np.sqrt(result.fun)),
        "projected_wins_calibration_teams": float(calibration_mask.sum()),
    }


def estimate_season_strength_correlation(
    oof: pd.DataFrame, first_test_season: int
) -> float:
    """Estimate within-team residual correlation from frozen preseason forecasts."""
    frame = oof[
        (oof["season"] >= first_test_season) & oof["score_probability"].notna()
    ].copy()
    probability = np.clip(frame["score_probability"].to_numpy(float), 0.01, 0.99)
    residual = (frame["target"].to_numpy(float) - probability) / np.sqrt(
        probability * (1.0 - probability)
    )
    team_residuals = pd.concat(
        [
            pd.DataFrame(
                {
                    "season": frame["season"].to_numpy(),
                    "team_id": frame["home_id"].astype(str).to_numpy(),
                    "residual": residual,
                }
            ),
            pd.DataFrame(
                {
                    "season": frame["season"].to_numpy(),
                    "team_id": frame["away_id"].astype(str).to_numpy(),
                    "residual": -residual,
                }
            ),
        ],
        ignore_index=True,
    )
    grouped = team_residuals.groupby(["season", "team_id"])["residual"]
    moments = grouped.agg(
        games="count",
        residual_sum="sum",
        residual_square_sum=lambda values: float(np.square(values).sum()),
    )
    moments = moments[moments["games"] >= 2]
    pair_products = (
        np.square(moments["residual_sum"]) - moments["residual_square_sum"]
    ) / 2.0
    pair_count = moments["games"] * (moments["games"] - 1) / 2.0
    estimate = float(pair_products.sum() / pair_count.sum())
    return float(np.clip(estimate, 0.0, 0.30))


def attach_probabilities(
    schedule_features: pd.DataFrame, fitted: FittedGameModel
) -> pd.DataFrame:
    frame = schedule_features.copy()
    frame["home_win_probability"] = fitted.predict_home(frame)
    frame["away_win_probability"] = 1.0 - frame["home_win_probability"]
    return frame


def attach_consensus_probabilities(
    schedule_features: pd.DataFrame,
    fitted: FittedGameModel,
    teams: pd.DataFrame,
    *,
    fpi_weight: float,
    fpi_logistic_scale: float,
    fpi_home_advantage: float,
) -> pd.DataFrame:
    if not 0.0 <= fpi_weight <= 1.0:
        raise ValueError("fpi_weight must be between zero and one")
    if fpi_logistic_scale <= 0.0:
        raise ValueError("fpi_logistic_scale must be positive")
    frame = attach_probabilities(schedule_features, fitted).rename(
        columns={
            "home_win_probability": "score_model_home_win_probability",
            "away_win_probability": "score_model_away_win_probability",
        }
    )
    fpi = dict(zip(teams["team_id"].astype(str), teams["fpi"].astype(float)))
    score_probability = frame["score_model_home_win_probability"].to_numpy(float)
    fpi_probability = np.empty(len(frame), dtype=float)
    has_fpi = np.zeros(len(frame), dtype=bool)
    for row_number, game in enumerate(frame.itertuples(index=False)):
        home_id, away_id = str(game.home_id), str(game.away_id)
        if home_id in fpi and away_id in fpi:
            point_edge = fpi[home_id] - fpi[away_id]
            if not bool(game.neutral_site):
                point_edge += fpi_home_advantage
            fpi_probability[row_number] = 1.0 / (
                1.0 + np.exp(-point_edge / fpi_logistic_scale)
            )
            has_fpi[row_number] = True
        else:
            fpi_probability[row_number] = score_probability[row_number]
    effective_weight = fpi_weight * has_fpi.astype(float)
    frame["fpi_home_win_probability"] = fpi_probability
    frame["fpi_weight_used"] = effective_weight
    frame["home_win_probability"] = (
        (1.0 - effective_weight) * score_probability
        + effective_weight * fpi_probability
    )
    frame["away_win_probability"] = 1.0 - frame["home_win_probability"]
    return frame
