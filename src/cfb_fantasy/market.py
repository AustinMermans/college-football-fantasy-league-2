from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


MARKET_TRAINING_START = 2023
MARKET_FIRST_TEST_SEASON = 2024


def update_fpi_probabilities(
    games: pd.DataFrame,
    fpi: pd.DataFrame,
    *,
    logistic_scale: float,
    home_advantage: float,
) -> pd.DataFrame:
    """Refresh remaining-game probabilities from the latest team FPI values."""
    if logistic_scale <= 0.0:
        raise ValueError("logistic_scale must be positive")
    frame = games.copy()
    rating = dict(zip(fpi["team_id"].astype(str), fpi["fpi"].astype(float)))
    home_rating = frame["home_id"].astype(str).map(rating)
    away_rating = frame["away_id"].astype(str).map(rating)
    covered = home_rating.notna() & away_rating.notna()
    venue = (~frame["neutral_site"].astype(bool)).astype(float) * home_advantage
    probability = expit((home_rating - away_rating + venue) / logistic_scale)
    frame.loc[covered, "fpi_home_win_probability"] = probability[covered]
    frame.loc[covered, "fpi_home_probability"] = probability[covered]
    frame.loc[covered, "home_win_probability"] = probability[covered]
    frame.loc[covered, "away_win_probability"] = 1.0 - probability[covered]
    frame["current_fpi_used"] = covered
    frame["probability_source"] = np.where(covered, "current_fpi", "baseline")
    return frame


def _metrics(target: pd.Series, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(target, probability)),
        "accuracy": float(accuracy_score(target, probability >= 0.5)),
        "roc_auc": float(roc_auc_score(target, probability)),
    }


def _market_frame(games: pd.DataFrame, betting: pd.DataFrame) -> pd.DataFrame:
    columns = ["game_id", "home_team_spread", "over_under", "odds_source"]
    available = [column for column in columns if column in betting]
    lines = betting[available].copy()
    lines["game_id"] = lines["game_id"].astype(str)
    frame = games.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame = frame.drop(
        columns=[column for column in available if column != "game_id" and column in frame],
        errors="ignore",
    )
    frame = frame.merge(lines, on="game_id", how="left", validate="one_to_one")
    if "fpi_home_probability" not in frame:
        frame["fpi_home_probability"] = frame["fpi_home_win_probability"]
    frame["fpi_logit"] = logit(
        frame["fpi_home_probability"].astype(float).clip(0.001, 0.999)
    )
    frame["spread_edge"] = -pd.to_numeric(frame["home_team_spread"], errors="coerce")
    return frame


@dataclass(frozen=True)
class MarketConsensus:
    mean: tuple[float, float]
    scale: tuple[float, float]
    coefficients: tuple[float, float]
    intercept: float
    training_games: int

    def predict(self, fpi_probability: np.ndarray, home_team_spread: np.ndarray) -> np.ndarray:
        design = np.column_stack(
            [
                logit(np.clip(np.asarray(fpi_probability, dtype=float), 0.001, 0.999)),
                -np.asarray(home_team_spread, dtype=float),
            ]
        )
        standardized = (design - np.asarray(self.mean)) / np.asarray(self.scale)
        return expit(standardized @ np.asarray(self.coefficients) + self.intercept)

    def summary(self) -> dict[str, Any]:
        return {
            "features": ["fpi_logit", "spread_edge"],
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "training_games": self.training_games,
            "training_start_season": MARKET_TRAINING_START,
        }

    @classmethod
    def from_summary(cls, value: dict[str, Any]) -> MarketConsensus:
        return cls(
            mean=tuple(float(item) for item in value["mean"]),
            scale=tuple(float(item) for item in value["scale"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            intercept=float(value["intercept"]),
            training_games=int(value["training_games"]),
        )


def fit_market_consensus(frame: pd.DataFrame) -> MarketConsensus:
    covered = frame.dropna(subset=["fpi_home_probability", "home_team_spread"])
    if len(covered) < 500:
        raise ValueError("market consensus needs at least 500 covered games")
    design = np.column_stack(
        [
            logit(covered["fpi_home_probability"].astype(float).clip(0.001, 0.999)),
            -covered["home_team_spread"].to_numpy(float),
        ]
    )
    scaler = StandardScaler().fit(design)
    estimator = LogisticRegression(C=0.1, max_iter=3000).fit(
        scaler.transform(design), covered["target"]
    )
    return MarketConsensus(
        mean=tuple(float(item) for item in scaler.mean_),
        scale=tuple(float(item) for item in scaler.scale_),
        coefficients=tuple(float(item) for item in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
        training_games=len(covered),
    )


def walk_forward_market_backtest(
    features: pd.DataFrame,
    betting: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, MarketConsensus, dict[str, float]]:
    frame = _market_frame(features, betting)
    rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    for test_season in sorted(
        int(value)
        for value in frame.loc[
            frame["season"] >= MARKET_FIRST_TEST_SEASON, "season"
        ].unique()
    ):
        history = frame[
            (frame["season"] >= MARKET_TRAINING_START)
            & (frame["season"] < test_season)
        ]
        test = frame[
            (frame["season"] == test_season)
            & frame["fpi_home_probability"].notna()
            & frame["home_team_spread"].notna()
        ].copy()
        if len(history.dropna(subset=["fpi_home_probability", "home_team_spread"])) < 500:
            continue
        market = fit_market_consensus(history)
        market_probability = market.predict(
            test["fpi_home_probability"], test["home_team_spread"]
        )
        score_train = features[features["season"] < test_season]
        score_test = features.set_index("game_id").loc[test["game_id"]]
        score_estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000),
        ).fit(score_train[FEATURE_COLUMNS], score_train["target"])
        score_probability = score_estimator.predict_proba(
            score_test[FEATURE_COLUMNS]
        )[:, 1]
        models = [
            ("enhanced_frozen_preseason_score", score_probability),
            ("updated_pregame_fpi", test["fpi_home_probability"].to_numpy(float)),
            ("fpi_market_consensus", market_probability),
        ]
        for name, probability in models:
            rows.append(
                {
                    "model": name,
                    "test_season": test_season,
                    "games": len(test),
                    **_metrics(test["target"], probability),
                }
            )
        prediction_rows.append(
            test[["game_id", "season", "target", "home_team_spread"]].assign(
                fpi_probability=test["fpi_home_probability"].to_numpy(float),
                score_probability=score_probability,
                market_consensus_probability=market_probability,
            )
        )
    backtest = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    aggregate = backtest.groupby("model")["log_loss"].mean()
    naive_loss = float(np.log(2.0))
    score_skill = 1.0 - float(aggregate["enhanced_frozen_preseason_score"]) / naive_loss
    market_skill = 1.0 - float(aggregate["fpi_market_consensus"]) / naive_loss
    summary = {
        "evaluation_games": float(
            backtest.loc[backtest["model"] == "fpi_market_consensus", "games"].sum()
        ),
        "score_log_loss": float(aggregate["enhanced_frozen_preseason_score"]),
        "fpi_log_loss": float(aggregate["updated_pregame_fpi"]),
        "market_log_loss": float(aggregate["fpi_market_consensus"]),
        "score_predictive_skill": score_skill,
        "market_predictive_skill": market_skill,
        "skill_improvement_vs_score": market_skill / score_skill - 1.0,
    }
    final_training = frame[frame["season"] >= MARKET_TRAINING_START]
    fitted = fit_market_consensus(final_training)
    return backtest, predictions, fitted, summary


def apply_market_consensus(
    games: pd.DataFrame,
    betting: pd.DataFrame,
    fitted: MarketConsensus,
) -> pd.DataFrame:
    frame = _market_frame(games, betting)
    covered = frame["fpi_home_probability"].notna() & frame["home_team_spread"].notna()
    frame["market_home_win_probability"] = np.nan
    frame.loc[covered, "market_home_win_probability"] = fitted.predict(
        frame.loc[covered, "fpi_home_probability"],
        frame.loc[covered, "home_team_spread"],
    )
    frame["market_line_used"] = covered
    frame["base_home_win_probability"] = frame["home_win_probability"]
    frame.loc[covered, "home_win_probability"] = frame.loc[
        covered, "market_home_win_probability"
    ]
    frame["away_win_probability"] = 1.0 - frame["home_win_probability"]
    frame["probability_source"] = np.where(covered, "fpi_market", "fpi_fallback")
    return frame
