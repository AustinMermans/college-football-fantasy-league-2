from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.base import ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


CALIBRATION_METHODS = [
    "identity",
    "intercept",
    "temperature",
    "platt",
    "beta",
    "isotonic",
]


def _clip(probability: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)


@dataclass
class ProbabilityCalibrator:
    method: str
    parameters: dict[str, float] = field(default_factory=dict)
    isotonic: IsotonicRegression | None = None

    def predict(self, probability: np.ndarray | pd.Series) -> np.ndarray:
        probability = _clip(probability)
        score = logit(probability)
        if self.method == "identity":
            calibrated = probability
        elif self.method == "intercept":
            calibrated = expit(score + self.parameters["intercept"])
        elif self.method == "temperature":
            calibrated = expit(self.parameters["slope"] * score)
        elif self.method == "platt":
            calibrated = expit(
                self.parameters["slope"] * score + self.parameters["intercept"]
            )
        elif self.method == "beta":
            calibrated = expit(
                self.parameters["a"] * np.log(probability)
                - self.parameters["b"] * np.log1p(-probability)
                + self.parameters["intercept"]
            )
        elif self.method == "isotonic" and self.isotonic is not None:
            calibrated = self.isotonic.predict(probability)
        else:
            raise ValueError(f"unsupported calibration method: {self.method}")
        return _clip(calibrated)

    def summary(self) -> dict[str, float | str]:
        summary: dict[str, float | str] = {"method": self.method}
        summary.update(self.parameters)
        if self.isotonic is not None:
            summary["isotonic_thresholds"] = float(len(self.isotonic.X_thresholds_))
        return summary


def fit_probability_calibrator(
    method: str,
    probability: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
) -> ProbabilityCalibrator:
    probability = _clip(probability)
    target = np.asarray(target, dtype=int)
    score = logit(probability)
    if method == "identity":
        return ProbabilityCalibrator(method)
    if method == "intercept":
        result = minimize_scalar(
            lambda intercept: log_loss(target, expit(score + intercept)),
            bounds=(-5.0, 5.0),
            method="bounded",
        )
        return ProbabilityCalibrator(method, {"intercept": float(result.x)})
    if method == "temperature":
        result = minimize_scalar(
            lambda slope: log_loss(target, expit(slope * score)),
            bounds=(0.05, 5.0),
            method="bounded",
        )
        return ProbabilityCalibrator(method, {"slope": float(result.x)})
    if method == "platt":
        result = minimize(
            lambda value: log_loss(target, expit(value[0] * score + value[1])),
            x0=np.array([1.0, 0.0]),
            bounds=[(0.05, 5.0), (-5.0, 5.0)],
            method="L-BFGS-B",
        )
        return ProbabilityCalibrator(
            method,
            {"slope": float(result.x[0]), "intercept": float(result.x[1])},
        )
    if method == "beta":
        design = np.column_stack([np.log(probability), -np.log1p(-probability)])
        result = minimize(
            lambda value: log_loss(
                target, expit(design @ value[:2] + value[2])
            ),
            x0=np.array([1.0, 1.0, 0.0]),
            bounds=[(0.001, 10.0), (0.001, 10.0), (-10.0, 10.0)],
            method="L-BFGS-B",
        )
        return ProbabilityCalibrator(
            method,
            {
                "a": float(result.x[0]),
                "b": float(result.x[1]),
                "intercept": float(result.x[2]),
            },
        )
    if method == "isotonic":
        estimator = IsotonicRegression(
            y_min=0.001,
            y_max=0.999,
            increasing=True,
            out_of_bounds="clip",
        ).fit(probability, target)
        return ProbabilityCalibrator(method, isotonic=estimator)
    raise ValueError(f"unsupported calibration method: {method}")


def _local_linear_calibration_curve(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    span: float = 0.75,
) -> np.ndarray:
    """LOWESS-style local-linear observed probabilities at each forecast."""
    n = len(probability)
    neighbor_count = min(n, max(20, int(np.ceil(span * n))))
    observed = np.empty(n, dtype=float)
    for row, center in enumerate(probability):
        distance = np.abs(probability - center)
        bandwidth = float(np.partition(distance, neighbor_count - 1)[neighbor_count - 1])
        if bandwidth <= 1e-12:
            observed[row] = float(target[distance <= 1e-12].mean())
            continue
        scaled = np.clip(distance / bandwidth, 0.0, 1.0)
        weight = np.power(1.0 - np.power(scaled, 3.0), 3.0)
        centered = probability - center
        sum_weight = float(weight.sum())
        sum_x = float(np.sum(weight * centered))
        sum_xx = float(np.sum(weight * centered * centered))
        sum_y = float(np.sum(weight * target))
        sum_xy = float(np.sum(weight * centered * target))
        denominator = sum_weight * sum_xx - sum_x * sum_x
        if abs(denominator) <= 1e-12:
            observed[row] = sum_y / sum_weight
        else:
            observed[row] = (sum_y * sum_xx - sum_xy * sum_x) / denominator
    return np.clip(observed, 0.0, 1.0)


def _calibration_intercept_slope(
    probability: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    score = logit(_clip(probability))
    result = minimize(
        lambda value: log_loss(target, expit(value[0] + value[1] * score)),
        x0=np.array([0.0, 1.0]),
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        method="L-BFGS-B",
    )
    return float(result.x[0]), float(result.x[1])


def _ece(probability: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    frame = pd.DataFrame({"probability": probability, "target": target})
    frame["bin"] = pd.qcut(frame["probability"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        games=("target", "size"),
        predicted=("probability", "mean"),
        observed=("target", "mean"),
    )
    return float(
        np.sum(
            grouped["games"]
            / len(frame)
            * np.abs(grouped["predicted"] - grouped["observed"])
        )
    )


def calibration_metrics(
    target: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    probability = _clip(probability)
    smooth_observed = _local_linear_calibration_curve(probability, target)
    absolute_error = np.abs(smooth_observed - probability)
    intercept, slope = _calibration_intercept_slope(probability, target)

    isotonic = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    ).fit_transform(probability, target)
    brier = float(brier_score_loss(target, probability))
    calibrated_brier = float(brier_score_loss(target, isotonic))
    uncertainty = float(np.mean(np.square(target - target.mean())))
    miscalibration = brier - calibrated_brier
    discrimination = uncertainty - calibrated_brier

    return {
        "log_loss": float(log_loss(target, probability)),
        "brier": brier,
        "roc_auc": float(roc_auc_score(target, probability)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ici": float(absolute_error.mean()),
        "e50": float(np.quantile(absolute_error, 0.50)),
        "e90": float(np.quantile(absolute_error, 0.90)),
        "emax": float(absolute_error.max()),
        "ece_15": _ece(probability, target),
        "murphy_miscalibration": miscalibration,
        "murphy_discrimination": discrimination,
        "murphy_uncertainty": uncertainty,
        "murphy_error": brier - (miscalibration - discrimination + uncertainty),
    }


def _aggregate_calibration_metrics(by_season: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in by_season.columns
        if column not in {"method", "test_season", "calibration_games", "test_games"}
    ]
    aggregate = by_season.groupby("method", as_index=False).agg(
        seasons=("test_season", "count"),
        games=("test_games", "sum"),
        **{f"mean_{column}": (column, "mean") for column in metric_columns},
    )
    aggregate = aggregate.sort_values(
        ["mean_log_loss", "mean_brier", "mean_ici"]
    ).reset_index(drop=True)
    aggregate.insert(0, "rank", np.arange(1, len(aggregate) + 1))
    return aggregate


def walk_forward_probability_calibration_backtest(
    frame: pd.DataFrame,
    probability_column: str,
    first_test_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate an existing historical probability using only earlier seasons."""
    covered = frame.dropna(subset=[probability_column]).copy()
    rows: list[dict[str, float | int | str]] = []
    seasons = sorted(int(value) for value in covered["season"].unique())
    for test_season in [season for season in seasons if season >= first_test_season]:
        history = covered[covered["season"] < test_season]
        test = covered[covered["season"] == test_season]
        for method in CALIBRATION_METHODS:
            calibrator = fit_probability_calibrator(
                method, history[probability_column], history["target"]
            )
            probability = calibrator.predict(test[probability_column])
            rows.append(
                {
                    "method": method,
                    "test_season": test_season,
                    "calibration_games": len(history),
                    "test_games": len(test),
                    **calibration_metrics(test["target"], probability),
                }
            )
    by_season = pd.DataFrame(rows)
    return by_season, _aggregate_calibration_metrics(by_season)


def walk_forward_calibration_backtest(
    features: pd.DataFrame,
    columns: list[str],
    estimator_factory: Callable[[], ClassifierMixin],
    first_test_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ProbabilityCalibrator]:
    """Nested expanding-window calibration using only prior OOF predictions."""
    seasons = sorted(int(value) for value in features["season"].unique())
    calibration_start = max(seasons[0] + 3, first_test_season - 5)
    raw_rows: list[pd.DataFrame] = []
    for test_season in [season for season in seasons if season >= calibration_start]:
        train = features[features["season"] < test_season]
        test = features[features["season"] == test_season]
        estimator = estimator_factory()
        estimator.fit(train[columns], train["target"])
        raw_rows.append(
            pd.DataFrame(
                {
                    "game_id": test["game_id"].astype(str).to_numpy(),
                    "season": test_season,
                    "target": test["target"].to_numpy(int),
                    "raw_probability": estimator.predict_proba(test[columns])[:, 1],
                }
            )
        )
    raw_oof = pd.concat(raw_rows, ignore_index=True)

    metric_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    for test_season in [season for season in seasons if season >= first_test_season]:
        history = raw_oof[raw_oof["season"] < test_season]
        test = raw_oof[raw_oof["season"] == test_season]
        for method in CALIBRATION_METHODS:
            calibrator = fit_probability_calibrator(
                method, history["raw_probability"], history["target"]
            )
            probability = calibrator.predict(test["raw_probability"])
            metric_rows.append(
                {
                    "method": method,
                    "test_season": test_season,
                    "calibration_games": len(history),
                    "test_games": len(test),
                    **calibration_metrics(test["target"], probability),
                }
            )
            prediction_rows.append(
                test.assign(method=method, calibrated_probability=probability)
            )

    by_season = pd.DataFrame(metric_rows)
    aggregate = _aggregate_calibration_metrics(by_season)
    selected_method = str(aggregate.iloc[0]["method"])
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions["selected"] = predictions["method"].eq(selected_method)
    final_calibrator = fit_probability_calibrator(
        selected_method, raw_oof["raw_probability"], raw_oof["target"]
    )
    return by_season, aggregate, predictions, final_calibrator
