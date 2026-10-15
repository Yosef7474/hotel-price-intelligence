"""Hotel-grouped evaluation utilities for synthetic price modeling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


MIN_ELIGIBLE_HOTELS = 50
TEST_GROUP_FRACTION = 0.20
MIN_TEST_HOTELS = 10
NUMERIC_FEATURES = ("star_rating", "guest_rating", "room_size_m2", "month", "distance_center_km")
CATEGORICAL_FEATURES = ("city", "room_type")
ALLOWED_FEATURES = ("city", "star_rating", "guest_rating", "room_size_m2", "room_type", "month", "distance_center_km")


@dataclass(frozen=True)
class HotelSplit:
    """A deterministic positional row split with disjoint hotel assignments."""

    dataset_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_hotel_ids: tuple[Any, ...]
    test_hotel_ids: tuple[Any, ...]
    eligible_hotel_count: int
    status: str
    unavailable_reason: str | None = None


@dataclass
class EvaluationResult:
    """Held-out metrics, predictions, and fitted candidate estimators."""

    dataset_id: str
    include_outliers: bool
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    models: dict[str, Any]
    train_hotel_ids: tuple[Any, ...]
    test_hotel_ids: tuple[Any, ...]
    unavailable_reason: str | None = None


METRIC_COLUMNS = [
    "model", "mae_eur", "rmse_eur", "r2", "train_rows", "test_rows",
    "train_hotel_count", "test_hotel_count", "dataset_id", "include_outliers",
]
PREDICTION_COLUMNS = ["model", "row_position", "hotel_id", "offer_id", "actual_price_eur", "predicted_price_eur"]


def _valid_price_mask(data: pd.DataFrame) -> pd.Series:
    if "price_eur" not in data.columns:
        raise ValueError("Hotel evaluation requires the 'price_eur' target column.")
    prices = pd.to_numeric(data["price_eur"], errors="coerce")
    return np.isfinite(prices) & prices.gt(0)


def make_hotel_split(data: pd.DataFrame, dataset_id: str, random_state: int = 42) -> HotelSplit:
    """Split eligible hotels 80/20, keeping every hotel wholly on one side.

    Indices are positional offsets for use with ``DataFrame.iloc``. Evaluation
    requires at least 50 hotels with a nonmissing ID and a valid positive price.
    """
    if "hotel_id" not in data.columns:
        raise ValueError("Hotel-group evaluation requires the 'hotel_id' column.")
    valid = data["hotel_id"].notna() & _valid_price_mask(data)
    eligible_positions = np.flatnonzero(valid.to_numpy())
    eligible_groups = data.iloc[eligible_positions]["hotel_id"]
    hotel_ids = sorted(pd.unique(eligible_groups).tolist(), key=lambda value: str(value))
    hotel_count = len(hotel_ids)
    if hotel_count < MIN_ELIGIBLE_HOTELS:
        reason = (
            f"Evaluation needs at least {MIN_ELIGIBLE_HOTELS} distinct hotels with valid prices; "
            f"the current data has {hotel_count}."
        )
        return HotelSplit(
            dataset_id=str(dataset_id), train_indices=(), test_indices=(),
            train_hotel_ids=(), test_hotel_ids=(), eligible_hotel_count=hotel_count,
            status="unavailable", unavailable_reason=reason,
        )

    rng = np.random.default_rng(random_state)
    permutation = rng.permutation(hotel_count)
    test_count = max(MIN_TEST_HOTELS, math.ceil(hotel_count * TEST_GROUP_FRACTION))
    test_count = min(test_count, hotel_count - 1)
    test_set = {hotel_ids[int(index)] for index in permutation[:test_count]}
    train_set = set(hotel_ids) - test_set
    positions_by_group = data.iloc[eligible_positions]["hotel_id"].to_numpy()
    train_positions = tuple(int(position) for position in eligible_positions[np.isin(positions_by_group, list(train_set))])
    test_positions = tuple(int(position) for position in eligible_positions[np.isin(positions_by_group, list(test_set))])
    return HotelSplit(
        dataset_id=str(dataset_id),
        train_indices=train_positions,
        test_indices=test_positions,
        train_hotel_ids=tuple(group for group in hotel_ids if group in train_set),
        test_hotel_ids=tuple(group for group in hotel_ids if group in test_set),
        eligible_hotel_count=hotel_count,
        status="ready",
    )


def _feature_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in ALLOWED_FEATURES if column in data.columns]


def build_candidate_pipelines(feature_columns: list[str] | None = None) -> dict[str, Any]:
    """Build fresh median, linear-regression, and random-forest estimators.

    Preprocessing is part of each sklearn pipeline and therefore learns only
    from the training rows passed to ``fit``. Unknown categories are ignored.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required for price evaluation. Install the project requirements with: "
            "python -m pip install -r requirements.txt"
        ) from error

    selected_features = list(feature_columns or ALLOWED_FEATURES)
    disallowed = sorted(set(selected_features) - set(ALLOWED_FEATURES))
    if disallowed:
        raise ValueError(f"Predictive inputs cannot include: {', '.join(disallowed)}.")
    numeric = [column for column in selected_features if column in NUMERIC_FEATURES]
    categorical = [column for column in selected_features if column in CATEGORICAL_FEATURES]
    if not numeric and not categorical:
        raise ValueError("At least one supported hotel characteristic is needed for regression candidates.")

    def preprocessor() -> ColumnTransformer:
        transformers = []
        if numeric:
            transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
        if categorical:
            categorical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, categorical))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    return {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear regression": Pipeline(
            [("preprocessor", preprocessor()), ("regressor", LinearRegression())]
        ),
        "Random forest": Pipeline(
            [
                ("preprocessor", preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_features=1.0,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def _empty_evaluation(split: HotelSplit, include_outliers: bool, reason: str) -> EvaluationResult:
    return EvaluationResult(
        dataset_id=split.dataset_id,
        include_outliers=include_outliers,
        metrics=pd.DataFrame(columns=METRIC_COLUMNS),
        predictions=pd.DataFrame(columns=PREDICTION_COLUMNS),
        models={},
        train_hotel_ids=split.train_hotel_ids,
        test_hotel_ids=split.test_hotel_ids,
        unavailable_reason=reason,
    )


def evaluate_candidates(data: pd.DataFrame, split: HotelSplit, include_outliers: bool) -> EvaluationResult:
    """Fit candidates on training hotels and score identical held-out rows."""
    if split.status != "ready":
        return _empty_evaluation(split, include_outliers, split.unavailable_reason or "Hotel split is unavailable.")
    if "hotel_id" not in data.columns:
        raise ValueError("Hotel-group evaluation requires the 'hotel_id' column.")
    if any(index < 0 or index >= len(data) for index in (*split.train_indices, *split.test_indices)):
        raise ValueError("The stored split does not match this dataset; create a new hotel split.")

    train_positions = np.asarray(split.train_indices, dtype=int)
    test_positions = np.asarray(split.test_indices, dtype=int)
    train_rows = data.iloc[train_positions]
    test_rows = data.iloc[test_positions]
    if set(train_rows["hotel_id"].dropna()) != set(split.train_hotel_ids) or set(test_rows["hotel_id"].dropna()) != set(split.test_hotel_ids):
        raise ValueError("Hotel assignments no longer match the stored split; create a new hotel split.")
    if set(split.train_hotel_ids) & set(split.test_hotel_ids):
        raise ValueError("Invalid hotel split: a hotel appears in both training and test groups.")

    if not include_outliers and "price_outlier" in data.columns:
        flags = data["price_outlier"]
        if not pd.api.types.is_bool_dtype(flags.dtype):
            flags = flags.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})
        else:
            flags = flags.fillna(False)
        train_keep = ~flags.iloc[train_positions].to_numpy(dtype=bool)
        test_keep = ~flags.iloc[test_positions].to_numpy(dtype=bool)
        train_positions = train_positions[train_keep]
        test_positions = test_positions[test_keep]
        train_rows = data.iloc[train_positions]
        test_rows = data.iloc[test_positions]

    train_prices = pd.to_numeric(train_rows["price_eur"], errors="coerce")
    test_prices = pd.to_numeric(test_rows["price_eur"], errors="coerce")
    train_valid = np.isfinite(train_prices) & train_prices.gt(0)
    test_valid = np.isfinite(test_prices) & test_prices.gt(0)
    train_positions = train_positions[train_valid.to_numpy()]
    test_positions = test_positions[test_valid.to_numpy()]
    train_rows = data.iloc[train_positions]
    test_rows = data.iloc[test_positions]
    y_train = pd.to_numeric(train_rows["price_eur"], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test_rows["price_eur"], errors="coerce").to_numpy(dtype=float)
    if train_rows.empty or test_rows.empty:
        return _empty_evaluation(
            split, include_outliers,
            "Evaluation has no valid training or held-out prices after applying the outlier setting.",
        )

    features = _feature_columns(data)
    if not features:
        return _empty_evaluation(split, include_outliers, "No supported hotel characteristics are available.")
    x_train = train_rows.loc[:, features]
    x_test = test_rows.loc[:, features]
    models = build_candidate_pipelines(features)
    fitted: dict[str, Any] = {}
    metric_rows = []
    prediction_rows = []
    train_hotel_count = int(train_rows["hotel_id"].nunique())
    test_hotel_count = int(test_rows["hotel_id"].nunique())
    for name, estimator in models.items():
        if name == "Median baseline":
            estimator.fit(np.zeros((len(y_train), 1)), y_train)
            predictions = np.asarray(estimator.predict(np.zeros((len(y_test), 1))), dtype=float)
        else:
            estimator.fit(x_train, y_train)
            predictions = np.asarray(estimator.predict(x_test), dtype=float)
        errors = predictions - y_test
        sst = float(np.square(y_test - y_test.mean()).sum())
        r_squared = 1.0 - float(np.square(errors).sum()) / sst if len(y_test) >= 2 and sst > 0 else None
        metric_rows.append(
            {
                "model": name,
                "mae_eur": float(np.mean(np.abs(errors))),
                "rmse_eur": float(np.sqrt(np.mean(np.square(errors)))),
                "r2": r_squared,
                "train_rows": int(len(train_rows)),
                "test_rows": int(len(test_rows)),
                "train_hotel_count": train_hotel_count,
                "test_hotel_count": test_hotel_count,
                "dataset_id": split.dataset_id,
                "include_outliers": bool(include_outliers),
            }
        )
        for position, actual, predicted in zip(test_positions, y_test, predictions):
            position = int(position)
            output = {
                "model": name,
                "row_position": position,
                "hotel_id": data.iloc[position]["hotel_id"],
                "offer_id": data.iloc[position]["offer_id"] if "offer_id" in data.columns else None,
                "actual_price_eur": float(actual),
                "predicted_price_eur": float(predicted),
            }
            prediction_rows.append(output)
        fitted[name] = estimator

    return EvaluationResult(
        dataset_id=split.dataset_id,
        include_outliers=bool(include_outliers),
        metrics=pd.DataFrame(metric_rows, columns=METRIC_COLUMNS),
        predictions=pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS),
        models=fitted,
        train_hotel_ids=split.train_hotel_ids,
        test_hotel_ids=split.test_hotel_ids,
    )


def predict_price(
    data: pd.DataFrame,
    split: HotelSplit,
    evaluation: EvaluationResult,
    model_name: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Predict one offer with a fitted evaluation pipeline and audit its inputs."""
    if split.dataset_id != evaluation.dataset_id:
        raise ValueError("The stored model and hotel split belong to different datasets.")
    if model_name not in evaluation.models:
        raise ValueError("Train and evaluate a regression candidate before submitting a prediction.")
    if not split.train_indices or max(split.train_indices) >= len(data):
        raise ValueError("The stored hotel split no longer matches the active dataset.")
    model = evaluation.models[model_name]
    if model_name == "Median baseline":
        raise ValueError("Choose Linear regression or Random forest to submit a characteristic-based prediction.")
    features = list(model.feature_names_in_)
    row = {feature: values.get(feature) for feature in features}
    prediction_frame = pd.DataFrame([row], columns=features)
    predicted = float(model.predict(prediction_frame)[0])

    train_rows = data.iloc[list(split.train_indices)].copy()
    if not evaluation.include_outliers and "price_outlier" in train_rows:
        flags = train_rows["price_outlier"]
        if not pd.api.types.is_bool_dtype(flags.dtype):
            flags = flags.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})
        else:
            flags = flags.fillna(False)
        train_rows = train_rows.loc[~flags]

    imputed_fields: list[str] = []
    imputed_values: dict[str, Any] = {}
    unsupported_categories: dict[str, list[str]] = {}
    outside_training_ranges: list[str] = []
    preprocessor = model.named_steps["preprocessor"]
    learned_imputations: dict[str, Any] = {}
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "numeric":
            learned_imputations.update(zip(columns, transformer.statistics_))
        elif transformer_name == "categorical":
            imputer = transformer.named_steps["imputer"]
            learned_imputations.update(zip(columns, imputer.statistics_))
    for feature in features:
        value = row[feature]
        training = train_rows[feature].dropna()
        if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
            imputed_fields.append(feature)
            imputed_values[feature] = learned_imputations.get(feature)
            continue
        if feature in NUMERIC_FEATURES:
            numeric_training = pd.to_numeric(training, errors="coerce").dropna()
            if not numeric_training.empty and (float(value) < numeric_training.min() or float(value) > numeric_training.max()):
                outside_training_ranges.append(feature)
        elif not training.empty and str(value) not in set(training.astype(str)):
            unsupported_categories[feature] = [str(value)]

    return {
        "predicted_price_eur": predicted,
        "model_name": model_name,
        "dataset_id": evaluation.dataset_id,
        "imputed_fields": imputed_fields,
        "imputed_values": imputed_values,
        "unsupported_categories": unsupported_categories,
        "outside_training_ranges": outside_training_ranges,
        "supplied_values": row,
    }
