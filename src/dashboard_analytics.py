"""Reusable population filters and summaries for the hotel dashboard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _selected_values(state: Mapping[str, Any], key: str) -> list[Any] | None:
    value = state.get(key)
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return [value]
    values = list(value)
    return values or None


def _require_column(data: pd.DataFrame, column: str) -> None:
    if column not in data.columns:
        raise ValueError(f"Cannot apply dashboard filter: required column '{column}' is missing.")


def apply_filters(data: pd.DataFrame, filter_state: Mapping[str, Any] | None) -> pd.DataFrame:
    """Return the one shared population selected by the dashboard controls.

    Numeric minimum and maximum bounds are inclusive. Empty or omitted category
    selections mean all values. Unknown room sizes are excluded only when a size
    bound is active, unless ``include_unknown_room_size`` is true.
    """
    state = filter_state or {}
    result = data.copy()
    category_filters = {
        "cities": "city",
        "star_ratings": "star_rating",
        "months": "month",
        "room_types": "room_type",
    }
    for state_key, column in category_filters.items():
        values = _selected_values(state, state_key)
        if values is not None:
            _require_column(result, column)
            result = result.loc[result[column].isin(values)]

    for state_key, column, bound in (
        ("price_min", "price_eur", "min"),
        ("price_max", "price_eur", "max"),
    ):
        value = state.get(state_key)
        if value is None:
            continue
        _require_column(result, column)
        numeric = pd.to_numeric(result[column], errors="coerce")
        result = result.loc[numeric.ge(value) if bound == "min" else numeric.le(value)]

    size_min = state.get("room_size_min")
    size_max = state.get("room_size_max")
    if size_min is not None or size_max is not None:
        _require_column(result, "room_size_m2")
        sizes = pd.to_numeric(result["room_size_m2"], errors="coerce")
        size_matches = sizes.notna()
        if size_min is not None:
            size_matches &= sizes.ge(size_min)
        if size_max is not None:
            size_matches &= sizes.le(size_max)
        if state.get("include_unknown_room_size", False):
            size_matches |= sizes.isna()
        result = result.loc[size_matches]

    if not state.get("include_outliers", True) and "price_outlier" in result:
        result = result.loc[~result["price_outlier"].fillna(False).astype(bool)]
    return result.copy()


def summarize_offerings(data: pd.DataFrame) -> dict[str, int | float | None]:
    """Summarize the supplied offering population without inventing empty metrics."""
    for column in ("hotel_id", "price_eur", "guest_rating", "room_size_m2"):
        _require_column(data, column)
    prices = pd.to_numeric(data["price_eur"], errors="coerce")
    ratings = pd.to_numeric(data["guest_rating"], errors="coerce").dropna()
    sizes = pd.to_numeric(data["room_size_m2"], errors="coerce").dropna()
    valid_prices = prices[np.isfinite(prices)]
    return {
        "offer_count": int(len(data)),
        "hotel_count": int(data["hotel_id"].nunique(dropna=True)),
        "median_price_eur": float(valid_prices.median()) if not valid_prices.empty else None,
        "median_guest_rating": float(ratings.median()) if not ratings.empty else None,
        "guest_rating_count": int(len(ratings)),
        "room_size_count": int(len(sizes)),
        "price_count": int(len(valid_prices)),
    }


def price_summaries(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Return count, mean, median, and interquartile spread by group."""
    _require_column(data, group_column)
    _require_column(data, "price_eur")
    columns = [group_column, "count", "mean_price_eur", "median_price_eur", "spread_eur"]
    working = data[[group_column, "price_eur"]].copy()
    working["price_eur"] = pd.to_numeric(working["price_eur"], errors="coerce")
    working = working.loc[np.isfinite(working["price_eur"])]
    if working.empty:
        return pd.DataFrame(columns=columns)

    def spread(values: pd.Series) -> float:
        return float(values.quantile(0.75) - values.quantile(0.25))

    grouped = working.groupby(group_column, dropna=False)["price_eur"]
    result = grouped.agg(count="count", mean_price_eur="mean", median_price_eur="median")
    result["spread_eur"] = grouped.apply(spread)
    return result.reset_index()[columns]


def correlation_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise Pearson correlations with complete-case counts.

    Identifier and derived-flag columns are intentionally excluded. Constant
    pairs are included with a missing coefficient so their sample count remains
    visible to callers.
    """
    excluded = {"hotel_id", "offer_id", "price_outlier"}
    numeric_columns = [
        column
        for column in data.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    columns = ["feature_1", "feature_2", "complete_pairs", "correlation"]
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(numeric_columns):
        for right in numeric_columns[left_index + 1 :]:
            pair = data[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            if pair.empty or pair[left].nunique() <= 1 or pair[right].nunique() <= 1:
                coefficient = None
            else:
                value = pair[left].corr(pair[right])
                coefficient = float(value) if pd.notna(value) else None
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "complete_pairs": int(len(pair)),
                    "correlation": coefficient,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def peer_comparison(
    data: pd.DataFrame,
    offer_id: str,
    *,
    min_peer_count: int = 5,
) -> dict[str, Any]:
    """Compare an offer with other filtered offers sharing its four peer fields.

    Peer offers must match city, star classification, room type, and month.
    The selected offer is excluded. Median and percentage difference are
    withheld until at least ``min_peer_count`` other valid-price offers exist.
    """
    required = ("offer_id", "city", "star_rating", "room_type", "month", "price_eur", "guest_rating")
    for column in required:
        _require_column(data, column)
    if min_peer_count < 1:
        raise ValueError("min_peer_count must be at least 1.")

    selected_rows = data.loc[data["offer_id"].astype(str).eq(str(offer_id))]
    if selected_rows.empty:
        return {
            "offer_id": offer_id, "peer_count": 0, "peer_median_eur": None,
            "percentage_difference": None, "status": "offer_not_found",
            "discussion_candidate": False, "minimum_peer_count": min_peer_count,
        }
    if len(selected_rows) != 1:
        raise ValueError(f"offer_id '{offer_id}' must identify exactly one offering.")

    selected = selected_rows.iloc[0]
    peer_fields = ("city", "star_rating", "room_type", "month")
    if selected.loc[list(peer_fields)].isna().any():
        peers = data.iloc[0:0]
    else:
        matches = pd.Series(True, index=data.index)
        for field in peer_fields:
            matches &= data[field].eq(selected[field])
        matches &= data["offer_id"].astype(str).ne(str(offer_id))
        peers = data.loc[matches]

    peer_prices = pd.to_numeric(peers["price_eur"], errors="coerce")
    valid = np.isfinite(peer_prices) & peer_prices.gt(0)
    peers = peers.loc[valid]
    peer_count = int(len(peers))
    result: dict[str, Any] = {
        "offer_id": str(offer_id),
        "peer_count": peer_count,
        "peer_median_eur": None,
        "percentage_difference": None,
        "status": "insufficient_evidence" if peer_count < min_peer_count else "sufficient",
        "discussion_candidate": False,
        "minimum_peer_count": int(min_peer_count),
    }
    if peer_count < min_peer_count:
        return result

    median = float(pd.to_numeric(peers["price_eur"], errors="coerce").median())
    selected_price = float(selected["price_eur"])
    if not np.isfinite(selected_price) or selected_price <= 0:
        result["status"] = "invalid_selected_price"
        return result
    difference = (selected_price - median) / median * 100.0
    rating = pd.to_numeric(pd.Series([selected["guest_rating"]]), errors="coerce").iloc[0]
    result.update(
        {
            "peer_median_eur": median,
            "percentage_difference": float(difference),
            "discussion_candidate": bool(pd.notna(rating) and rating >= 8 and selected_price < median),
        }
    )
    return result


def outlier_sensitivity_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Compare price and size-price statistics with and without flagged rows.

    The existing ``price_outlier`` value is respected; this function never
    recalculates flags. Correlation counts include only complete finite pairs.
    """
    for column in ("price_eur", "room_size_m2"):
        _require_column(data, column)
    if "price_outlier" in data.columns:
        flags = data["price_outlier"]
        if pd.api.types.is_bool_dtype(flags.dtype):
            flagged = flags.fillna(False)
        else:
            flagged = flags.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})
    else:
        flagged = pd.Series(False, index=data.index)

    populations = (
        ("All flagged prices included", data),
        ("Flagged prices excluded", data.loc[~flagged]),
    )
    columns = [
        "population", "offer_count", "price_count", "mean_price_eur", "median_price_eur",
        "spread_eur", "size_price_pairs", "room_size_price_correlation",
    ]
    rows: list[dict[str, Any]] = []
    for label, frame in populations:
        prices = pd.to_numeric(frame["price_eur"], errors="coerce")
        valid_price_mask = np.isfinite(prices) & prices.gt(0)
        valid_prices = prices.loc[valid_price_mask]
        if valid_prices.empty:
            mean_price = median_price = spread = None
        else:
            mean_price = float(valid_prices.mean())
            median_price = float(valid_prices.median())
            spread = float(valid_prices.quantile(0.75) - valid_prices.quantile(0.25))

        sizes = pd.to_numeric(frame["room_size_m2"], errors="coerce")
        pair_mask = np.isfinite(sizes) & sizes.gt(0) & valid_price_mask
        pair_sizes = sizes.loc[pair_mask]
        pair_prices = prices.loc[pair_mask]
        coefficient = (
            pair_sizes.corr(pair_prices)
            if len(pair_sizes) >= 2 and pair_sizes.nunique() > 1 and pair_prices.nunique() > 1
            else None
        )
        if coefficient is not None and pd.isna(coefficient):
            coefficient = None
        rows.append(
            {
                "population": label,
                "offer_count": int(len(frame)),
                "price_count": int(len(valid_prices)),
                "mean_price_eur": mean_price,
                "median_price_eur": median_price,
                "spread_eur": spread,
                "size_price_pairs": int(len(pair_sizes)),
                "room_size_price_correlation": float(coefficient) if coefficient is not None else None,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def generate_insights(
    data: pd.DataFrame,
    *,
    minimum_group_count: int = 5,
    max_insights: int = 3,
) -> list[str]:
    """Generate stable, descriptive statements from the supplied population."""
    if minimum_group_count < 1:
        raise ValueError("minimum_group_count must be at least 1.")
    if max_insights < 0:
        raise ValueError("max_insights cannot be negative.")
    if max_insights == 0 or data.empty:
        return []
    _require_column(data, "price_eur")
    statements: list[str] = []

    for column, label in (("city", "city"), ("month", "month")):
        if column not in data.columns:
            continue
        grouped = price_summaries(data, column)
        grouped = grouped.loc[grouped["count"] >= minimum_group_count].copy()
        if len(grouped) < 2:
            continue
        grouped["_label_sort"] = grouped[column].astype(str)
        grouped = grouped.sort_values(["median_price_eur", "_label_sort"], kind="stable")
        low, high = grouped.iloc[0], grouped.iloc[-1]
        if float(high["median_price_eur"]) == float(low["median_price_eur"]):
            continue
        low_label = str(low[column]) if label == "city" else f"month {int(low[column]):02d}"
        high_label = str(high[column]) if label == "city" else f"month {int(high[column]):02d}"
        statements.append(
            f"In the matching offers, the highest median nightly price was in {high_label} "
            f"(€{high['median_price_eur']:.2f}, n={int(high['count'])}) and the lowest was in "
            f"{low_label} (€{low['median_price_eur']:.2f}, n={int(low['count'])}); "
            f"only {label} groups with at least {minimum_group_count} offers are shown."
        )
        if len(statements) >= max_insights:
            break
    return statements[:max_insights]
