"""Validate and clean synthetic hotel-offering data for price analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "hotels_raw.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "hotels_cleaned.csv"

REQUIRED_COLUMNS = [
    "hotel_id",
    "hotel_name",
    "city",
    "star_rating",
    "guest_rating",
    "room_size_m2",
    "room_type",
    "month",
    "distance_center_km",
    "price_eur",
]

NON_PRICE_NUMERIC_DOMAINS = {
    "hotel_id": (None, None, True),
    "star_rating": (1, 5, True),
    "guest_rating": (0, 10, False),
    "room_size_m2": (0, None, False),
    "month": (1, 12, True),
    "distance_center_km": (0, None, False),
}
TEXT_COLUMNS = ["hotel_name", "city", "room_type"]
HOTEL_STABLE_COLUMNS = ["hotel_name", "city", "star_rating", "distance_center_km"]


def _validate_required_columns(data: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {', '.join(missing)}")


def _validate_non_price_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize non-price fields, preserving allowed nulls."""
    result = data.copy()

    for column in TEXT_COLUMNS:
        values = result[column]
        is_text = values.map(lambda value: isinstance(value, str))
        invalid = values.isna() | ~is_text | values.astype("string").str.strip().eq("")
        if invalid.any():
            rows = result.index[invalid].tolist()[:5]
            raise ValueError(f"Invalid non-price field '{column}' at row index(es) {rows}: expected non-empty text.")
        result[column] = values.astype(str)

    for column, (minimum, maximum, integer_only) in NON_PRICE_NUMERIC_DOMAINS.items():
        original = result[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        invalid |= converted.notna() & ~np.isfinite(converted.astype(float))
        if original.isna().any() and column not in {"guest_rating", "room_size_m2"}:
            invalid |= original.isna()
        if minimum is not None:
            invalid |= converted.notna() & (converted < minimum)
        if maximum is not None:
            invalid |= converted.notna() & (converted > maximum)
        if integer_only:
            invalid |= converted.notna() & (converted % 1 != 0)
        if column == "room_size_m2":
            invalid |= converted.notna() & (converted <= 0)
        if invalid.any():
            rows = result.index[invalid].tolist()[:5]
            domain = "finite integer" if integer_only else "finite number"
            if minimum is not None or maximum is not None:
                domain += f" in the range {minimum if minimum is not None else '(open)'} to {maximum if maximum is not None else '(open)'}"
            if column == "room_size_m2":
                domain = "positive finite number when present"
            raise ValueError(f"Invalid non-price field '{column}' at row index(es) {rows}: expected {domain}.")

        # Nullable values in the two learner-measurement fields remain NaN.
        result[column] = converted
        if integer_only:
            result[column] = result[column].astype("int64")

    # A hotel's identity attributes must not change between its offerings.
    for column in HOTEL_STABLE_COLUMNS:
        varying = result.groupby("hotel_id", dropna=False)[column].nunique(dropna=False) > 1
        if varying.any():
            hotel_ids = varying[varying].index.tolist()[:5]
            raise ValueError(
                f"Invalid non-price field '{column}': values must remain consistent for each hotel_id; "
                f"affected hotel_id(s): {hotel_ids}."
            )
    return result


def clean_dataset(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return cleaned price rows, invalid-price rows, and a reconciliation audit.

    Exact full-row duplicates are removed using the original input columns,
    before numeric normalization or the derived ``price_outlier`` flag.
    """
    _validate_required_columns(raw_data)
    original = raw_data.loc[:, REQUIRED_COLUMNS].copy()
    input_rows = len(original)

    deduplicated = original.drop_duplicates(keep="first").copy()
    duplicate_copies_removed = input_rows - len(deduplicated)
    normalized = _validate_non_price_fields(deduplicated)

    numeric_prices = pd.to_numeric(normalized["price_eur"], errors="coerce")
    valid_price_mask = numeric_prices.notna() & np.isfinite(numeric_prices.astype(float)) & (numeric_prices > 0)
    rejected = normalized.loc[~valid_price_mask].copy()
    retained = normalized.loc[valid_price_mask].copy()
    retained["price_eur"] = numeric_prices.loc[valid_price_mask].astype(float)
    rejected["price_eur"] = numeric_prices.loc[~valid_price_mask]

    if not retained.empty:
        q1 = float(retained["price_eur"].quantile(0.25))
        q3 = float(retained["price_eur"].quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        retained["price_outlier"] = (retained["price_eur"] < lower_fence) | (retained["price_eur"] > upper_fence)
        flagged_prices = int(retained["price_outlier"].sum())
    else:
        q1 = q3 = iqr = lower_fence = upper_fence = None
        retained["price_outlier"] = pd.Series(index=retained.index, dtype=bool)
        flagged_prices = 0

    missing_values = {column: int(count) for column, count in retained.isna().sum().items()}
    audit: dict[str, Any] = {
        "input_rows": int(input_rows),
        "duplicate_copies_removed": int(duplicate_copies_removed),
        "deduplicated_rows": int(len(deduplicated)),
        "invalid_price_rows_separated": int(len(rejected)),
        "retained_rows": int(len(retained)),
        "remaining_missing_values": missing_values,
        "flagged_prices": flagged_prices,
        "outlier_rule": "Global Tukey fences: below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.",
        "outlier_fences_eur": {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower_fence,
            "upper": upper_fence,
        },
        "row_counts_reconcile": input_rows - duplicate_copies_removed == len(retained) + len(rejected),
        "note": "Outlier flags identify unusual values under a global rule; they do not establish data errors.",
    }
    return retained.reset_index(drop=True), rejected.reset_index(drop=True), audit


def clean_file(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Read raw CSV, clean it, and refresh the separate cleaned CSV."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {input_path}. Generate it first with: python src/generate_data.py"
        )

    raw_data = pd.read_csv(input_path)
    cleaned, rejected, audit = clean_dataset(raw_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    audit["input_path"] = str(input_path)
    audit["output_path"] = str(output_path)
    audit["rejected_rows_available_in_memory"] = int(len(rejected))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Raw CSV path (default: {DEFAULT_INPUT}).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Cleaned CSV path (default: {DEFAULT_OUTPUT}).")
    args = parser.parse_args()
    audit = clean_file(args.input, args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
