"""Focused checks for price cleaning, validation, and audit behavior."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.clean_data import REQUIRED_COLUMNS, clean_dataset


def make_row(
    hotel_id,
    price,
    *,
    month=1,
    room_type="Standard",
    guest_rating=7.5,
    room_size_m2=24.0,
):
    return {
        "hotel_id": hotel_id,
        "hotel_name": f"Practice Hotel {hotel_id}",
        "city": "Lyon",
        "star_rating": 3,
        "guest_rating": guest_rating,
        "room_size_m2": room_size_m2,
        "room_type": room_type,
        "month": month,
        "distance_center_km": 2.0,
        "price_eur": price,
    }


def test_removes_only_exact_duplicates_and_preserves_allowed_missing_values():
    first = make_row(1, 100.0, guest_rating=np.nan)
    second = make_row(1, 120.0, month=2, room_type="Deluxe", room_size_m2=np.nan)
    third = make_row(2, 140.0, room_type="Suite", room_size_m2=48.0)
    raw = pd.DataFrame([first, second, third, first.copy()])
    original = raw.copy(deep=True)

    cleaned, rejected, audit = clean_dataset(raw)

    assert len(cleaned) == 3
    assert rejected.empty
    assert cleaned["hotel_id"].value_counts().loc[1] == 2
    assert cleaned["guest_rating"].isna().sum() == 1
    assert cleaned["room_size_m2"].isna().sum() == 1
    assert audit["duplicate_copies_removed"] == 1
    assert audit["retained_rows"] == 3
    assert audit["row_counts_reconcile"] is True
    pdt.assert_frame_equal(raw, original)


def test_separates_missing_nonnumeric_infinite_zero_and_negative_prices():
    prices = [None, "not a price", np.inf, 0, -4, 100]
    raw = pd.DataFrame([make_row(i + 1, price) for i, price in enumerate(prices)])

    cleaned, rejected, audit = clean_dataset(raw)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["price_eur"] == 100
    assert len(rejected) == 5
    assert audit["invalid_price_rows_separated"] == 5
    assert audit["retained_rows"] + audit["invalid_price_rows_separated"] == audit["deduplicated_rows"]
    assert audit["row_counts_reconcile"] is True


def test_flags_but_retains_a_valid_extreme_price():
    prices = list(range(100, 109)) + [1000]
    raw = pd.DataFrame([make_row(i + 1, price) for i, price in enumerate(prices)])

    cleaned, rejected, audit = clean_dataset(raw)

    assert rejected.empty
    assert len(cleaned) == len(prices)
    assert cleaned["price_outlier"].sum() == 1
    assert bool(cleaned.loc[cleaned["price_eur"] == 1000, "price_outlier"].item())
    assert audit["flagged_prices"] == 1


def test_missing_required_columns_name_the_missing_fields():
    raw = pd.DataFrame([make_row(1, 100.0)]).drop(columns=["city", "month"])

    with pytest.raises(ValueError, match="city, month"):
        clean_dataset(raw)


@pytest.mark.parametrize(
    ("column", "value"),
    [("star_rating", 6), ("month", 13), ("room_size_m2", 0), ("city", "")],
)
def test_invalid_non_price_fields_raise_informative_errors(column, value):
    raw = pd.DataFrame([make_row(1, 100.0)])
    raw.loc[0, column] = value

    with pytest.raises(ValueError, match=column):
        clean_dataset(raw)


def test_hotel_identity_attributes_must_remain_consistent():
    first = make_row(1, 100.0)
    second = make_row(1, 120.0, month=2, room_type="Deluxe")
    second["city"] = "Paris"
    raw = pd.DataFrame([first, second], columns=REQUIRED_COLUMNS)

    with pytest.raises(ValueError, match="consistent for each hotel_id"):
        clean_dataset(raw)
