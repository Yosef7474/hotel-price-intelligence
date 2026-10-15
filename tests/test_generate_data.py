"""Focused checks for reproducible synthetic data generation."""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.generate_data import BASE_OFFERING_COUNT, DUPLICATE_COPY_COUNT, generate_dataset


def test_same_seed_reproduces_rows_and_audit():
    first_data, first_audit = generate_dataset(seed=42)
    second_data, second_audit = generate_dataset(seed=42)

    pdt.assert_frame_equal(first_data, second_data)
    assert first_audit == second_audit


def test_schema_ranges_and_controlled_issue_counts():
    raw, audit = generate_dataset(seed=7)
    base = raw.iloc[:BASE_OFFERING_COUNT]

    assert list(raw.columns) == [
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
    assert len(base) == BASE_OFFERING_COUNT
    assert len(raw) == BASE_OFFERING_COUNT + DUPLICATE_COPY_COUNT
    assert base[["hotel_id", "room_type", "month"]].drop_duplicates().shape[0] == BASE_OFFERING_COUNT
    assert raw.duplicated().sum() == DUPLICATE_COPY_COUNT
    assert set(base["star_rating"]).issubset({1, 2, 3, 4, 5})
    assert set(base["month"]).issubset(set(range(1, 13)))
    assert base["room_size_m2"].dropna().gt(0).all()
    assert base["guest_rating"].dropna().between(0, 10).all()
    assert np.isfinite(base["price_eur"]).all()
    assert base["price_eur"].gt(0).all()

    assert base["guest_rating"].isna().sum() == 50
    assert base["room_size_m2"].isna().sum() == 30
    assert audit["injected_high_prices_in_base"] == 10
    assert audit["seed"] == 7
    assert audit["base_offerings"] == BASE_OFFERING_COUNT
    assert audit["duplicate_copies_appended"] == DUPLICATE_COPY_COUNT


def test_hotel_attributes_are_stable_across_offerings():
    raw, _ = generate_dataset(seed=19)

    for column in ["hotel_name", "city", "star_rating", "distance_center_km"]:
        assert raw.groupby("hotel_id")[column].nunique(dropna=False).le(1).all()
