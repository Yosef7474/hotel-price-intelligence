"""Generate a reproducible synthetic hotel-offering dataset.

The generated prices illustrate designed relationships for learning. They are
not estimates of actual hotel prices or market behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "hotels_raw.csv"

BASE_OFFERING_COUNT = 1_000
DUPLICATE_COPY_COUNT = 20
CITY_BASELINES_EUR = {
    "Paris": 150.0,
    "Lyon": 112.0,
    "Nice": 138.0,
    "Lille": 98.0,
    "Marseille": 116.0,
    "Bordeaux": 108.0,
}
ROOM_SIZE_RANGES_M2 = {
    "Standard": (18.0, 30.0),
    "Deluxe": (28.0, 44.0),
    "Suite": (42.0, 72.0),
}
ROOM_PRICE_ADJUSTMENTS_EUR = {"Standard": 0.0, "Deluxe": 22.0, "Suite": 55.0}
MONTHLY_MULTIPLIERS = {
    1: 0.88,
    2: 0.90,
    3: 0.96,
    4: 1.02,
    5: 1.08,
    6: 1.15,
    7: 1.23,
    8: 1.20,
    9: 1.08,
    10: 1.00,
    11: 0.92,
    12: 1.12,
}


def generate_dataset(seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return 1,020 raw rows and an audit of the controlled imperfections.

    The first 1,000 rows are distinct offerings. The final 20 rows are copied
    from those base rows, so full-row deduplication can identify them exactly.
    """
    rng = np.random.default_rng(seed)
    cities = list(CITY_BASELINES_EUR)
    room_types = list(ROOM_SIZE_RANGES_M2)
    hotel_count = 120

    # Hotel-level attributes are drawn once and reused for every offering.
    hotel_cities = np.resize(np.array(cities, dtype=object), hotel_count)
    rng.shuffle(hotel_cities)
    hotel_stars = np.resize(np.array([1, 2, 3, 4, 5], dtype=int), hotel_count)
    rng.shuffle(hotel_stars)
    hotel_distances = np.round(rng.gamma(shape=2.2, scale=1.7, size=hotel_count), 1)
    hotel_distances = np.clip(hotel_distances, 0.1, 18.0)
    hotel_guest_quality = np.clip(rng.normal(loc=7.7, scale=0.85, size=hotel_count), 5.0, 9.8)

    # Sampling without replacement guarantees unique (hotel, room type, month)
    # keys and therefore distinct base offerings.
    combinations = np.array(
        [(hotel_id, room_type, month)
         for hotel_id in range(1, hotel_count + 1)
         for room_type in room_types
         for month in range(1, 13)],
        dtype=object,
    )
    selected_indices = rng.choice(len(combinations), size=BASE_OFFERING_COUNT, replace=False)
    selected = combinations[selected_indices]

    rows: list[dict[str, Any]] = []
    for hotel_id_value, room_type_value, month_value in selected:
        hotel_id = int(hotel_id_value)
        room_type = str(room_type_value)
        month = int(month_value)
        hotel_index = hotel_id - 1
        city = str(hotel_cities[hotel_index])
        stars = int(hotel_stars[hotel_index])
        low_size, high_size = ROOM_SIZE_RANGES_M2[room_type]
        room_size = float(rng.uniform(low_size, high_size))

        # Rating has its own hotel-level variation and does not determine stars.
        guest_rating = float(np.clip(hotel_guest_quality[hotel_index] + rng.normal(0, 0.45), 0, 10))
        seasonal_multiplier = MONTHLY_MULTIPLIERS[month]
        base_price = (
            CITY_BASELINES_EUR[city]
            + (stars - 1) * 26.0
            + room_size * 1.15
            + ROOM_PRICE_ADJUSTMENTS_EUR[room_type]
            - float(hotel_distances[hotel_index]) * 1.3
        )
        price = (base_price * seasonal_multiplier) + float(rng.normal(0, 14.0))
        price = max(25.0, price)

        rows.append(
            {
                "hotel_id": hotel_id,
                "hotel_name": f"{city} Practice Hotel {hotel_id:03d}",
                "city": city,
                "star_rating": stars,
                "guest_rating": round(guest_rating, 1),
                "room_size_m2": round(room_size, 1),
                "room_type": room_type,
                "month": month,
                "distance_center_km": float(hotel_distances[hotel_index]),
                "price_eur": round(price, 2),
            }
        )

    base = pd.DataFrame(rows)

    # Missingness and high prices are injected at the base-offering level.
    rating_missing_indices = rng.choice(base.index.to_numpy(), size=50, replace=False)
    size_missing_indices = rng.choice(base.index.to_numpy(), size=30, replace=False)
    base.loc[rating_missing_indices, "guest_rating"] = np.nan
    base.loc[size_missing_indices, "room_size_m2"] = np.nan

    high_price_indices = rng.choice(base.index.to_numpy(), size=10, replace=False)
    base.loc[high_price_indices, "price_eur"] = (
        base.loc[high_price_indices, "price_eur"] * rng.uniform(2.5, 3.2, size=10)
    ).round(2)

    duplicate_indices = rng.choice(base.index.to_numpy(), size=DUPLICATE_COPY_COUNT, replace=False)
    duplicate_copies = base.loc[duplicate_indices].copy()
    raw = pd.concat([base, duplicate_copies], ignore_index=True)

    audit: dict[str, Any] = {
        "seed": int(seed),
        "base_offerings": int(len(base)),
        "duplicate_copies_appended": int(len(duplicate_copies)),
        "raw_rows": int(len(raw)),
        "injected_missing_guest_ratings_in_base": 50,
        "injected_missing_room_sizes_in_base": 30,
        "injected_high_prices_in_base": 10,
        "city_list": cities,
        "room_categories": room_types,
        "hotel_population": hotel_count,
        "price_assumptions": {
            "city_baseline_eur": CITY_BASELINES_EUR,
            "star_increment_eur": 26.0,
            "room_size_increment_eur_per_m2": 1.15,
            "room_type_adjustments_eur": ROOM_PRICE_ADJUSTMENTS_EUR,
            "month_multipliers": MONTHLY_MULTIPLIERS,
            "distance_adjustment_eur_per_km": -1.3,
            "random_price_noise_standard_deviation_eur": 14.0,
            "high_price_multiplier_range": [2.5, 3.2],
        },
        "note": "All values and relationships are synthetic teaching assumptions, not market estimates.",
    }
    return raw, audit


def write_dataset(output_path: Path, *, seed: int = 42, overwrite: bool = False) -> dict[str, Any]:
    """Generate and write the raw CSV, refusing accidental replacement."""
    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing raw data at {output_path}. "
            "Choose another output path or rerun with --force after reviewing the file."
        )

    raw, audit = generate_dataset(seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_path, index=False)
    audit["output_path"] = str(output_path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the selected output CSV if it already exists.",
    )
    args = parser.parse_args()
    audit = write_dataset(args.output, seed=args.seed, overwrite=args.force)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
