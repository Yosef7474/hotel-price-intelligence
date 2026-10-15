"""Data-ingress and session-state tests for the Streamlit dashboard."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.clean_data import REQUIRED_COLUMNS
from src.dashboard_data import (
    DatasetValidationError,
    activate_uploaded_data,
    add_offering_ids,
    dataset_fingerprint,
    load_project_data,
    validate_upload,
)


def row(hotel_id: int, price: object, *, month: int = 1, room_type: str = "Standard") -> dict:
    return {
        "hotel_id": hotel_id,
        "hotel_name": f"Practice Hotel {hotel_id}",
        "city": "Lyon",
        "star_rating": 3,
        "guest_rating": 7.5,
        "room_size_m2": 24.0,
        "room_type": room_type,
        "month": month,
        "distance_center_km": 2.0,
        "price_eur": price,
    }


def csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")


class DashboardDataTests(unittest.TestCase):
    def test_upload_cleans_duplicate_and_separates_invalid_price(self):
        first = row(1, 100.0)
        source = pd.DataFrame(
            [
                first,
                row(1, 110.0, month=2, room_type="Deluxe"),
                row(2, ""),
                first.copy(),
            ],
            columns=REQUIRED_COLUMNS,
        )

        result = validate_upload(csv_bytes(source))

        self.assertEqual(len(result.data), 2)
        self.assertEqual(len(result.rejected_prices), 1)
        self.assertEqual(result.audit["duplicate_copies_removed"], 1)
        self.assertEqual(result.audit["invalid_price_rows_separated"], 1)
        self.assertEqual(len(result.fingerprint), 64)
        self.assertIn("offer_id", result.data.columns)

    def test_upload_rejects_missing_required_columns(self):
        source = pd.DataFrame([row(1, 100.0)]).drop(columns=["city"])

        with self.assertRaisesRegex(DatasetValidationError, "city"):
            validate_upload(csv_bytes(source))

    def test_failed_upload_leaves_active_session_data_unchanged(self):
        previous = validate_upload(csv_bytes(pd.DataFrame([row(1, 100.0)])))
        state = {
            "active_dataset": previous.data,
            "active_audit": previous.audit,
            "active_fingerprint": previous.fingerprint,
            "comparison_selection": ["offer-a"],
            "model_state": {"trained": True},
            "filter_price_range": (10.0, 500.0),
        }
        before = dict(state)
        invalid = pd.DataFrame([row(2, 100.0)]).drop(columns=["month"])

        with self.assertRaises(DatasetValidationError):
            activate_uploaded_data(state, csv_bytes(invalid))

        self.assertEqual(state.keys(), before.keys())
        for key, value in before.items():
            self.assertIs(state[key], value)

    def test_successful_upload_replaces_data_and_clears_stale_state(self):
        previous = validate_upload(csv_bytes(pd.DataFrame([row(1, 100.0)])))
        state = {
            "active_dataset": previous.data,
            "active_audit": previous.audit,
            "active_fingerprint": previous.fingerprint,
            "comparison_selection": ["offer-a"],
            "model_state": {"trained": True},
        }
        replacement = pd.DataFrame([row(2, 180.0)])

        result = activate_uploaded_data(state, csv_bytes(replacement))

        self.assertEqual(state["active_fingerprint"], result.fingerprint)
        self.assertNotEqual(result.fingerprint, previous.fingerprint)
        self.assertNotIn("comparison_selection", state)
        self.assertNotIn("model_state", state)
        self.assertNotIn("filter_price_range", state)
        self.assertEqual(state["active_dataset"].iloc[0]["hotel_id"], 2)

    def test_offering_ids_remain_with_offerings_after_sorting(self):
        source = pd.DataFrame(
            [row(7, 120.0, month=2, room_type="Deluxe"), row(7, 100.0)],
            columns=REQUIRED_COLUMNS,
        )
        identified = add_offering_ids(source)
        ids_by_month = identified.set_index("month")["offer_id"].to_dict()
        sorted_data = identified.sort_values("price_eur")

        self.assertEqual(sorted_data.set_index("month")["offer_id"].to_dict(), ids_by_month)
        self.assertEqual(sorted_data["offer_id"].nunique(), 2)

    def test_dataset_fingerprint_ignores_row_order_but_tracks_content(self):
        source = pd.DataFrame([row(1, 100.0), row(2, 120.0, month=2)])
        reversed_rows = source.iloc[::-1].reset_index(drop=True)
        changed = source.copy()
        changed.loc[0, "price_eur"] = 101.0

        self.assertEqual(dataset_fingerprint(source), dataset_fingerprint(reversed_rows))
        self.assertNotEqual(dataset_fingerprint(source), dataset_fingerprint(changed))

    def test_cleaned_only_load_does_not_claim_historical_cleaning_counts(self):
        cleaned_source = pd.DataFrame([row(4, 150.0)])
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "missing_raw.csv"
            cleaned_path = Path(temp_dir) / "hotels_cleaned.csv"
            cleaned_source.to_csv(cleaned_path, index=False)

            result = load_project_data(raw_path=raw_path, cleaned_path=cleaned_path)

        self.assertEqual(result.audit["source_kind"], "cleaned")
        self.assertIsNone(result.audit["duplicate_copies_removed"])
        self.assertIsNone(result.audit["invalid_price_rows_separated"])
        self.assertEqual(len(result.data), 1)


if __name__ == "__main__":
    unittest.main()
