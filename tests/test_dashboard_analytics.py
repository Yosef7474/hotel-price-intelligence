import unittest

import numpy as np
import pandas as pd

from src.dashboard_analytics import (
    apply_filters,
    correlation_summary,
    generate_insights,
    peer_comparison,
    price_summaries,
    outlier_sensitivity_summary,
    summarize_offerings,
)


def sample_data():
    return pd.DataFrame(
        {
            "offer_id": ["a", "b", "c", "d"],
            "hotel_id": [1, 1, 2, 3],
            "city": ["Paris", "Paris", "Lyon", "Lyon"],
            "star_rating": [3, 3, 4, 4],
            "guest_rating": [8.0, np.nan, 9.0, 7.0],
            "room_size_m2": [20.0, np.nan, 30.0, 40.0],
            "room_type": ["Standard", "Deluxe", "Suite", "Suite"],
            "month": [1, 2, 1, 2],
            "price_eur": [100.0, 200.0, 300.0, 400.0],
            "distance_center_km": [1.0, 2.0, 3.0, 4.0],
            "price_outlier": [False, True, False, True],
        }
    )


class DashboardAnalyticsTests(unittest.TestCase):
    def test_filter_boundaries_are_inclusive_and_combine(self):
        result = apply_filters(
            sample_data(),
            {"cities": ["Paris"], "price_min": 100, "price_max": 100, "months": [1]},
        )
        self.assertEqual(result["offer_id"].tolist(), ["a"])

    def test_room_size_range_excludes_unknown_unless_requested(self):
        data = sample_data()
        state = {"room_size_min": 20, "room_size_max": 30}
        self.assertEqual(apply_filters(data, state)["offer_id"].tolist(), ["a", "c"])
        state["include_unknown_room_size"] = True
        self.assertEqual(apply_filters(data, state)["offer_id"].tolist(), ["a", "b", "c"])

    def test_outlier_flag_is_global_and_only_applied_as_a_view_filter(self):
        data = sample_data()
        filtered = apply_filters(data, {"cities": ["Paris"], "include_outliers": False})
        self.assertEqual(filtered["offer_id"].tolist(), ["a"])
        self.assertTrue(data.loc[data.offer_id.eq("b"), "price_outlier"].item())

    def test_summaries_use_matching_population_and_report_usable_counts(self):
        filtered = apply_filters(sample_data(), {"cities": ["Paris"]})
        summary = summarize_offerings(filtered)
        self.assertEqual(summary["offer_count"], 2)
        self.assertEqual(summary["hotel_count"], 1)
        self.assertEqual(summary["median_price_eur"], 150)
        self.assertEqual(summary["guest_rating_count"], 1)
        self.assertEqual(summary["room_size_count"], 1)

    def test_empty_summary_values_are_missing_not_zero(self):
        summary = summarize_offerings(sample_data().iloc[0:0])
        self.assertEqual(summary["offer_count"], 0)
        self.assertIsNone(summary["median_price_eur"])
        self.assertIsNone(summary["median_guest_rating"])

    def test_price_summaries_return_count_mean_median_and_iqr(self):
        result = price_summaries(sample_data(), "city").set_index("city")
        self.assertEqual(result.loc["Paris", "count"], 2)
        self.assertEqual(result.loc["Paris", "mean_price_eur"], 150)
        self.assertEqual(result.loc["Paris", "median_price_eur"], 150)
        self.assertEqual(result.loc["Paris", "spread_eur"], 50)

    def test_correlation_summary_reports_pair_counts_and_omits_identifiers(self):
        result = correlation_summary(sample_data())
        pair = result[(result["feature_1"] == "room_size_m2") & (result["feature_2"] == "price_eur")].iloc[0]
        self.assertEqual(pair["complete_pairs"], 3)
        self.assertNotIn("hotel_id", set(result["feature_1"]) | set(result["feature_2"]))
        self.assertNotIn("offer_id", set(result["feature_1"]) | set(result["feature_2"]))

    def test_peer_comparison_excludes_selected_offer_and_calculates_difference(self):
        data = peer_fixture(5)
        result = peer_comparison(data, "selected")
        self.assertEqual(result["peer_count"], 5)
        self.assertEqual(result["peer_median_eur"], 130)
        self.assertAlmostEqual(result["percentage_difference"], (100 - 130) / 130 * 100)
        self.assertEqual(result["status"], "sufficient")

    def test_sparse_peer_group_is_insufficient_and_not_a_candidate(self):
        data = peer_fixture(4)
        result = peer_comparison(data, "selected")
        self.assertEqual(result["peer_count"], 4)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["peer_median_eur"])
        self.assertFalse(result["discussion_candidate"])

    def test_discussion_candidate_needs_rating_at_least_eight_and_below_peer_median(self):
        data = peer_fixture(5)
        self.assertTrue(peer_comparison(data, "selected")["discussion_candidate"])
        data.loc[data.offer_id.eq("selected"), "guest_rating"] = 7.9
        self.assertFalse(peer_comparison(data, "selected")["discussion_candidate"])
        data.loc[data.offer_id.eq("selected"), "guest_rating"] = 8.0
        data.loc[data.offer_id.eq("selected"), "price_eur"] = 140
        self.assertFalse(peer_comparison(data, "selected")["discussion_candidate"])

    def test_insights_are_deterministic_and_name_population_counts(self):
        data = insight_fixture()
        first = generate_insights(data)
        second = generate_insights(data.copy())
        self.assertEqual(first, second)
        self.assertTrue(any("Paris" in item and "n=5" in item and "matching offers" in item for item in first))
        self.assertLessEqual(len(first), 3)

    def test_outlier_sensitivity_reports_row_and_complete_pair_counts(self):
        result = outlier_sensitivity_summary(sample_data()).set_index("population")
        included = result.loc["All flagged prices included"]
        excluded = result.loc["Flagged prices excluded"]
        self.assertEqual(included["offer_count"], 4)
        self.assertEqual(excluded["offer_count"], 2)
        self.assertEqual(included["size_price_pairs"], 3)
        self.assertEqual(excluded["size_price_pairs"], 2)
        self.assertIsNotNone(included["room_size_price_correlation"])
        self.assertIsNotNone(excluded["room_size_price_correlation"])

    def test_outlier_sensitivity_constant_size_keeps_pair_count_but_no_correlation(self):
        data = sample_data()
        data["room_size_m2"] = 25.0
        result = outlier_sensitivity_summary(data).set_index("population")
        self.assertEqual(result.loc["All flagged prices included", "size_price_pairs"], 4)
        self.assertIsNone(result.loc["All flagged prices included", "room_size_price_correlation"])

    def test_correlation_summary_constant_field_keeps_pair_count_without_coefficient(self):
        data = sample_data()
        data["room_size_m2"] = 25.0
        result = correlation_summary(data)
        pair = result[(result.feature_1 == "room_size_m2") & (result.feature_2 == "price_eur")].iloc[0]
        self.assertEqual(pair["complete_pairs"], 4)
        self.assertTrue(pd.isna(pair["correlation"]))

    def test_outlier_sensitivity_empty_population_has_no_fabricated_statistics(self):
        result = outlier_sensitivity_summary(sample_data().iloc[0:0])
        self.assertEqual(result["offer_count"].tolist(), [0, 0])
        self.assertTrue(result["median_price_eur"].isna().all())
        self.assertTrue(result["room_size_price_correlation"].isna().all())

    def test_city_and_star_context_summaries_keep_group_counts(self):
        data = sample_data()
        city = price_summaries(data, "city").set_index("city")
        stars = price_summaries(data, "star_rating").set_index("star_rating")
        self.assertEqual(city.loc["Lyon", "count"], 2)
        self.assertEqual(stars.loc[3, "count"], 2)


def peer_fixture(peer_count):
    rows = [{
        "offer_id": "selected", "hotel_id": 1, "city": "Paris", "star_rating": 3,
        "room_type": "Standard", "month": 1, "price_eur": 100.0, "guest_rating": 8.5,
    }]
    for number in range(peer_count):
        rows.append({
            "offer_id": f"peer-{number}", "hotel_id": number + 2, "city": "Paris", "star_rating": 3,
            "room_type": "Standard", "month": 1, "price_eur": 110.0 + 10 * number, "guest_rating": 7.0,
        })
    return pd.DataFrame(rows)


def insight_fixture():
    rows = []
    for city, base_price in (("Paris", 150), ("Lyon", 100)):
        for index in range(5):
            rows.append({"city": city, "month": 1, "price_eur": base_price + index})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
