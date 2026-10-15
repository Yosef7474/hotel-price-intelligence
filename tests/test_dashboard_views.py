import unittest

import pandas as pd

from src.dashboard_views import build_html_report, csv_export, select_offerings


class DashboardViewHelperTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "offer_id": ["offer-b", "offer-a"],
                "hotel_id": [7, 7],
                "hotel_name": ["<b>Hotel B</b>", "Hotel A"],
                "city": ["Paris", "Paris"],
                "guest_rating": [8.1, 7.9],
                "room_size_m2": [20.0, 25.0],
                "price_eur": [120.0, 140.0],
            }
        )

    def test_selection_uses_ids_and_preserves_user_order_with_repeated_hotel(self):
        selected = select_offerings(self.data, ["offer-a", "offer-b"])
        self.assertEqual(selected["offer_id"].tolist(), ["offer-a", "offer-b"])
        self.assertEqual(selected["hotel_id"].nunique(), 1)

    def test_csv_export_contains_exact_supplied_rows_and_header(self):
        exported = csv_export(self.data.iloc[[1]])
        self.assertTrue(exported.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"offer-a", exported)
        self.assertNotIn(b"offer-b", exported)

    def test_html_report_describes_population_and_escapes_table_content(self):
        report = build_html_report(self.data, {"cities": ["Paris"]})
        self.assertIn("2 matching offers", report)
        self.assertIn("Cities:</strong> Paris", report)
        self.assertIn("&lt;b&gt;Hotel B&lt;/b&gt;", report)
        self.assertIn("synthetic learning data", report)


if __name__ == "__main__":
    unittest.main()
