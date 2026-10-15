import importlib.util
import unittest

import numpy as np
import pandas as pd

from src.prediction import evaluate_candidates, make_hotel_split, predict_price


def hotel_data(hotel_count=60, rows_per_hotel=2):
    rows = []
    for hotel_id in range(1, hotel_count + 1):
        for month in range(1, rows_per_hotel + 1):
            rows.append(
                {
                    "hotel_id": hotel_id,
                    "hotel_name": f"Hotel {hotel_id}",
                    "city": "Paris" if hotel_id % 2 else "Lyon",
                    "star_rating": 3 + hotel_id % 3,
                    "guest_rating": 6.0 + (hotel_id % 4) * 0.5,
                    "room_size_m2": 20.0 + hotel_id % 10,
                    "room_type": "Standard" if hotel_id % 2 else "Deluxe",
                    "month": month,
                    "distance_center_km": float(hotel_id % 8),
                    "price_eur": 90.0 + hotel_id + month,
                    "price_outlier": bool(hotel_id % 17 == 0),
                    "offer_id": f"offer-{hotel_id}-{month}",
                }
            )
    return pd.DataFrame(rows)


class HotelSplitTests(unittest.TestCase):
    def test_split_is_hotel_disjoint_deterministic_and_twenty_percent(self):
        data = hotel_data()
        first = make_hotel_split(data, "dataset-a", random_state=19)
        second = make_hotel_split(data, "dataset-a", random_state=19)
        self.assertEqual(first.train_indices, second.train_indices)
        self.assertEqual(first.test_indices, second.test_indices)
        self.assertFalse(set(first.train_hotel_ids) & set(first.test_hotel_ids))
        self.assertEqual(len(first.test_hotel_ids), 12)
        self.assertEqual(first.dataset_id, "dataset-a")

    def test_evaluation_is_unavailable_below_fifty_eligible_hotels(self):
        data = hotel_data(hotel_count=49)
        split = make_hotel_split(data, "small")
        self.assertEqual(split.status, "unavailable")
        result = evaluate_candidates(data, split, include_outliers=True)
        self.assertIn("50", result.unavailable_reason)
        self.assertTrue(result.metrics.empty)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class CandidateEvaluationTests(unittest.TestCase):
    def test_candidates_share_held_out_rows_and_exclude_leaky_features(self):
        data = hotel_data()
        split = make_hotel_split(data, "dataset-a")
        result = evaluate_candidates(data, split, include_outliers=True)
        self.assertEqual(set(result.metrics["model"]), {"Median baseline", "Linear regression", "Random forest"})
        self.assertEqual(result.metrics["test_rows"].nunique(), 1)
        row_sets = [set(group["row_position"]) for _, group in result.predictions.groupby("model")]
        self.assertTrue(row_sets and all(rows == row_sets[0] for rows in row_sets))
        self.assertTrue(set(result.predictions["hotel_id"]).issubset(set(split.test_hotel_ids)))
        self.assertTrue(set(result.models["Linear regression"].feature_names_in_).issubset({
            "city", "star_rating", "guest_rating", "room_size_m2", "room_type", "month", "distance_center_km"
        }))

    def test_preprocessing_uses_training_rows_only_and_unknown_categories_are_safe(self):
        data = hotel_data()
        split = make_hotel_split(data, "dataset-b", random_state=7)
        data.loc[list(split.train_indices), "guest_rating"] = 4.0
        data.loc[list(split.test_indices), "guest_rating"] = 999.0
        data.loc[list(split.test_indices), "city"] = "Unknown test city"
        result = evaluate_candidates(data, split, include_outliers=True)
        pipeline = result.models["Linear regression"]
        imputer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
        numeric_features = list(pipeline.named_steps["preprocessor"].transformers_[0][2])
        guest_rating_index = numeric_features.index("guest_rating")
        self.assertEqual(imputer.statistics_[guest_rating_index], 4.0)
        self.assertTrue(np.isfinite(result.predictions["predicted_price_eur"]).all())

    def test_outlier_sensitivity_uses_same_hotel_assignment(self):
        data = hotel_data()
        split = make_hotel_split(data, "dataset-c")
        included = evaluate_candidates(data, split, include_outliers=True)
        excluded = evaluate_candidates(data, split, include_outliers=False)
        self.assertEqual(included.train_hotel_ids, excluded.train_hotel_ids)
        self.assertEqual(included.test_hotel_ids, excluded.test_hotel_ids)
        self.assertLessEqual(excluded.metrics["test_rows"].iloc[0], included.metrics["test_rows"].iloc[0])
        self.assertEqual(included.metrics["test_rows"].nunique(), 1)
        self.assertEqual(excluded.metrics["test_rows"].nunique(), 1)

    def test_prediction_reports_imputed_unknown_and_out_of_range_inputs(self):
        data = hotel_data()
        split = make_hotel_split(data, "dataset-p")
        evaluation = evaluate_candidates(data, split, include_outliers=True)
        prediction = predict_price(
            data, split, evaluation, "Linear regression",
            {"city": "Unknown City", "star_rating": 99, "guest_rating": None,
             "room_size_m2": 24, "room_type": "Suite", "month": 1,
             "distance_center_km": 0.5},
        )
        self.assertTrue(np.isfinite(prediction["predicted_price_eur"]))
        self.assertEqual(prediction["imputed_fields"], ["guest_rating"])
        self.assertEqual(prediction["imputed_values"]["guest_rating"], 6.5)
        self.assertIn("Unknown City", prediction["unsupported_categories"]["city"])
        self.assertIn("star_rating", prediction["outside_training_ranges"])

    def test_prediction_uses_the_fitted_evaluation_pipeline(self):
        data = hotel_data()
        split = make_hotel_split(data, "dataset-q")
        evaluation = evaluate_candidates(data, split, include_outliers=True)
        row = data.iloc[split.test_indices[0]]
        values = {column: row[column] for column in (
            "city", "star_rating", "guest_rating", "room_size_m2", "room_type", "month", "distance_center_km"
        )}
        result = predict_price(data, split, evaluation, "Random forest", values)
        model = evaluation.models["Random forest"]
        expected = model.predict(pd.DataFrame([values], columns=model.feature_names_in_))[0]
        self.assertAlmostEqual(result["predicted_price_eur"], expected)


class PredictionInvalidationTests(unittest.TestCase):
    def test_dataset_change_clears_prediction_session_bundle(self):
        from src.dashboard_data import activate_uploaded_data

        data = hotel_data()
        session = {"active_fingerprint": "old", "selected_offerings": ["stale"]}
        for key in ("prediction_model", "prediction_split", "prediction_evaluation", "prediction_model_name", "prediction_result"):
            session[key] = "stale"
        raw_csv = data.drop(columns=["offer_id", "price_outlier"]).to_csv(index=False).encode()
        activate_uploaded_data(session, raw_csv)
        self.assertEqual(session["active_fingerprint"] != "old", True)
        for key in ("prediction_model", "prediction_split", "prediction_evaluation", "prediction_model_name", "prediction_result", "selected_offerings"):
            self.assertNotIn(key, session)


if __name__ == "__main__":
    unittest.main()
