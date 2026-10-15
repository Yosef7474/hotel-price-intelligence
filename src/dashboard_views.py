"""Streamlit rendering for the hotel price dashboard."""

from __future__ import annotations

from html import escape
from typing import Any, Mapping

import pandas as pd

from src.dashboard_analytics import (
    correlation_summary,
    generate_insights,
    outlier_sensitivity_summary,
    peer_comparison,
    price_summaries,
    summarize_offerings,
)


DISPLAY_COLUMNS = [
    "offer_id", "hotel_name", "city", "star_rating", "guest_rating", "room_size_m2",
    "room_type", "month", "distance_center_km", "price_eur", "price_outlier",
]


def csv_export(data: pd.DataFrame) -> bytes:
    """Serialize exactly the active filtered population as a downloadable CSV."""
    return data.to_csv(index=False).encode("utf-8-sig")


def select_offerings(data: pd.DataFrame, offer_ids: list[str]) -> pd.DataFrame:
    """Select records by stable offer ID while preserving the user's chosen order."""
    if "offer_id" not in data.columns:
        raise ValueError("Offer selection requires the stable 'offer_id' column.")
    positions = {str(offer_id): index for index, offer_id in enumerate(offer_ids)}
    selected = data.loc[data["offer_id"].astype(str).isin(positions)].copy()
    selected["_selection_order"] = selected["offer_id"].astype(str).map(positions)
    return selected.sort_values("_selection_order", kind="stable").drop(columns="_selection_order")


def build_html_report(
    data: pd.DataFrame,
    filter_state: Mapping[str, Any],
    summary: Mapping[str, Any] | None = None,
) -> str:
    """Create a compact, self-contained HTML report for the supplied population."""
    summary = dict(summary or summarize_offerings(data))
    filter_rows = []
    for key, value in filter_state.items():
        if value is None or value is False or value == []:
            continue
        formatted = ", ".join(map(str, value)) if isinstance(value, (list, tuple, set)) else str(value)
        filter_rows.append(
            f"<li><strong>{escape(str(key.replace('_', ' ').title()))}:</strong> {escape(formatted)}</li>"
        )
    filter_items = "".join(filter_rows) or "<li>No filters applied</li>"
    summary_rows = "".join(
        f"<tr><th>{escape(key.replace('_', ' ').title())}</th><td>{escape('Unavailable' if value is None else str(value))}</td></tr>"
        for key, value in summary.items()
    )
    city_findings = ""
    if not data.empty and "city" in data and "price_eur" in data:
        cities = price_summaries(data, "city").sort_values("median_price_eur")
        if len(cities) >= 2:
            low, high = cities.iloc[0], cities.iloc[-1]
            city_findings = (
                f"Among matching offers, {escape(str(low['city']))} has the lowest city median "
                f"(€{low['median_price_eur']:.2f}, n={int(low['count'])}); "
                f"{escape(str(high['city']))} has the highest (€{high['median_price_eur']:.2f}, "
                f"n={int(high['count'])}). These are descriptive comparisons."
            )
    if not city_findings:
        city_findings = "Not enough matching city groups to make a city comparison."
    table_columns = [column for column in DISPLAY_COLUMNS if column in data.columns]
    data_table = data.loc[:, table_columns].to_html(index=False, escape=True, border=0)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Hotel Price Analysis</title><style>
body{{font:15px/1.5 system-ui,sans-serif;color:#18333a;background:#f4f7f7;margin:0;padding:2rem}}
main{{max-width:1100px;margin:auto;background:white;padding:2rem;border-radius:12px}}
h1,h2{{color:#126d70}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}td,th{{border-bottom:1px solid #dbe5e5;padding:.5rem;text-align:left}}
th{{background:#f0f6f6}}.scroll{{overflow:auto}}small{{color:#52676c}}
</style></head><body><main><h1>Hotel Price Analysis</h1>
<p>{int(summary.get('offer_count', len(data)))} matching offers in this report.</p>
<h2>Filters</h2><ul>{filter_items}</ul><h2>Summary</h2><table>{summary_rows}</table>
<h2>Finding</h2><p>{city_findings}</p>
<h2>Limitations</h2><p>These records are synthetic learning data. Group comparisons are descriptive, rows from a hotel may repeat, and results do not establish real market prices or causation.</p>
<h2>Matching offers</h2><div class="scroll">{data_table}</div>
<small>Generated locally from the current filtered dashboard population.</small></main></body></html>"""


def _currency(value: Any) -> str:
    return "Unavailable" if value is None or pd.isna(value) else f"€{float(value):,.2f}"


def _empty_message(st: Any, data: pd.DataFrame) -> bool:
    if data.empty:
        st.info("No hotel offerings match the current filters. Adjust or reset the filters to see results.")
        return True
    return False


def render_overview(data: pd.DataFrame, filter_state: Mapping[str, Any]) -> None:
    import plotly.express as px
    import streamlit as st

    st.header("Overview")
    if _empty_message(st, data):
        return
    summary = summarize_offerings(data)
    first, second, third, fourth = st.columns(4)
    first.metric("Matching offers", f"{summary['offer_count']:,}")
    second.metric("Distinct hotels", f"{summary['hotel_count']:,}")
    third.metric("Median nightly price", _currency(summary["median_price_eur"]))
    rating = summary["median_guest_rating"]
    fourth.metric("Median guest rating", "Unavailable" if rating is None else f"{rating:.1f} / 10")
    st.caption(
        f"Guest rating uses {summary['guest_rating_count']:,} recorded ratings; "
        f"room size uses {summary['room_size_count']:,} known values."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Nightly price distribution")
        fig = px.histogram(data, x="price_eur", nbins=30, labels={"price_eur": "Nightly price (€)"})
        fig.update_layout(yaxis_title="Offers", xaxis_title="Nightly price (€)")
        st.plotly_chart(fig, width="stretch", key="overview_price_histogram")
    with right:
        st.subheader("Median price by city")
        cities = price_summaries(data, "city").sort_values("median_price_eur", ascending=False)
        fig = px.bar(
            cities, x="city", y="median_price_eur", color="city", hover_data=["count", "mean_price_eur"],
            labels={"city": "City", "median_price_eur": "Median nightly price (€)", "count": "Offers"},
        )
        fig.update_layout(showlegend=False, yaxis_title="Median nightly price (€)")
        st.plotly_chart(fig, width="stretch", key="overview_city_prices")
    st.subheader("Calculated findings")
    insights = generate_insights(data)
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("There are not enough matching offers in at least two city or month groups to produce a finding.")


def render_explore(data: pd.DataFrame, filter_state: Mapping[str, Any]) -> None:
    import plotly.express as px
    import streamlit as st

    st.header("Explore offers")
    if _empty_message(st, data):
        return
    first, second = st.columns(2)
    with first:
        st.subheader("Room size and nightly price")
        scatter_data = data.dropna(subset=["room_size_m2"])
        if scatter_data.empty:
            st.info("No matching offers have a recorded room size.")
        else:
            fig = px.scatter(
                scatter_data, x="room_size_m2", y="price_eur", color="city",
                hover_data=["offer_id", "hotel_name", "room_type", "month"],
                labels={"room_size_m2": "Room size (m²)", "price_eur": "Nightly price (€)"},
            )
            st.plotly_chart(fig, width="stretch", key="explore_size_price")
    with second:
        st.subheader("Price by star classification")
        fig = px.box(
            data, x="star_rating", y="price_eur", points="outliers",
            labels={"star_rating": "Star classification", "price_eur": "Nightly price (€)"},
        )
        st.plotly_chart(fig, width="stretch", key="explore_star_price")
    st.subheader("Price by month")
    month_data = data.copy()
    month_data["month_label"] = month_data["month"].map(lambda month: f"{int(month):02d}")
    fig = px.box(month_data, x="month_label", y="price_eur", points=False,
                 labels={"month_label": "Month", "price_eur": "Nightly price (€)"})
    st.plotly_chart(fig, width="stretch", key="explore_month_price")
    st.subheader("Pairwise correlations")
    st.caption("Each coefficient uses complete observations for that pair. Numeric identifiers are excluded.")
    correlations = correlation_summary(data)
    if correlations.empty:
        st.info("There are not enough numeric fields to calculate correlations.")
    else:
        st.dataframe(correlations, hide_index=True, width="stretch")

    st.subheader("Context within the filtered population")
    st.caption("These are descriptive group comparisons. They do not show that group membership causes a price difference.")
    city_context, star_context = st.columns(2)
    with city_context:
        st.markdown("**Price by city**")
        st.dataframe(price_summaries(data, "city"), hide_index=True, width="stretch")
    with star_context:
        st.markdown("**Price by star classification**")
        st.dataframe(price_summaries(data, "star_rating"), hide_index=True, width="stretch")

    st.subheader("Outlier sensitivity")
    st.caption(
        "The flagged rows use the original global rule. This comparison filters those flags without recalculating them. "
        "Counts show the price observations and complete room-size/price pairs used."
    )
    sensitivity = outlier_sensitivity_summary(data).rename(
        columns={
            "population": "Population",
            "offer_count": "Offers",
            "price_count": "Valid prices",
            "mean_price_eur": "Mean price (€)",
            "median_price_eur": "Median price (€)",
            "spread_eur": "Price IQR (€)",
            "size_price_pairs": "Room-size/price pairs",
            "room_size_price_correlation": "Room-size/price correlation",
        }
    )
    st.dataframe(sensitivity, hide_index=True, width="stretch")

    st.subheader("Matching offer table")
    search = st.text_input("Search hotel, city, room type, or offer ID", key="table_search")
    table = data.copy()
    if search.strip():
        searchable = [c for c in ("hotel_name", "city", "room_type", "offer_id") if c in table]
        matches = pd.Series(False, index=table.index)
        for column in searchable:
            matches |= table[column].astype("string").str.contains(search.strip(), case=False, na=False, regex=False)
        table = table.loc[matches]
    sort_column = st.selectbox("Sort rows by", [c for c in DISPLAY_COLUMNS if c in table.columns], key="table_sort_column")
    ascending = st.checkbox("Sort ascending", value=True, key="table_sort_ascending")
    if not table.empty:
        table = table.sort_values(sort_column, ascending=ascending, kind="stable", na_position="last")
    st.dataframe(table[[c for c in DISPLAY_COLUMNS if c in table.columns]], hide_index=True, width="stretch")
    st.caption("Use the stable offer ID when choosing an offering; row position and hotel name alone are not unique.")


def render_compare(data: pd.DataFrame, filter_state: Mapping[str, Any]) -> None:
    import streamlit as st

    st.header("Compare offers")
    if _empty_message(st, data):
        return
    options = data["offer_id"].astype(str).tolist()
    names = data.set_index(data["offer_id"].astype(str))["hotel_name"].to_dict()
    selected = st.multiselect(
        "Choose up to three distinct matching offers",
        options=options,
        format_func=lambda offer_id: f"{names.get(offer_id, 'Hotel')} · {offer_id}",
        max_selections=3,
        key="selected_offerings",
    )
    if not selected:
        st.info("Select one or more offers to see their attributes and peer context.")
        return
    chosen = select_offerings(data, selected)
    st.subheader("Side-by-side details")
    columns = [c for c in DISPLAY_COLUMNS if c in chosen.columns]
    details = chosen.set_index("offer_id")[columns[1:]].transpose().astype("string")
    st.dataframe(details, width="stretch")
    st.subheader("Peer price comparison")
    st.caption(
        "Peers share city, star classification, room type, and month within the current filtered population. "
        "The selected offer is excluded; at least five other peer offers are required for a comparison."
    )
    rows = []
    for _, offer in chosen.iterrows():
        comparison = peer_comparison(data, str(offer["offer_id"]))
        enough_peers = comparison["status"] == "sufficient"
        rows.append({
            "Offer ID": offer["offer_id"], "Hotel": offer["hotel_name"],
            "Other peer offers": comparison["peer_count"],
            "Peer median (€)": _currency(comparison["peer_median_eur"]) if enough_peers else "Unavailable",
            "Difference from peer median": (
                f"{comparison['percentage_difference']:+.1f}%" if enough_peers else "Unavailable"
            ),
            "Status": (
                "Sufficient peer evidence" if enough_peers
                else f"Insufficient evidence (need {comparison['minimum_peer_count']} other offers)"
            ),
            "Discussion marker": (
                "Candidate for discussion" if comparison["discussion_candidate"] else "No candidate marker"
            ),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "A discussion candidate has a recorded guest rating of at least 8 and a price below its peer median. "
        "This is a prompt for discussion, not a recommendation or evidence of value."
    )


def render_data_quality(data: pd.DataFrame, audit: Mapping[str, Any], rejected_prices: pd.DataFrame) -> None:
    import streamlit as st

    st.header("Data quality")
    available = audit.get("historical_cleaning_available", False)
    labels = {
        "Source": audit.get("source_kind", "Unavailable"),
        "Rows in current source": audit.get("source_rows"),
        "Exact duplicate copies removed": audit.get("duplicate_copies_removed") if available else None,
        "Invalid price rows separated during cleaning": audit.get("invalid_price_rows_separated") if available else None,
        "Invalid price rows in current source": audit.get("current_invalid_price_rows"),
        "Rows available for price analysis": audit.get("retained_rows"),
        "Current flagged prices": audit.get("flagged_prices"),
    }
    display = pd.DataFrame([
        {"Measure": label, "Value": "Unavailable" if value is None else f"{value:,}" if isinstance(value, int) else value}
        for label, value in labels.items()
    ])
    st.dataframe(display, hide_index=True, width="stretch")
    if not available:
        st.info("Historical duplicate-removal and invalid-price counts are unavailable because the active source is already cleaned.")
    st.markdown("**Cleaning policy**")
    st.markdown(
        "- Exact full-row duplicates are removed; distinct room or month offers from the same hotel are retained.\n"
        "- Rows with missing, nonnumeric, nonfinite, zero, or negative prices are separated from price analysis.\n"
        "- Missing guest ratings and room sizes remain missing.\n"
        "- Positive prices outside global Tukey fences are flagged and retained; a flag does not prove an error."
    )
    st.caption(audit.get("outlier_rule", "Outlier rule unavailable."))
    st.subheader("Missing values in matching offers")
    missing = data.isna().sum().rename("missing values").rename_axis("field").reset_index()
    st.dataframe(missing, hide_index=True, width="stretch")
    if not rejected_prices.empty:
        st.subheader("Rows separated for invalid prices")
        st.dataframe(rejected_prices, hide_index=True, width="stretch")


def render_prediction(data: pd.DataFrame, dataset_id: str) -> None:
    """Show explicit hotel-group evaluation and one-off prediction actions."""
    import streamlit as st

    from src.prediction import evaluate_candidates, make_hotel_split, predict_price

    st.header("Prediction")
    st.warning("Synthetic training demonstration. Predictions are conditional estimates from synthetic data, not real hotel prices.")
    st.caption(
        "The model is trained only when you click Train/Evaluate. Dashboard filters do not retrain it. "
        "Evaluation and prediction use the complete active dataset. No uncertainty interval is available."
    )

    if st.session_state.get("prediction_dataset_fingerprint") not in (None, dataset_id):
        for key in ("prediction_model", "prediction_split", "prediction_evaluation", "prediction_model_name", "prediction_dataset_fingerprint", "prediction_result"):
            st.session_state.pop(key, None)

    candidate_names = ["Linear regression", "Random forest"]
    chosen_model = st.selectbox("Candidate to use for prediction", candidate_names, key="prediction_candidate_choice")
    include_outliers = st.checkbox("Include flagged prices in training", value=True, key="prediction_include_outliers")
    if st.button("Train/Evaluate", key="train_evaluate_prediction", type="primary"):
        split = make_hotel_split(data, dataset_id)
        evaluation = evaluate_candidates(data, split, include_outliers)
        st.session_state["prediction_split"] = split
        st.session_state["prediction_evaluation"] = evaluation
        st.session_state["prediction_model_name"] = chosen_model
        st.session_state["prediction_model"] = evaluation.models.get(chosen_model)
        st.session_state["prediction_dataset_fingerprint"] = dataset_id
        st.session_state.pop("prediction_result", None)

    evaluation = st.session_state.get("prediction_evaluation")
    split = st.session_state.get("prediction_split")
    if evaluation is None or split is None:
        st.info("Choose a candidate and click Train/Evaluate to create a hotel-disjoint holdout evaluation.")
        return
    if evaluation.dataset_id != dataset_id or st.session_state.get("prediction_dataset_fingerprint") != dataset_id:
        st.warning("The active dataset changed. Train/Evaluate again before submitting a prediction.")
        return
    if evaluation.unavailable_reason:
        st.info(evaluation.unavailable_reason)
        return

    st.subheader("Held-out evaluation")
    st.caption(
        f"Dataset identity: {dataset_id[:16]}… · {len(split.train_hotel_ids)} training hotels · "
        f"{len(split.test_hotel_ids)} held-out hotels · {evaluation.metrics['test_rows'].iloc[0]} held-out offers."
    )
    metric_view = evaluation.metrics.rename(columns={
        "model": "Model", "mae_eur": "MAE (€)", "rmse_eur": "RMSE (€)", "r2": "R²",
        "train_rows": "Training offers", "test_rows": "Held-out offers",
        "train_hotel_count": "Training hotels", "test_hotel_count": "Held-out hotels",
    })
    st.dataframe(metric_view[["Model", "MAE (€)", "RMSE (€)", "R²", "Training offers", "Held-out offers", "Training hotels", "Held-out hotels"]], hide_index=True, width="stretch")
    st.caption(f"Prediction candidate: {st.session_state['prediction_model_name']}. Uses that candidate's fitted training-only preprocessing and model from this evaluation.")

    train_rows = data.iloc[list(split.train_indices)]
    numeric_defaults = {
        "star_rating": float(pd.to_numeric(train_rows.get("star_rating"), errors="coerce").median()),
        "guest_rating": float(pd.to_numeric(train_rows.get("guest_rating"), errors="coerce").median()),
        "room_size_m2": float(pd.to_numeric(train_rows.get("room_size_m2"), errors="coerce").median()),
        "month": float(pd.to_numeric(train_rows.get("month"), errors="coerce").median()),
        "distance_center_km": float(pd.to_numeric(train_rows.get("distance_center_km"), errors="coerce").median()),
    }
    categorical_defaults = {
        column: str(train_rows[column].dropna().mode().iloc[0]) if not train_rows[column].dropna().empty else ""
        for column in ("city", "room_type")
    }
    st.subheader("Submit a prediction")
    st.caption("Blank optional characteristics are imputed using training rows. Values outside the training range and unseen categories are flagged below.")
    with st.form("hotel_prediction_form"):
        left, right = st.columns(2)
        inputs: dict[str, Any] = {}
        with left:
            inputs["city"] = st.text_input("City", value=categorical_defaults["city"])
            inputs["star_rating"] = st.number_input("Star classification", value=numeric_defaults["star_rating"], step=1.0)
            missing_rating = st.checkbox("Leave guest rating missing", value=False)
            inputs["guest_rating"] = None if missing_rating else st.number_input("Guest rating (0–10)", value=numeric_defaults["guest_rating"], step=0.1)
            missing_size = st.checkbox("Leave room size missing", value=False)
            inputs["room_size_m2"] = None if missing_size else st.number_input("Room size (m²)", value=numeric_defaults["room_size_m2"], step=1.0)
        with right:
            inputs["room_type"] = st.text_input("Room type", value=categorical_defaults["room_type"])
            inputs["month"] = st.number_input("Month", min_value=1, max_value=12, value=int(numeric_defaults["month"]), step=1)
            inputs["distance_center_km"] = st.number_input("Distance to center (km)", value=numeric_defaults["distance_center_km"], step=0.5)
        submitted = st.form_submit_button("Submit prediction")
    if submitted:
        try:
            result = predict_price(data, split, evaluation, st.session_state["prediction_model_name"], inputs)
            st.session_state["prediction_result"] = result
        except ValueError as error:
            st.error(str(error))
    result = st.session_state.get("prediction_result")
    if result:
        st.metric("Estimated nightly price", _currency(result["predicted_price_eur"]))
        st.caption(f"Model: {result['model_name']} · Training dataset identity: {result['dataset_id'][:16]}…")
        st.write("Supplied characteristic values")
        st.dataframe(pd.DataFrame([result["supplied_values"]]), hide_index=True, width="stretch")
        if result["imputed_fields"]:
            st.info("Values imputed from training rows: " + ", ".join(f"{field}={value}" for field, value in result["imputed_values"].items()))
        if result["unsupported_categories"]:
            st.warning("Categories not observed during training: " + ", ".join(f"{field}={', '.join(values)}" for field, values in result["unsupported_categories"].items()))
        if result["outside_training_ranges"]:
            st.warning("Numeric inputs outside training ranges: " + ", ".join(result["outside_training_ranges"]))
        if not result["imputed_fields"] and not result["unsupported_categories"] and not result["outside_training_ranges"]:
            st.success("All supplied inputs were observed within the training categories and ranges.")


