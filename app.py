"""Local Streamlit entry point for Hotel Price Intelligence."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import pandas as pd

from src.dashboard_analytics import apply_filters, summarize_offerings
from src.dashboard_data import DatasetValidationError, activate_uploaded_data, load_project_data
from src.dashboard_views import build_html_report, csv_export, render_compare, render_data_quality, render_explore, render_overview, render_prediction


FILTER_WIDGET_KEYS = (
    "filter_cities", "filter_stars", "filter_months", "filter_room_types",
    "filter_price_range", "filter_size_range", "filter_include_unknown", "filter_include_outliers",
)


def _reset_filters() -> None:
    import streamlit as st

    for key in FILTER_WIDGET_KEYS:
        st.session_state.pop(key, None)


def _ensure_project_data(st: Any) -> None:
    if "active_dataset" in st.session_state:
        return
    try:
        loaded = load_project_data()
    except (FileNotFoundError, DatasetValidationError) as error:
        st.error(str(error))
        st.stop()
    st.session_state["active_dataset"] = loaded.data
    st.session_state["active_audit"] = loaded.audit
    st.session_state["active_rejected_prices"] = loaded.rejected_prices
    st.session_state["active_fingerprint"] = loaded.fingerprint
    st.session_state["active_upload_bytes"] = None


def _filters_sidebar(st: Any, data: pd.DataFrame) -> dict[str, Any]:
    st.sidebar.header("Filters")
    st.sidebar.button("Reset filters", on_click=_reset_filters, width="stretch")
    cities = sorted(data["city"].dropna().astype(str).unique().tolist())
    stars = sorted(int(value) for value in data["star_rating"].dropna().unique())
    months = sorted(int(value) for value in data["month"].dropna().unique())
    room_types = sorted(data["room_type"].dropna().astype(str).unique().tolist())
    chosen_cities = st.sidebar.multiselect("City", cities, key="filter_cities")
    chosen_stars = st.sidebar.multiselect("Star classification", stars, key="filter_stars")
    chosen_months = st.sidebar.multiselect("Month", months, key="filter_months")
    chosen_rooms = st.sidebar.multiselect("Room type", room_types, key="filter_room_types")

    price_values = pd.to_numeric(data["price_eur"], errors="coerce").dropna()
    price_low, price_high = float(price_values.min()), float(price_values.max())
    if price_low < price_high:
        selected_price = st.sidebar.slider(
            "Nightly price (€)", min_value=price_low, max_value=price_high,
            value=(price_low, price_high), key="filter_price_range", format="€%.2f",
        )
    else:
        selected_price = (price_low, price_high)
        st.sidebar.caption(f"All matching prices currently equal €{price_low:,.2f}.")
    size_values = pd.to_numeric(data["room_size_m2"], errors="coerce").dropna()
    if size_values.empty:
        size_low = size_high = None
        selected_size = None
        size_active = False
        st.sidebar.caption("No matching source rows have a recorded room size.")
    else:
        size_low, size_high = float(size_values.min()), float(size_values.max())
        if size_low < size_high:
            selected_size = st.sidebar.slider(
                "Room size (m²)", min_value=size_low, max_value=size_high,
                value=(size_low, size_high), key="filter_size_range", format="%.1f",
            )
        else:
            selected_size = (size_low, size_high)
            st.sidebar.caption(f"All recorded room sizes currently equal {size_low:.1f} m².")
        size_active = selected_size != (size_low, size_high)
    include_unknown = False
    if size_active:
        include_unknown = st.sidebar.checkbox(
            "Include offers with unknown room size", value=False, key="filter_include_unknown"
        )
    include_outliers = st.sidebar.checkbox(
        "Include flagged prices", value=True, key="filter_include_outliers",
        help="Outlier flags come from the original global rule and are not recalculated by these filters.",
    )

    return {
        "cities": chosen_cities,
        "star_ratings": chosen_stars,
        "months": chosen_months,
        "room_types": chosen_rooms,
        "price_min": selected_price[0] if selected_price != (price_low, price_high) else None,
        "price_max": selected_price[1] if selected_price != (price_low, price_high) else None,
        "room_size_min": selected_size[0] if size_active else None,
        "room_size_max": selected_size[1] if size_active else None,
        "include_unknown_room_size": include_unknown,
        "include_outliers": include_outliers,
    }


def _active_filter_labels(state: dict[str, Any]) -> list[str]:
    labels = []
    names = {"cities": "City", "star_ratings": "Stars", "months": "Month", "room_types": "Room type"}
    for key, label in names.items():
        if state[key]:
            labels.append(f"{label}: {', '.join(map(str, state[key]))}")
    if state["price_min"] is not None or state["price_max"] is not None:
        labels.append(f"Price range: {state['price_min']} to {state['price_max']} EUR")
    if state["room_size_min"] is not None or state["room_size_max"] is not None:
        value = f"Room size: {state['room_size_min']} to {state['room_size_max']} m²"
        labels.append(value + (" (unknown included)" if state["include_unknown_room_size"] else ""))
    if not state["include_outliers"]:
        labels.append("Flagged prices excluded")
    return labels or ["No filters applied"]


def render_app() -> None:
    """Own app navigation, active dataset, shared filters, and downloads."""
    import streamlit as st

    st.set_page_config(page_title="Hotel Price Intelligence", page_icon="🏨", layout="wide")
    _ensure_project_data(st)
    st.title("Hotel Price Intelligence")
    st.caption("Explore synthetic hotel offers for learning. The data does not describe real hotel prices.")

    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Replace active data with a CSV", type=["csv"], key="hotel_csv_upload")
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        upload_fingerprint = sha256(file_bytes).hexdigest()
        if st.session_state.get("last_upload_attempt") != upload_fingerprint:
            st.session_state["last_upload_attempt"] = upload_fingerprint
            try:
                activate_uploaded_data(st.session_state, file_bytes)
                st.session_state["active_upload_bytes"] = file_bytes
                st.session_state["upload_error"] = None
            except DatasetValidationError as error:
                st.session_state["upload_error"] = str(error)
        if st.session_state.get("upload_error"):
            st.sidebar.error(st.session_state["upload_error"])
        else:
            st.sidebar.success("Uploaded data is active for this session.")

    active_data = st.session_state["active_dataset"]
    filter_state = _filters_sidebar(st, active_data)
    filtered = apply_filters(active_data, filter_state)
    summary = summarize_offerings(filtered)
    st.caption(f"**Matching offers:** {summary['offer_count']:,} · **Filters:** {'; '.join(_active_filter_labels(filter_state))}")

    page = st.sidebar.radio("Dashboard page", ["Overview", "Explore", "Compare", "Data quality", "Prediction"], key="dashboard_page")
    if page == "Overview":
        render_overview(filtered, filter_state)
    elif page == "Explore":
        render_explore(filtered, filter_state)
    elif page == "Compare":
        render_compare(filtered, filter_state)
    elif page == "Data quality":
        render_data_quality(filtered, st.session_state.get("active_audit", {}),
                            st.session_state.get("active_rejected_prices", pd.DataFrame()))
    else:
        render_prediction(active_data, st.session_state["active_fingerprint"])

    st.divider()
    st.subheader("Download this filtered population")
    csv_bytes = csv_export(filtered)
    report = build_html_report(filtered, filter_state, summary)
    left, right = st.columns(2)
    left.download_button(
        "Download filtered CSV", data=csv_bytes, file_name="hotel_offers_filtered.csv",
        mime="text/csv", key="download_filtered_csv",
    )
    right.download_button(
        "Download HTML analysis report", data=report.encode("utf-8"),
        file_name="hotel_price_analysis.html", mime="text/html", key="download_analysis_html",
    )


if __name__ == "__main__":
    render_app()

