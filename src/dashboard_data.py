"""Session-safe loading, validation, identity, and upload handling."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any, MutableMapping

import numpy as np
import pandas as pd

from src.clean_data import DEFAULT_INPUT, DEFAULT_OUTPUT, REQUIRED_COLUMNS, clean_dataset


@dataclass(frozen=True)
class DatasetLoad:
    """Validated active data and source information for the dashboard."""

    data: pd.DataFrame
    audit: dict[str, Any]
    rejected_prices: pd.DataFrame
    fingerprint: str


class DatasetValidationError(ValueError):
    """Raised when uploaded or project data cannot be used by the dashboard."""


def dataset_fingerprint(data: pd.DataFrame) -> str:
    """Hash dataset content independently of its row order or DataFrame index."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise DatasetValidationError(f"Cannot identify dataset; missing columns: {', '.join(missing)}")

    frame = data.loc[:, REQUIRED_COLUMNS]
    try:
        row_hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(dtype=np.uint64)
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(f"Could not calculate a stable dataset identity: {error}") from error

    payload = {
        "columns": REQUIRED_COLUMNS,
        "row_hashes": sorted(int(value) for value in row_hashes),
    }
    return sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def add_offering_ids(data: pd.DataFrame) -> pd.DataFrame:
    """Add a stable content-based ID before the UI sorts or filters rows."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise DatasetValidationError(f"Cannot identify offerings; missing columns: {', '.join(missing)}")

    result = data.reset_index(drop=True).copy()
    row_hashes = pd.util.hash_pandas_object(
        result.loc[:, REQUIRED_COLUMNS], index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    occurrence_counts: dict[int, int] = {}
    offer_ids: list[str] = []
    for raw_hash in row_hashes:
        row_hash = int(raw_hash)
        occurrence = occurrence_counts.get(row_hash, 0)
        occurrence_counts[row_hash] = occurrence + 1
        offer_ids.append(f"offer_{row_hash:016x}_{occurrence:03d}")
    result.insert(0, "offer_id", offer_ids)
    return result


def _read_csv_bytes(file_bytes: bytes, *, source_name: str) -> pd.DataFrame:
    if not file_bytes:
        raise DatasetValidationError(f"{source_name} is empty. Choose a CSV file containing hotel offerings.")
    try:
        return pd.read_csv(BytesIO(file_bytes))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise DatasetValidationError(
            f"Could not read {source_name} as a CSV. Check its encoding and row formatting."
        ) from error


def _build_result(raw_data: pd.DataFrame, *, source_kind: str, historical_audit: bool) -> DatasetLoad:
    missing = [column for column in REQUIRED_COLUMNS if column not in raw_data.columns]
    if missing:
        raise DatasetValidationError(
            f"CSV is missing required columns: {', '.join(missing)}. "
            "Upload the hotel-offering schema listed in the project README."
        )

    try:
        cleaned, rejected, cleaning_audit = clean_dataset(raw_data.loc[:, REQUIRED_COLUMNS].copy())
    except (ValueError, TypeError, KeyError) as error:
        raise DatasetValidationError(f"Hotel data failed validation: {error}") from error

    # A cleaned-only source cannot tell us what happened before it was saved.
    audit: dict[str, Any] = {
        "source_kind": source_kind,
        "historical_cleaning_available": bool(historical_audit),
        "generator_audit_available": False,
        "source_rows": int(len(raw_data)),
        "duplicate_copies_removed": (
            int(cleaning_audit["duplicate_copies_removed"]) if historical_audit else None
        ),
        "invalid_price_rows_separated": (
            int(cleaning_audit["invalid_price_rows_separated"]) if historical_audit else None
        ),
        "current_invalid_price_rows": int(len(rejected)),
        "retained_rows": int(len(cleaned)),
        "remaining_missing_values": {
            column: int(count) for column, count in cleaning_audit["remaining_missing_values"].items()
        },
        "flagged_prices": int(cleaning_audit["flagged_prices"]),
        "outlier_rule": cleaning_audit["outlier_rule"],
        "outlier_fences_eur": cleaning_audit["outlier_fences_eur"],
        "row_counts_reconcile": bool(cleaning_audit["row_counts_reconcile"]),
    }
    fingerprint = dataset_fingerprint(cleaned)
    identified = add_offering_ids(cleaned)
    return DatasetLoad(
        data=identified,
        audit=audit,
        rejected_prices=rejected.reset_index(drop=True),
        fingerprint=fingerprint,
    )


def validate_upload(file_bytes: bytes) -> DatasetLoad:
    """Validate an uploaded CSV and return its cleaned, session-ready data."""
    source = _read_csv_bytes(file_bytes, source_name="Uploaded file")
    return _build_result(source, source_kind="upload", historical_audit=True)


def load_project_data(
    raw_path: Path = DEFAULT_INPUT,
    cleaned_path: Path = DEFAULT_OUTPUT,
) -> DatasetLoad:
    """Load existing project data, preferring raw data when it is available."""
    raw_path = Path(raw_path)
    cleaned_path = Path(cleaned_path)

    if raw_path.exists():
        try:
            source = pd.read_csv(raw_path)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            raise DatasetValidationError(f"Could not read project raw CSV at {raw_path}: {error}") from error
        return _build_result(source, source_kind="raw", historical_audit=True)

    if cleaned_path.exists():
        try:
            source = pd.read_csv(cleaned_path)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            raise DatasetValidationError(f"Could not read project cleaned CSV at {cleaned_path}: {error}") from error
        return _build_result(source, source_kind="cleaned", historical_audit=False)

    raise FileNotFoundError(
        f"No hotel CSV was found. Expected {raw_path} or {cleaned_path}. "
        "Generate the project data with python src/generate_data.py."
    )


def activate_uploaded_data(
    session_state: MutableMapping[str, Any],
    file_bytes: bytes,
) -> DatasetLoad:
    """Validate first, then replace active session data and clear stale state."""
    result = validate_upload(file_bytes)
    previous_fingerprint = session_state.get("active_fingerprint")

    if previous_fingerprint != result.fingerprint:
        for key in (
            "comparison_selection",
            "selected_offerings",
            "model_state",
            "trained_model",
            "model_evaluation",
            "prediction_result",
            "prediction_model",
            "prediction_split",
            "prediction_evaluation",
            "prediction_model_name",
            "prediction_dataset_fingerprint",
            "prediction_inputs",
            "filter_state",
            "dashboard_filters",
            "filter_cities",
            "filter_stars",
            "filter_months",
            "filter_room_types",
            "filter_price_range",
            "filter_size_range",
            "filter_include_unknown",
            "filter_include_outliers",
        ):
            session_state.pop(key, None)

    session_state["active_dataset"] = result.data
    session_state["active_audit"] = result.audit
    session_state["active_rejected_prices"] = result.rejected_prices
    session_state["active_fingerprint"] = result.fingerprint
    return result
