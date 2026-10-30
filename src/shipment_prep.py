"""Shared constants and preprocessing for the Delay Intelligence pipeline.

Used by Parts A-D and by predict_delay.py so that training and scoring
apply exactly the same cleaning and feature logic.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_LIMIT = 100  # network accepts parcels up to 40 kg; anything above 100 is a scanner fault


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export.

    Returns (clean_df, medians). If `medians` is None they are computed from the
    non-missing values left after steps 1-6; otherwise the dict passed in is used.
    """
    df = raw.copy()

    # i. exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # ii. service_type: strip + title case
    df["service_type"] = df["service_type"].str.strip().str.title()

    # iii. payment_mode: missing -> "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # iv. declared_value: "₹1,290" -> 1290.0 in declared_value_inr
    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"].astype("string").str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    ).astype(float)
    df = df.drop(columns=["declared_value"])

    # v. dispatch_time -> datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # vi. weight sentinel -> missing
    df["parcel_weight_kg"] = df["parcel_weight_kg"].astype(float)
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_LIMIT, "parcel_weight_kg"] = np.nan

    # vii. median fill
    if medians is None:
        medians = {c: float(df[c].median()) for c in MEDIAN_COLUMNS}
    else:
        medians = {c: float(medians[c]) for c in MEDIAN_COLUMNS}
    for c in MEDIAN_COLUMNS:
        df[c] = df[c].fillna(medians[c])

    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and derive model features. Never touches delivery_hours."""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = clean.merge(hubs[hub_cols], on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
