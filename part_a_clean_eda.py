"""Part A - clean the raw export, build features, write the modelling table and EDA outputs."""
import os
import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# ---- "before" measurements (after dropping duplicates, before any filling) ----
dedup = raw.drop_duplicates()
sentinel_rows = int((dedup["parcel_weight_kg"] > 100).sum())
spellings_before = int(dedup["service_type"].nunique())

# ---- clean + features + target ----
clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv("data/modelling_table.csv", index=False)

# ---- checks file ----
q1, q3 = table["delivery_hours"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = int(((table["delivery_hours"] < lower) | (table["delivery_hours"] > upper)).sum())

checks = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(dedup["parcel_weight_kg"].isna().sum())),
    ("sentinel_weight_rows", sentinel_rows),
    ("missing_declared_value_before", int(dedup["declared_value"].isna().sum())),
    ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", spellings_before),
    ("service_type_spellings_after", int(table["service_type"].nunique())),
    ("declared_value_mean_inr", table["declared_value_inr"].mean()),
    ("declared_value_median_inr", table["declared_value_inr"].median()),
    ("log_declared_value_mean", np.log(table["declared_value_inr"]).mean()),
    ("log_declared_value_median", np.log(table["declared_value_inr"]).median()),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(table)),
    ("nulls_in_features_and_target", int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(table["is_delayed"].sum())),
    ("delay_rate", table["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower),
    ("hours_upper_bound", upper),
    ("hours_outlier_count", outliers),
]
# keep counts as integers and round decimals to 4 places
rows = [(m, int(v) if isinstance(v, (int, np.integer)) else round(float(v), 4)) for m, v in checks]
pd.DataFrame({"metric": [m for m, _ in rows], "value": pd.Series([v for _, v in rows], dtype=object)}).to_csv(
    "outputs/part_a_checks.csv", index=False)

# ---- delay evidence by group ----
table["distance_band"] = pd.cut(
    table["distance_km"], bins=[0, 200, 400, 600, 2000], right=True,
    labels=["0-200", "201-400", "401-600", "601+"],
)
frames = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = table.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index().rename(columns={dim: "group"})
    g.insert(0, "dimension", dim)
    g["group"] = g["group"].astype(str)
    frames.append(g)
pd.concat(frames, ignore_index=True).round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# ---- correlation matrix ----
corr = table.drop(columns=["distance_band"]).corr(numeric_only=True).round(4)
corr.to_csv("outputs/part_a_correlations.csv", index=True, index_label="column")

print("Part A done:", len(table), "rows written to data/modelling_table.csv")
