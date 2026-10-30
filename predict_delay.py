"""Part D (task 6) - score new parcels and turn them into an action list."""
import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments

new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = list(pd.read_csv("models/feature_order.csv")["feature"])
impute = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute["column"], impute["median"]))
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new_raw, medians=medians)       # same cleaning as training, stored medians
feats = build_features(clean, hubs)
proba = model.predict_proba(feats[feature_order])[:, 1]

out = feats[["shipment_id", "hub_id"]].copy()
out["delay_probability"] = pd.Series(proba).round(4).values
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
out = out.merge(segments, on="hub_id", how="left")


def action(row):
    if row["flagged"] == 1 and row["segment"] == "Escalate":
        return "Escalate to hub manager"
    if row["flagged"] == 1:
        return "Proactive customer call"
    return "No action"


out["action"] = out.apply(action, axis=1)
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(f"{len(out)} parcels scored; flagged {int(out['flagged'].sum())}; "
      f"escalations {int((out['action'] == 'Escalate to hub manager').sum())}")
