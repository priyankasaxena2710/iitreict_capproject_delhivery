"""Part D (tasks 1-4) - hub profile, K-Means segments, separation evidence, tier check."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. hub profile (groupby sorts by hub_id ascending: HUB01 ... HUB24)
profile = table.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).reset_index().sort_values("hub_id").reset_index(drop=True)
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. K-Means on the scaled profile
CLUSTER_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(profile[CLUSTER_COLS])
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)
profile["cluster"] = km.labels_

# 3. separation evidence: mean across-cluster pair distance / mean within-cluster pair distance
dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
iu = np.triu_indices(len(Z), k=1)                       # 276 unordered pairs
sep_rows = []
for K in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Z).labels_
    same = (labels[:, None] == labels[None, :])[iu]
    d = dist[iu]
    sizes = sorted(np.bincount(labels), reverse=True)
    sep_rows.append({"K": K, "across_over_within": d[~same].mean() / d[same].mean(),
                     "cluster_sizes": "-".join(str(int(s)) for s in sizes)})
pd.DataFrame(sep_rows).round(4).to_csv("outputs/part_d_separation.csv", index=False)

# 4. name segments by mean delay_rate of their hubs: lowest -> Standard, middle -> Watch, highest -> Escalate
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

seg = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg.to_csv("outputs/hub_segments.csv", index=False)

seg_profile = profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean"),
).reindex(["Standard", "Watch", "Escalate"]).reset_index()
seg_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = seg[seg["segment"] != seg["hub_tier"].map(implied)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print("segments done; mismatches:", list(mismatch["hub_id"]))
