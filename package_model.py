"""Part D (task 5) - rebuild the final Part C pipeline, save it and verify the reload."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])
threshold = float(cfg["threshold"])

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
    ("model", RandomForestClassifier(random_state=42, n_estimators=int(cfg["n_estimators"]),
                                     max_depth=max_depth, min_samples_leaf=int(cfg["min_samples_leaf"]))),
]).fit(X_train, y_train)

joblib.dump(model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
pd.DataFrame({"column": list(medians.keys()), "median": [round(v, 4) for v in medians.values()]}).to_csv(
    "models/impute_values.csv", index=False)

reloaded = joblib.load("models/delay_model.pkl")
diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - model.predict_proba(X_test)[:, 1]).max()
pd.DataFrame({"metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
              "value": pd.Series([len(X_test), round(float(diff), 4), threshold], dtype=object)}).to_csv(
    "outputs/part_d_reload_check.csv", index=False)
print("packaged; reload diff", diff)
