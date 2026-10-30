"""Part B - delivery-hours regression."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["delivery_hours"]

# 1. split 840 / 180 / 180
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. scaling without leakage (scaler fitted on training rows only)
scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)
pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": pd.Series([len(X_train), len(X_val), len(X_test),
                        round(float(np.abs(train_scaled.mean(axis=0)).max()), 4),
                        round(float(np.abs(test_scaled.mean(axis=0)).max()), 4)], dtype=object),
}).to_csv("outputs/part_b_checks.csv", index=False)


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


# 3. baseline
rows = [{"model": "mean_baseline", "split": "val",
         **metrics(y_val, np.full(len(y_val), y_train.mean()))}]

# 4. tree-depth diagnostic
depth_rows = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    depth_rows.append({
        "max_depth": "unlimited" if d is None else d,
        "train_rmse": np.sqrt(mean_squared_error(y_train, t.predict(X_train))),
        "val_rmse": np.sqrt(mean_squared_error(y_val, t.predict(X_val))),
        "_d": d,
    })
depth_df = pd.DataFrame(depth_rows)
# lowest val_rmse; ties -> smaller depth (unlimited counts as the largest)
depth_df["_order"] = depth_df["_d"].apply(lambda d: 10**6 if pd.isna(d) else d)
best_row = depth_df.sort_values(["val_rmse", "_order"]).iloc[0]
best_depth = None if pd.isna(best_row["_d"]) else int(best_row["_d"])
depth_df[["max_depth", "train_rmse", "val_rmse"]].round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5. four candidates
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name, est in candidates.items():
    est.fit(X_train, y_train)
    rows.append({"model": name, "split": "val", **metrics(y_val, est.predict(X_val))})

# 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    m = metrics(y_val, g.predict(X_val))
    lr_rows.append({"learning_rate": lr, "val_rmse": m["RMSE"], "val_r2": m["R2"]})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. rank, cross-validate the two best, final model on test (once)
val_df = pd.DataFrame(rows[1:]).sort_values("RMSE", kind="stable")
ranked = list(val_df["model"])
top_two = ranked[:2]
kf = KFold(n_splits=5, shuffle=True, random_state=42)   # one object shared by both models
X_trval = pd.concat([X_train, X_val])
y_trval = pd.concat([y_train, y_val])
cv_rows = []
for name in top_two:
    s = cross_val_score(candidates[name], X_trval, y_trval, cv=kf)
    cv_rows.append({"model": name, "mean_r2": s.mean(), "std_r2": s.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final_name = ranked[0]
rows.append({"model": final_name, "split": "test", **metrics(y_test, candidates[final_name].predict(X_test))})
pd.DataFrame(rows)[["model", "split", "MAE", "MSE", "RMSE", "R2"]].round(4).to_csv(
    "outputs/part_b_metrics.csv", index=False)

print("best depth:", best_depth, "| ranking:", ranked, "| final:", final_name)
