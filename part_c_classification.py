"""Part C - delay classifier, feature selection, tuning and alert threshold."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]

# 1. stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "| delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cell_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def write(df, path):
    df.round(4).to_csv(path, index=False)


# 2. blind baseline + three classifiers (default 0.5 rule)
rows = [{"model": "blind_all_on_time", **cell_metrics(y_val, np.zeros(len(y_val), dtype=int))}]
rows[0]["precision"] = 0

logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
rows.append({"model": "logistic_regression", **cell_metrics(y_val, logit.predict(X_val))})

best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:        # ascending, strict '>' keeps the smaller d on ties
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f1:
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
rows.append({"model": f"decision_tree_depth_{best_d}", **cell_metrics(y_val, tree.predict(X_val))})

rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append({"model": "random_forest_default", **cell_metrics(y_val, rf_default.predict(X_val))})
write(pd.DataFrame(rows), "outputs/part_c_models.csv")

# 3. which columns carry signal?
sel_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sel_all.scores_,
                   "rf_importance": rf_default.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs.sort_values("f_rank")[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
write(fs, "outputs/part_c_feature_scores.csv")

k_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)),
                  ("rf", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    k_rows.append({"k": k, "val_f1": f1_score(y_val, p.predict(X_val))})
write(pd.DataFrame(k_rows), "outputs/part_c_selectk.csv")

# 4. tuning on k = 7 (selector fitted on training rows only)
selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
Xtr7, Xva7 = selector.transform(X_train), selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}


def py(params):
    return {k: (None if v is None else int(v)) for k, v in params.items()}


gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5,
                        random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)

tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 27 * 5,
     "best_params": json.dumps(py(gs.best_params_), sort_keys=True),
     "best_cv_accuracy": gs.best_score_, "val_f1": f1_score(y_val, gs.best_estimator_.predict(Xva7))},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 9 * 5,
     "best_params": json.dumps(py(rs.best_params_), sort_keys=True),
     "best_cv_accuracy": rs.best_score_, "val_f1": f1_score(y_val, rs.best_estimator_.predict(Xva7))},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": json.dumps({"max_depth": None, "min_samples_leaf": 1, "n_estimators": 100}, sort_keys=True),
     "best_cv_accuracy": np.nan, "val_f1": f1_score(y_val, default7.predict(Xva7))},
])
write(tuning, "outputs/part_c_tuning.csv")

# 5. final configuration: highest val_f1; ties -> default, grid, randomized
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
tuning["_p"] = tuning["search"].map(priority)
tuning["_f1r"] = tuning["val_f1"].round(4)
winner = tuning.sort_values(["_f1r", "_p"], ascending=[False, True]).iloc[0]
params = json.loads(winner["best_params"])
final = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                  ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_train, y_train)

# 6. alert threshold from validation probabilities
p_val = final.predict_proba(X_val)[:, 1]
th_rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    th_rows.append({"threshold": t, **cell_metrics(y_val, (p_val >= t).astype(int))})
th = pd.DataFrame(th_rows)
write(th, "outputs/part_c_thresholds.csv")
ok = th[th["recall"] >= 0.80]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1

# 7. one test evaluation + config
p_test = final.predict_proba(X_test)[:, 1]
write(pd.DataFrame([{"threshold": threshold, **cell_metrics(y_test, (p_test >= threshold).astype(int))}]),
      "outputs/part_c_final_test.csv")
write(pd.DataFrame([{
    "selector_k": 7, "n_estimators": params["n_estimators"],
    "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
    "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
    "chosen_from": winner["search"]}]), "outputs/part_c_final_config.csv")
print("chosen:", winner["search"], params, "threshold", threshold)
