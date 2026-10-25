# Step 3: Machine Learning - predict Land Surface Temperature (LST)
#   Part 1  (Practical 3)  : LabelEncoder, train_test_split, LinearRegression, MAE/MSE/RMSE/R2,
#                            actual vs predicted, residual plot, coefficients, GridSearchCV
#   Part 2  (Practical 4)  : StandardScaler, KNeighborsRegressor
#   Part 3  (comparison)   : Linear Regression, KNN + Decision Tree, Random Forest, SVR, Gradient Boosting
# Run:  python 03_train_models.py

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# ================================================================ load data
df = pd.read_csv("data/processed/lst_long_clean.csv", parse_dates=["date"])
df.shape
df.head()
df.info()
df.describe()
df.isnull().sum()

# remove the 28 flagged LST values and keep the columns we need
df = df[df["lst_outlier"] == False]
year = df["date"].dt.year.values                      # used for the unseen-year test (2025)
df = df[["city", "zone", "season", "month", "no2_umol_m2", "wind_u", "wind_v", "lst_c"]]

# month -> season (used by the prediction form on the website)
season_of_month = df.groupby("month")["season"].first().to_dict()

# ---------------------------------------------------------------- LabelEncoder (Practical 3)
encoders = {}
categorical_columns = ("city", "zone", "season")
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------------------------------------------------------- x, y and train / test split
x = df.drop("lst_c", axis=1)
y = df["lst_c"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42)

print("Training data :", x_train.shape)
print("Testing data  :", x_test.shape)

# ================================================================ PART 1: Linear Regression (Practical 3)
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\n--- Linear Regression (Practical 3) ---")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.title("Actual vs Predicted")
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("outputs/figures/10_linear_actual_vs_predicted.png", dpi=150)
plt.close()

residual = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residual)
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.savefig("outputs/figures/11_linear_residual_plot.png", dpi=150)
plt.close()

coef = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient": model.coef_,
})
print(coef.sort_values(by="Coefficient", ascending=False))
coef.sort_values(by="Coefficient", ascending=False).to_csv("outputs/metrics/linear_regression_coefficients.csv", index=False)

# GridSearchCV (Practical 3a)
param_grid = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'positive': [True, False]
}

grid = GridSearchCV(
    estimator=LinearRegression(),
    param_grid=param_grid,
    cv=5,
    scoring='r2'
)

grid.fit(x_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross Validation R2 Score:", grid.best_score_)

# ================================================================ PART 2: KNN Regression (Practical 4)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train_scaled, y_train)

y_pred_knn = knn.predict(x_test_scaled)

mae = mean_absolute_error(y_test, y_pred_knn)
mse = mean_squared_error(y_test, y_pred_knn)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_knn)
print("\n--- KNN Regression, k=5 (Practical 4) ---")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.title("KNN Regression Actual vs Predicted")
plt.scatter(y_test, y_pred_knn)
plt.plot(y_test, y_test, color="green")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("outputs/figures/12_knn_actual_vs_predicted.png", dpi=150)
plt.close()

# ================================================================ PART 3: model comparison
def get_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# every model gets GridSearchCV (cv=5, scoring='r2') on the scaled training data
models = {
    "Linear Regression": (LinearRegression(), {}),
    "KNN Regression": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 9, 15], "weights": ["uniform", "distance"]}),
    "Decision Tree": (DecisionTreeRegressor(random_state=42), {"max_depth": [4, 8, 12, None], "min_samples_leaf": [5, 20]}),
    "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=42), {"max_depth": [8, None], "min_samples_leaf": [2, 5]}),
    "SVR": (SVR(), {"C": [1, 10, 50], "epsilon": [0.1, 0.5]}),
    "Gradient Boosting": (HistGradientBoostingRegressor(random_state=42), {"learning_rate": [0.05, 0.1], "max_depth": [3, 5, None]}),
}

# unseen-year test: train on 2020-2024 and test on 2025
train_time = year <= 2024
test_time = year == 2025
scaler_time = StandardScaler()
xt_train = scaler_time.fit_transform(x[train_time])
xt_test = scaler_time.transform(x[test_time])

rows = []
fitted = {}
predictions = pd.DataFrame({
    "city": encoders["city"].inverse_transform(x_test["city"]),
    "zone": encoders["zone"].inverse_transform(x_test["zone"]),
    "Actual": y_test.values,
})

for name, (est, grid_params) in models.items():
    g = GridSearchCV(estimator=est, param_grid=grid_params, cv=5, scoring="r2")
    g.fit(x_train_scaled, y_train)
    best_estimator = g.best_estimator_

    y_pred_m = best_estimator.predict(x_test_scaled)
    result = get_metrics(y_test, y_pred_m)

    time_model = clone(best_estimator)
    time_model.fit(xt_train, y[train_time])
    time_result = get_metrics(y[test_time], time_model.predict(xt_test))

    print(f"{name}: CV R2 = {g.best_score_:.3f}, test R2 = {result['R2']:.3f}, 2025 R2 = {time_result['R2']:.3f}, best = {g.best_params_}")

    rows.append({"Model": name, "Best Parameters": json.dumps(g.best_params_, default=str), "CV R2": g.best_score_,
                 **result, "RMSE 2025": time_result["RMSE"], "R2 2025": time_result["R2"]})
    fitted[name] = best_estimator
    predictions[name] = y_pred_m

results = pd.DataFrame(rows)
print("\n", results[["Model", "CV R2", "MAE", "RMSE", "R2", "RMSE 2025", "R2 2025"]].round(3).to_string(index=False))

# the model with the highest cross-validated R2 is used on the website
best_name = results.loc[results["CV R2"].idxmax(), "Model"]
best_model = fitted[best_name]
best_result = results[results["Model"] == best_name].iloc[0]
print("\nBest model:", best_name)

# ---------------------------------------------------------------- analysis of the best model
y_pred_best = best_model.predict(x_test_scaled)
importance = permutation_importance(best_model, x_test_scaled, y_test, scoring="neg_root_mean_squared_error",
                                    n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({
    "feature": x.columns,
    "importance_rmse_increase": importance.importances_mean,
    "std": importance.importances_std,
}).sort_values(by="importance_rmse_increase", ascending=False)
print(feature_importance)

# ---------------------------------------------------------------- save files
results.to_csv("outputs/metrics/model_results.csv", index=False)
predictions.to_csv("outputs/metrics/predictions.csv", index=False)
feature_importance.to_csv("outputs/metrics/feature_importance.csv", index=False)

joblib.dump({
    "model": best_model, "scaler": scaler, "encoders": encoders,
    "columns": list(x.columns), "season_of_month": season_of_month, "name": best_name,
}, "outputs/models/best_model.joblib")

with open("outputs/metrics/best_model.json", "w") as f:
    json.dump({
        "model_name": best_name,
        "metrics": {k: float(best_result[k]) for k in ("MAE", "MSE", "RMSE", "R2")},
        "cv_r2": float(best_result["CV R2"]),
        "unseen_year_2025": {"RMSE": float(best_result["RMSE 2025"]), "R2": float(best_result["R2 2025"])},
        "split": "random 80/20, random_state=42",
        "selection_rule": "highest cross-validated R2 (GridSearchCV, cv=5)",
        "train_rows": int(len(x_train)), "test_rows": int(len(x_test)),
    }, f, indent=2)

# ---------------------------------------------------------------- plots
plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["R2"])
plt.xticks(rotation=20)
plt.ylabel("R2 Score")
plt.title("Model comparison (test set)")
plt.tight_layout()
plt.savefig("outputs/figures/13_model_comparison.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.barh(feature_importance["feature"], feature_importance["importance_rmse_increase"])
plt.xlabel("Increase in RMSE when the feature is shuffled")
plt.title(f"Feature importance ({best_name})")
plt.tight_layout()
plt.savefig("outputs/figures/14_feature_importance.png", dpi=150)
plt.close()
