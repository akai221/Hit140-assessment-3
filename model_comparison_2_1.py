# HIT140 Assessment 3 - Linear Regression 2.1
# Compares linear regression with Ridge, Lasso, Random Forest and Gradient Boosting
# using repeated 5-fold cross-validation (10 repeats, seed 42) on all 104 matches.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

df = pd.read_csv("LR_2.1_dataset_clean.csv")
x_cols = ["fifa_rank_diff", "attack_diff", "defence_diff", "h2h_avg_gd",
          "squad_value_diff_eur_m", "squad_age_diff", "venue_altitude_m",
          "host_advantage_diff"]
X, y = df[x_cols], df["goal_difference"]

# variance inflation factors (collinearity)
vif = dict(zip(x_cols, np.diag(np.linalg.inv(X.corr().values))))
print("VIF:", {k: round(v, 2) for k, v in vif.items()})

# signed log transform for squad value difference (can be negative)
def signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))

value_idx = [x_cols.index("squad_value_diff_eur_m")]
log_value = ColumnTransformer(
    [("log", FunctionTransformer(signed_log), value_idx)], remainder="passthrough")

models = {
    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Linear + log value": make_pipeline(log_value, StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
    "Lasso": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=20000)),
    "Random forest": RandomForestRegressor(n_estimators=300, max_depth=3,
                                           min_samples_leaf=5, random_state=42),
    "Gradient boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                   max_depth=2, subsample=0.8, random_state=42),
}

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
rows = []
for name, m in models.items():
    s = cross_validate(m, X, y, cv=cv, scoring=scoring)
    rows.append({"model": name,
                 "rmse": -s["test_rmse"].mean(), "rmse_sd": s["test_rmse"].std(),
                 "mae": -s["test_mae"].mean(), "r2": s["test_r2"].mean(),
                 "r2_sd": s["test_r2"].std()})
res = pd.DataFrame(rows).round(3)
res["nrmse"] = (res["rmse"] / (y.max() - y.min())).round(3)
print(res.to_string(index=False))
res.to_csv("model_comparison_2_1_results.csv", index=False)

# which variables Lasso keeps, and Ridge/Lasso standardised coefficients on all data
lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=20000)).fit(X, y)
print("Lasso alpha:", round(lasso[-1].alpha_, 4))
print("Lasso coefficients:", dict(zip(x_cols, lasso[-1].coef_.round(3))))
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(X, y)
print("Ridge alpha:", round(ridge[-1].alpha_, 3))
rf = models["Random forest"].fit(X, y)
print("RF importance:", dict(zip(x_cols, rf.feature_importances_.round(3))))

# figure: (a) correlation with goal difference, (b) cross-validated RMSE by model
labels = {"fifa_rank_diff": "FIFA rank diff", "attack_diff": "Attack diff",
          "defence_diff": "Defence diff", "h2h_avg_gd": "Head-to-head GD",
          "squad_value_diff_eur_m": "Squad value diff", "squad_age_diff": "Squad age diff",
          "venue_altitude_m": "Altitude", "host_advantage_diff": "Host advantage"}
corr = df[x_cols].corrwith(y).sort_values()
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.35))
cols = ["#9A9A9A" if v < 0 else "#333333" for v in corr.values]
ax[0].barh([labels[i] for i in corr.index], corr.values, color=cols)
ax[0].axvline(0, color="#444", lw=0.6)
ax[0].set_xlabel("Correlation with goal difference")
ax[0].set_title("(a) Single-variable relationships", fontsize=8.5, loc="left")
for i, v in enumerate(corr.values):
    ax[0].text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.2f}", va="center",
               ha="left" if v >= 0 else "right", fontsize=7)
ax[0].set_xlim(-0.45, 0.8)
r = res.iloc[::-1].reset_index(drop=True)
ypos = np.arange(len(r))
ax[1].errorbar(r["rmse"], ypos, xerr=r["rmse_sd"], fmt="o", color="#222222",
               ecolor="#9A9A9A", capsize=2, ms=4)
ax[1].set_yticks(ypos)
ax[1].set_yticklabels(r["model"])
ax[1].axvline(r["rmse"].iloc[-1], color="#999", lw=0.6, ls="--")
ax[1].set_xlim(1.2, 2.55)
ax[1].set_xlabel("Cross-validated RMSE (goals)")
ax[1].set_title("(b) Models compared (lower is better)", fontsize=8.5, loc="left")
for i, (v, sd) in enumerate(zip(r["rmse"].values, r["rmse_sd"].values)):
    ax[1].text(v + sd + 0.03, i, f"{v:.2f}", va="center", ha="left", fontsize=7)
for a in ax:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("fig_2_1.png", dpi=300)

# ---- Out-of-fold predictions (used in the report text) ----
from sklearn.model_selection import KFold, cross_val_predict
ridge_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
pred = cross_val_predict(ridge_model, X, y, cv=KFold(5, shuffle=True, random_state=42))
oof_r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("Out-of-fold Ridge R2:", round(oof_r2, 3), "SD pred:", round(pred.std(), 2), "SD actual:", round(y.std(), 2))
big = df["goal_difference"].abs() >= 3
print("Mean |actual| for |GD|>=3:", round(y[big].abs().mean(), 2), "mean |pred|:", round(np.abs(pred[big]).mean(), 2), "n =", int(big.sum()))
correct = (np.sign(pred) == np.sign(y))[y != 0].mean()
print("Winner called correctly (non-draws):", round(correct, 3))
