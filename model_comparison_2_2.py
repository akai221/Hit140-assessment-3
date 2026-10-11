# HIT140 Assessment 3 - Linear Regression 2.2
# Compares linear regression with Ridge, Lasso, Poisson regression, random forest and
# gradient boosting using repeated 5-fold cross-validation grouped by match, so the two
# rows of one match always stay in the same fold (10 repeats, seed 42).

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("Data_set_2.2.csv")
x_cols = ["team_fifa_rank", "opp_fifa_rank", "team_is_host", "is_knockout",
          "team_prev_wc_appearances", "opp_prev_wc_appearances",
          "team_best_wc_finish", "opp_best_wc_finish"]
y = df["goals_scored"].to_numpy()
X = df[x_cols]

# combined version: the overlapping variables turned into team-minus-opponent differences
X2 = pd.DataFrame({
    "rank_diff": df["opp_fifa_rank"] - df["team_fifa_rank"],
    "apps_diff": df["team_prev_wc_appearances"] - df["opp_prev_wc_appearances"],
    "best_diff": df["team_best_wc_finish"] - df["opp_best_wc_finish"],
    "team_is_host": df["team_is_host"], "is_knockout": df["is_knockout"]})

models = {
    "Baseline (mean)": (DummyRegressor(), X),
    "Linear regression": (make_pipeline(StandardScaler(), LinearRegression()), X),
    "Linear (combined differences)": (make_pipeline(StandardScaler(), LinearRegression()), X2),
    "Ridge": (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))), X),
    "Lasso": (make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=20000)), X),
    "Poisson regression": (make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000)), X),
    "Random forest": (RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=5, random_state=42), X),
    "Gradient boosting": (GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2,
                                                    subsample=0.8, random_state=42), X),
}

groups = df["match_id"].to_numpy()
ids = np.unique(groups)
rng = np.random.default_rng(42)
splits = []
for rep in range(10):
    perm = rng.permutation(ids)
    fold_of = {m: i % 5 for i, m in enumerate(perm)}
    f = np.array([fold_of[g] for g in groups])
    splits += [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)]

rows = []
for name, (m, feats) in models.items():
    rm, ma, r2 = [], [], []
    for tr, te in splits:
        m.fit(feats.iloc[tr], y[tr])
        p = m.predict(feats.iloc[te])
        rm.append(np.sqrt(mean_squared_error(y[te], p)))
        ma.append(mean_absolute_error(y[te], p))
        r2.append(r2_score(y[te], p))
    rows.append({"model": name, "rmse": np.mean(rm), "rmse_sd": np.std(rm),
                 "mae": np.mean(ma), "r2": np.mean(r2), "r2_sd": np.std(r2)})
res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
res.to_csv("model_comparison_2_2_results.csv", index=False)

# ---- Figure: (a) correlation of each variable with goals scored, (b) model comparison ----
labels = {"team_fifa_rank": "Team FIFA rank", "opp_fifa_rank": "Opponent FIFA rank",
          "team_is_host": "Team is host", "is_knockout": "Knockout match",
          "team_prev_wc_appearances": "Team appearances", "opp_prev_wc_appearances": "Opp. appearances",
          "team_best_wc_finish": "Team best finish", "opp_best_wc_finish": "Opp. best finish"}
corr = X.corrwith(df["goals_scored"]).sort_values()
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.35))
ax[0].barh([labels[i] for i in corr.index], corr.values,
           color=["#9A9A9A" if v < 0 else "#333333" for v in corr.values])
ax[0].axvline(0, color="#444", lw=0.6)
ax[0].set_xlabel("Correlation with goals scored")
ax[0].set_title("(a) Single-variable relationships", fontsize=8.5, loc="left")
for i, v in enumerate(corr.values):
    ax[0].text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.2f}", va="center",
               ha="left" if v >= 0 else "right", fontsize=7)
ax[0].set_xlim(-0.6, 0.6)
r = res.iloc[::-1].reset_index(drop=True)
ypos = np.arange(len(r))
ax[1].errorbar(r["rmse"], ypos, xerr=r["rmse_sd"], fmt="o", color="#222222",
               ecolor="#9A9A9A", capsize=2, ms=4)
ax[1].set_yticks(ypos)
ax[1].set_yticklabels(r["model"])
ax[1].axvline(r["rmse"].iloc[-1], color="#999", lw=0.6, ls="--")
ax[1].set_xlim(0.95, 1.85)
ax[1].set_xlabel("Cross-validated RMSE (goals)")
ax[1].set_title("(b) Models compared", fontsize=8.5, loc="left")
for i, (v, sd_) in enumerate(zip(r["rmse"].values, r["rmse_sd"].values)):
    ax[1].text(v + sd_ + 0.02, i, f"{v:.2f}", va="center", ha="left", fontsize=7)
for a_ in ax:
    for s_ in ("top", "right"):
        a_.spines[s_].set_visible(False)
plt.tight_layout()
plt.savefig("fig_2_2.png", dpi=300)
