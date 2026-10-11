# HIT140 Assessment 3 - Linear Regression 2.2
# Inference for the linear model on all 208 rows. Each match appears twice, so the two
# rows are not independent: standard errors are clustered by match (CR1 estimator).

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("Data_set_2.2.csv")
x_cols = ["team_fifa_rank", "opp_fifa_rank", "team_is_host", "is_knockout",
          "team_prev_wc_appearances", "opp_prev_wc_appearances",
          "team_best_wc_finish", "opp_best_wc_finish"]
X = df[x_cols]
y = df["goals_scored"].to_numpy()
A = np.column_stack([np.ones(len(y)), X.to_numpy()])
n, k = A.shape
g = df["match_id"].to_numpy()

beta = np.linalg.lstsq(A, y, rcond=None)[0]
e = y - A @ beta
bread = np.linalg.inv(A.T @ A)
meat = np.zeros((k, k))
ids = np.unique(g)
for m in ids:
    s = A[g == m].T @ e[g == m]
    meat += np.outer(s, s)
G = len(ids)
cov = bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / (n - k))
se = np.sqrt(np.diag(cov))
t = beta / se
p = 2 * stats.t.sf(np.abs(t), G - 1)
tcrit = stats.t.ppf(0.975, G - 1)
sd = np.r_[1.0, X.std(ddof=0).to_numpy()]
out = pd.DataFrame({"variable": ["intercept"] + x_cols, "coef": beta,
                    "std_coef": beta * sd / y.std(ddof=0), "se_clustered": se,
                    "p": p, "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se})
print(out.round(3).to_string(index=False))

# joint test that all eight slopes are zero (Wald, clustered)
R = np.eye(k)[1:]
W = (R @ beta) @ np.linalg.inv(R @ cov @ R.T) @ (R @ beta)
print(f"\nWald chi2(8) = {W:.2f}, p = {stats.chi2.sf(W, k - 1):.4f}")
r2 = 1 - (e @ e) / ((y - y.mean()) ** 2).sum()
print(f"R2 {r2:.3f}, adjusted R2 {1 - (1 - r2) * (n - 1) / (n - k):.3f}")
res = stats.shapiro(e)
print(f"Shapiro-Wilk on residuals: W = {res.statistic:.3f}, p = {res.pvalue:.4f}")
print("Residual range:", e.min().round(2), e.max().round(2),
      "| fitted values below 0:", int(((A @ beta) < 0).sum()))
out.round(4).to_csv("ols_inference_2_2_results.csv", index=False)
