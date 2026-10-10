# HIT140 Assessment 3 - Linear Regression 2.1
# Inference for the linear model on all 104 matches: coefficient tests, 95% confidence
# intervals, overall F-test and residual checks. Predictors are standardised so the
# coefficients can be compared (goals of goal difference per 1 SD of the variable).

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("LR_2.1_dataset_clean.csv")
x_cols = ["fifa_rank_diff", "attack_diff", "defence_diff", "h2h_avg_gd",
          "squad_value_diff_eur_m", "squad_age_diff", "venue_altitude_m",
          "host_advantage_diff"]
X = df[x_cols]
y = df["goal_difference"].to_numpy()
Z = ((X - X.mean()) / X.std(ddof=0)).to_numpy()
A = np.column_stack([np.ones(len(y)), Z])
n, k = A.shape

beta = np.linalg.lstsq(A, y, rcond=None)[0]
resid = y - A @ beta
dof = n - k
sigma2 = resid @ resid / dof
cov = sigma2 * np.linalg.inv(A.T @ A)
se = np.sqrt(np.diag(cov))
t = beta / se
p = 2 * stats.t.sf(np.abs(t), dof)
tcrit = stats.t.ppf(0.975, dof)
out = pd.DataFrame({"variable": ["intercept"] + x_cols, "coef_per_SD": beta, "se": se,
                    "t": t, "p": p, "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se})
print(out.round(3).to_string(index=False))

ss_res, ss_tot = resid @ resid, ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
adj = 1 - (1 - r2) * (n - 1) / dof
F = (r2 / (k - 1)) / ((1 - r2) / dof)
print(f"\nR2 {r2:.3f}  adjusted R2 {adj:.3f}  F({k-1},{dof}) = {F:.2f}  p = {stats.f.sf(F, k-1, dof):.2g}")

# residual checks
sw = stats.shapiro(resid)
print(f"Shapiro-Wilk on residuals: W = {sw.statistic:.3f}, p = {sw.pvalue:.3f}")
e2 = resid ** 2
g = np.linalg.lstsq(A, e2, rcond=None)[0]
r2_bp = 1 - ((e2 - A @ g) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
lm = n * r2_bp
print(f"Breusch-Pagan: LM = {lm:.2f}, p = {stats.chi2.sf(lm, k-1):.3f}")
h = np.einsum("ij,jk,ik->i", A, np.linalg.inv(A.T @ A), A)
cook = (resid ** 2 / (k * sigma2)) * h / (1 - h) ** 2
print(f"Largest Cook's distance: {cook.max():.2f} (match {df['match_id'].iloc[cook.argmax()]})")
out.round(4).to_csv("ols_inference_2_1_results.csv", index=False)
