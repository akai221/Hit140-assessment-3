# HIT140 Assessment 3 - Linear Regression 2.1
# Predicting goal difference (team_a minus team_b) for WC 2026 matches

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# load data
df = pd.read_csv("LR_2_1_dataset_clean.csv")
print(df.shape)
print(df.isnull().sum().sum(), "missing values")

# explanatory variables and response
x_cols = ["fifa_rank_diff", "attack_diff", "defence_diff", "h2h_avg_gd",
          "squad_value_diff_eur_m", "squad_age_diff", "venue_altitude_m",
          "host_advantage_diff"]
X = df[x_cols]
y = df["goal_difference"]

# collinearity check between explanatory variables
print("\nCorrelation between explanatory variables")
print(X.corr().round(2))

# correlation of each variable with goal difference
print("\nCorrelation with goal difference")
print(df[x_cols + ["goal_difference"]].corr()["goal_difference"].round(2))

# split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# build model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nIntercept:", round(model.intercept_, 4))
print("Coefficients:")
for col, c in zip(x_cols, model.coef_):
    print(col, round(c, 4))

# predict on test set
y_pred = model.predict(X_test)

# evaluate
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
nrmse = rmse / (y_test.max() - y_test.min())
r2 = metrics.r2_score(y_test, y_pred)
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\nTest set results")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("NRMSE:", round(nrmse, 4))
print("R2:", round(r2, 4))
print("Adjusted R2:", round(adj_r2, 4))

# training set results (adjusted R2 on test set is harsh because test set is small)
y_train_pred = model.predict(X_train)
r2_train = metrics.r2_score(y_train, y_train_pred)
n_train = len(y_train)
adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
print("\nTraining set results")
print("R2:", round(r2_train, 4))
print("Adjusted R2:", round(adj_r2_train, 4))

# baseline model - always predicts mean of training set
y_base = np.full(len(y_test), y_train.mean())
rmse_base = np.sqrt(metrics.mean_squared_error(y_test, y_base))
print("\nBaseline RMSE:", round(rmse_base, 4))

# standardised model so coefficients can be compared
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

model_std = LinearRegression()
model_std.fit(X_train_std, y_train)
y_pred_std = model_std.predict(X_test_std)

print("\nStandardised coefficients")
for col, c in zip(x_cols, model_std.coef_):
    print(col, round(c, 4))
print("R2 (standardised):", round(metrics.r2_score(y_test, y_pred_std), 4))
