# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


def rmsre(y_true, y_pred):  # RMSRE calculation
    relative_errors = (y_true - y_pred) / y_true
    return np.sqrt(np.mean(relative_errors**2))


# HELPER FUNCTIONS
def get_tile_count(row):
    return (row["INPUT_0_Y"] * (row["INPUT_0_X"] // row["CORE COUNT"])) // (
        32 * 32
    )  # only height reduction supported for sharding


# DATA PREPARATION
def prepare_df(data):
    data = data[data["OP CODE"] == "Reduce"].copy()
    data["TILE / CORE"] = data.apply(get_tile_count, axis=1)
    data = pd.get_dummies(data, columns=["INPUT_0_DATATYPE"], drop_first=False)
    return data


# ---------------------------------------------------------------------------------------------------------------------
# DATA MODELING
#   - Unique model for sharded input
#   - Primary features are tile/core and input datatype(one hot encoded)
#   - Derived feature is tile/core * input datatype
#   - Ridge regression is used for modeling, hyperparameter alpha is 0.1
# ---------------------------------------------------------------------------------------------------------------------
data = pd.read_csv("perf_estimates/reduction/sharded/example.csv")
data = prepare_df(data)

X = data[["TILE / CORE"] + [col for col in data.columns if "INPUT_0_DATATYPE" in col]].copy()
y = data["DEVICE KERNEL DURATION [ns]"]

for col in X.columns:
    if "INPUT_0_DATATYPE" in col:
        X[f"TILE / CORE{col}"] = X["TILE / CORE"] * X[col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

train_score = ridge_model.score(X_train, y_train)
test_score = ridge_model.score(X_test, y_test)

print("Train Score:", train_score)
print("Test Score:", test_score)
# print("Intercept:", ridge_model.intercept_)
# print("Coefficients:", ridge_model.coef_)

y_test_pred = ridge_model.predict(X_test)
test_rmsre = rmsre(y_test, y_test_pred)
print("Test RMSRE:", test_rmsre)
