# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def rmsre(y_true, y_pred):  # RMSRE calculation
    relative_errors = (y_true - y_pred) / y_true
    return np.sqrt(np.mean(relative_errors**2))


# HELPER FUNCTIONS
def get_tile_count(row):
    if row["DIM"] == "H":
        return (row["INPUT_0_Y"] // 32) * (math.ceil((row["INPUT_0_X"] // 32) / row["CORE COUNT"]))
    else:
        return (row["INPUT_0_X"] // 32) * (math.ceil((row["INPUT_0_Y"] // 32) / row["CORE COUNT"]))


def get_unit_count(row):
    if row["DIM"] == "H":
        return math.ceil((row["INPUT_0_X"] // 32) / row["CORE COUNT"])
    else:
        return math.ceil((row["INPUT_0_Y"] // 32) / row["CORE COUNT"])


def get_overworked_cores(row):
    if row["DIM"] == "H":
        return (row["INPUT_0_X"] // 32) % row["CORE COUNT"]
    else:
        return (row["INPUT_0_Y"] // 32) % row["CORE COUNT"]


def parse_dim(entry):
    attr = (((entry.split(";"))[5]).split(":", 1))[1]
    return attr.rsplit(":", 1)[1].strip("'")


def get_mem_layout_type(row):
    ret = ""
    if "L1" in row["INPUT_0_MEMORY"]:
        ret += "L1"
    else:
        ret += "DRAM"
    return ret


def get_out_mem_layout_type(row):
    if "L1" in row["OUTPUT_0_MEMORY"]:
        ret = 0
    else:
        ret = 1
    return ret


def get_dram_pattern(entry):
    return int(12 / math.gcd(int(entry["INPUT_0_X"]) // 32, 12))


# DATA PREPARATION
def prepare_df(data):
    data = data[data["OP CODE"] == "Reduce"].copy()
    data.loc[:, "DIM"] = data["ATTRIBUTES"].apply(parse_dim)
    data.loc[:, "TILE / CORE"] = data.apply(get_tile_count, axis=1)
    data.loc[:, "INPUT STORAGE"] = data.apply(get_mem_layout_type, axis=1)
    data.loc[:, "OUTPUT STORAGE"] = data.apply(get_out_mem_layout_type, axis=1)
    data.loc[:, "UNIT / CORE"] = data.apply(get_unit_count, axis=1)
    data.loc[:, "OVW NUM"] = data.apply(get_overworked_cores, axis=1)
    return data


# ---------------------------------------------------------------------------------------------------------------------
# DATA MODELING
#   - Separate models for each group of input storage, input datatype and dimension
#   - Primary parameters are tile/core, work unit/core and output storage
#   - Polinomial features are used to capture non-linear relationships, currently degree 3 is used
#   - Features are scaled using standard scaler
#   - Ridge regression is used for modeling, hyperparameter alpha is selected using grid search
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    data = pd.read_csv("perf_modeling/reduction/interleaved/example_full_grid.csv")
    data = prepare_df(data)

    # optinal exclusion of data with work unit less than 8, this data may require a separate model
    # test_data = data[(data['OVW NUM'] <= 8) & (data['OVW NUM'] > 0)]
    # data = data[(data['OVW NUM'] > 8) | (data['OVW NUM'] == 0)]

    for keys, group in data.groupby(["INPUT STORAGE", "INPUT_0_DATATYPE", "DIM"]):
        df = group[["TILE / CORE", "UNIT / CORE", "OVW NUM", "DEVICE KERNEL DURATION [ns]"]].copy()
        y = df["DEVICE KERNEL DURATION [ns]"]
        X = df.drop(columns="DEVICE KERNEL DURATION [ns]")

        poly = PolynomialFeatures(degree=3, include_bias=False)
        X_poly = poly.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)

        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

        alphas = [0.1, 1.0, 10.0, 100.0]
        param_grid = {"alpha": alphas}
        model = GridSearchCV(Ridge(), param_grid, scoring="r2", cv=5)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        re = rmsre(y_test, y_pred)

        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)

        best_model = model.best_estimator_

        print("Group: ", keys)
        # print('Coeffs:', best_model.coef_)
        print("Alpha:", model.best_params_)
        print("Train R²:", train_r2)
        print("Test R²:", test_r2)
        print("RMSRE:", re)
        print("-----------------")

        # optional model evaluation on data with work unit less than 8
        # tst_X = test_data[(test_data['INPUT STORAGE'] == keys[0]) & (test_data['INPUT_0_DATATYPE'] == keys[1]) & (test_data['DIM'] == keys[2])]
        # tst_X = tst_X[['TILE / CORE', 'UNIT / CORE', 'OVW NUM', 'DEVICE KERNEL DURATION [ns]']].copy()
        # tst_y = tst_X['DEVICE KERNEL DURATION [ns]']
        # tst_X = tst_X.drop(columns='DEVICE KERNEL DURATION [ns]')

        # tst_X = scaler.transform(poly.transform(tst_X))
        # tst_pred = best_model.predict(tst_X)
        # print("Test data R²:", best_model.score(tst_X, tst_y))
        # print("Test data RMSRE:", rmsre(tst_y, tst_pred))
        # print("-----------------")
