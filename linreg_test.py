import pandas as pd
import numpy as np
import os
from IPython.display import display as disp
import preprocessing
import lstm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import mlflow
import optuna
import optuna.visualization as vis
from optuna.integration.mlflow import MLflowCallback
from sklearn import linear_model
import evaluation_and_residual as ear
from sklearn.metrics import mean_squared_error # Changed from log_loss

def process_test_data(data_dict, x_scaler, y_scaler, dropped_sensors=None):
    """
    Processes test data to be evaluated on both Linear Regression and LSTM models.
    Returns scaled features and targets, as well as a reference dataframe (for the LSTM).
    """
    testing_data = data_dict["test"].copy()
    # CMAPSS labels are usually stored under 'RUL' or 'y_test' depending on your parse config
    testing_labels = data_dict.get("RUL", data_dict.get("labels", data_dict.get("y_test")))

    if dropped_sensors is None:
        pipe_A_test, dropped_sensors = preprocessing.drop_low_cv_sensors(testing_data, threshold=0.05)
    else:
        pipe_A_test = {}
        for k, df in testing_data.items():
            pipe_A_test[k] = df.drop(columns=dropped_sensors, errors='ignore')

    pipe_A_test = preprocessing.compute_RUL(pipe_A_test)
    
    first_key = list(pipe_A_test.keys())[0]
    sensor_cols = [col for col in pipe_A_test[first_key].columns if col.startswith("Sensor")]
    
    pipe_A_test = preprocessing.compute_lags(pipe_A_test, sensor_cols=sensor_cols, lags=[1, 5, 20], drop_na=False)
    pipe_A_test = preprocessing.compute_window_features(pipe_A_test, sensor_cols=sensor_cols, window_size=10, drop_na=False)
    pipe_A_test = preprocessing.clip_RUL(pipe_A_test, max_RUL=125)

    xa_test_list = []
    ya_test_list = []
    ref_test_list = []

    for k in pipe_A_test:
        df = pipe_A_test[k].copy()
        labels = testing_labels[k]

        unique_units = df["Unit Number"].unique()
        label_map = dict(zip(unique_units, labels.iloc[:, 0]))

        # Calculate true RUL and cap it at 125
        true_rul = df["Unit Number"].map(label_map) + df["RUL"]
        df["RUL"] = true_rul.clip(upper=125)

        ref_test_list.append(df.copy()) # Full dataframe reference required for LSTM
        
        # Split using the same preprocessing rules as the training process
        xa, ya = preprocessing.target_feature_split(df)
        xa_test_list.append(xa)
        ya_test_list.append(ya)

    xa_test_unscaled = pd.concat(xa_test_list, ignore_index=True)
    ya_test_unscaled = pd.concat(ya_test_list, ignore_index=True)
    test_ref = pd.concat(ref_test_list, ignore_index=True)

    xa_test_scaled = x_scaler.transform(xa_test_unscaled)
    ya_test_scaled = y_scaler.transform(ya_test_unscaled.values.reshape(-1, 1)).flatten()

    return xa_test_scaled, ya_test_scaled, test_ref

def main():
    data_dict = preprocessing.parse_data()
    
    # Rebuild the scalers and determine dropped sensors based off the training pipeline
    pipe_A_train = preprocessing.pipeline_A(data_dict)
    train_ref = pd.concat(pipe_A_train["train"].values(), ignore_index=True)
    xA_train, yA_train = preprocessing.target_feature_split(train_ref)
    
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(xA_train)
    y_scaler.fit(yA_train.values.reshape(-1, 1))
    
    _, dropped_sensors = preprocessing.drop_low_cv_sensors(data_dict["train"].copy(), threshold=0.05)
    
    xa_test_scaled, ya_test_scaled, test_ref = process_test_data(data_dict, x_scaler, y_scaler, dropped_sensors)

    print(f"Feature matrix shape: {xa_test_scaled.shape}")
    print(f"Target vector shape: {ya_test_scaled.shape}")
    print(f"Test reference shape (for LSTM): {test_ref.shape}")

if __name__ == "__main__":
    main()