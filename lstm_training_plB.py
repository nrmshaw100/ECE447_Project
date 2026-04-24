import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

def main():
    mlflow.set_tracking_uri("http://localhost:5000")

    exp_name = "lstm_runs"
    mlflow.set_experiment(exp_name)
    
    data_dict = preprocessing.parse_data()

    # pipeline A data preprocessing
    pipe_A = preprocessing.pipeline_A(data_dict)
    train_ref = pd.concat(pipe_A["train"].values(), ignore_index=True)
    val_ref = pd.concat(pipe_A["val"].values(), ignore_index=True)
    xA_train, yA_train = preprocessing.target_feature_split(train_ref)
    xA_val, yA_val = preprocessing.target_feature_split(val_ref)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    xA_train_std = x_scaler.fit_transform(xA_train)
    xA_val_std = x_scaler.transform(xA_val)
    yA_train_std = y_scaler.fit_transform(yA_train.values.reshape(-1, 1)).flatten()
    yA_val_std = y_scaler.transform(yA_val.values.reshape(-1, 1)).flatten()

    # pipeline B data preprocessing
    pipe_B = preprocessing.pipeline_B(data_dict)
    train_ref_B = pd.concat(pipe_B["train"].values(), ignore_index=True)
    val_ref_B = pd.concat(pipe_B["val"].values(), ignore_index=True)
    xB_train, yB_train = preprocessing.target_feature_split(train_ref_B)
    xB_val, yB_val = preprocessing.target_feature_split(val_ref_B)

    #train-validation feature matrix and target vector
    xB_scaler = MinMaxScaler()
    yB_scaler = MinMaxScaler()
    xB_train_minMax = xB_scaler.fit_transform(xB_train)
    xB_val_minMax = xB_scaler.transform(xB_val)
    yB_train_minMax = yB_scaler.fit_transform(yB_train.values.reshape(-1, 1)).flatten()
    yB_val_minMax = yB_scaler.transform(yB_val.values.reshape(-1, 1)).flatten()

    # pipeline C preprocessing
    pipe_C_split = preprocessing.train_val_split(data_dict.copy(), test_size=0.3)
    train_ref_C = pd.concat(pipe_C_split["train"].values(), ignore_index=True)
    val_ref_C = pd.concat(pipe_C_split["val"].values(), ignore_index=True)

    #train-validation feature matrix and target vector
    xC_train, yC_train = preprocessing.target_feature_split(train_ref_C)
    xC_val, yC_val = preprocessing.target_feature_split(val_ref_C)
    xC_train_raw = xC_train.values
    xC_val_raw = xC_val.values
    yC_train_raw = yC_train.values.reshape(-1, 1).flatten()
    yC_val_raw = yC_val.values.reshape(-1, 1).flatten()

    # need to rearrange the data so that it plays nicely with the LSTM model. There is no feature engineering 
    # or preprocessing that happens in this step
    split_dict = preprocessing.pipeline_A(data_dict)
    dfa_std = preprocessing.standardize_data(split_dict)

    train_df = pd.concat(split_dict["train"].values(), ignore_index=True) 
    val_df = pd.concat(split_dict["val"].values(), ignore_index=True)

    experiment = mlflow.get_experiment_by_name(exp_name)
    exp_id = experiment.experiment_id

    objective = lstm.Objective(
        X_t=xB_train_minMax, 
        y_t=yB_train_minMax, 
        X_val=xB_val_minMax, 
        y_val=yB_val_minMax, 
        train_ref=train_ref_B, # The LSTM needs this to group by "Unit Number"
        val_ref=val_ref_B
    )

    with mlflow.start_run(run_name="pipeline_B"):
        study = optuna.create_study(direction="minimize", study_name="LSTM_Optimization")

        # n_trials is the total number of hyperparameter combinations to test
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        
        mlflow.log_params(best_params)
        mlflow.log_metric("best_mse", study.best_value)
        print(f"Best hyperparameters found: {best_params}")

    


if __name__=="__main__":
    main()