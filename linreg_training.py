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

class LinRegObjective:
  def __init__(self, X_train, y_train, X_val, y_val, exp_id):
    self.X_train = X_train.copy()
    self.y_train = y_train.copy()
    self.X_val = X_val.copy()
    self.y_val = y_val.copy()
    self.exp_id = exp_id

  def __call__(self, trial):
    with mlflow.start_run(nested=True, experiment_id=self.exp_id):
      alpha = trial.suggest_float("alpha", 1e-3, 1e+3)
      mlflow.log_params({"alpha": alpha})

      reg = linear_model.Ridge(alpha=alpha)
      reg.fit(self.X_train, self.y_train)

      metrics, predictions = ear.evaluate_model_numerics(reg, self.X_val, self.y_val)
      mlflow.log_metrics(metrics)

      res_metrics = ear.residuals_analysis_numerics(self.y_val, predictions)
      mlflow.log_metrics(res_metrics)
      mse = mean_squared_error(self.y_val, predictions)

      mlflow.sklearn.log_model(reg)

      print(metrics["r2_score"])

      # Return a regression metric for Optuna to minimize
      return mse # Changed from log_loss

def main():
    mlflow.set_tracking_uri("http://localhost:5000")

    exp_name = "ridge_runs"
    mlflow.set_experiment(exp_name)
    
    data_dict = preprocessing.parse_data()

    # pipeline A data preprocessing
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
    [train_B, val_B] = [pipe_B[i] for i in pipe_B]
    xB_train, yB_train = preprocessing.target_feature_split(train_B)
    xB_val, yB_val = preprocessing.target_feature_split(val_B)

    #train-validation feature matrix and target vector
    xB_train_minMax = MinMaxScaler().fit_transform(xB_train)
    yB_train_minMax = MinMaxScaler().fit_transform(yB_train)
    xB_val_minMax = MinMaxScaler().fit_transform(xB_val)
    yB_val_minMax = MinMaxScaler().fit_transform(yB_val)

    # pipeline C preprocessing
    pipe_C_split = preprocessing.train_val_split(data_dict.copy(), test_size=0.3)
    [train_C, val_C] = [pipe_C_split[i] for i in pipe_C_split]

    #train-validation feature matrix and target vector
    xC_train, yC_train = preprocessing.target_feature_split(train_C)
    xC_val, yC_val = preprocessing.target_feature_split(val_C)

    experiment = mlflow.get_experiment_by_name(exp_name)
    exp_id = experiment.experiment_id

    # Pass exp_id to your objective class
    objective = LinRegObjective(xC_train, yC_train, xC_val, yC_val, exp_id)

    with mlflow.start_run(run_name="pipeline_C"):
        study = optuna.create_study(direction="minimize", study_name="LinReg_Optimization")

        # n_trials is the total number of hyperparameter combinations to test
        study.optimize(objective, n_trials=100)

        best_params = study.best_params
        
        best_reg = linear_model.Ridge(**best_params)
        best_reg.fit(xA_train, yA_train)
        mlflow.sklearn.log_model(best_reg, artifact_path="best-model")
        mlflow.set_tag("model_status", "production_candidate")

        mlflow.log_params(best_params)
        mlflow.log_metric("best_mse", study.best_value)

    


if __name__=="__main__":
    main()