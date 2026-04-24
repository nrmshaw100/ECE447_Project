###
# in this file the functions for the Long Short Term Memory (LSTM) model
# are written
###

from collections.abc import Hashable, Mapping
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
import optuna
import evaluation_and_residual as ear

def build_combined_tf_dataset(df_X: np.ndarray, df_Y: np.ndarray, ref_df: pd.DataFrame, time_steps: int, batch_size: int = 32) -> tf.data.Dataset:
    """
    Builds a combined tf.data.Dataset across multiple engine units to prevent data leakage.
    Assumes the DataFrame is already scaled and contains a 'Unit Number' column.
    """ 
    combined_dataset = None
    
    for unit_id, group in ref_df.groupby(["Unit Number", "Dataset"]):
        X = df_X.loc[group.index].values
        y = df_Y.loc[group.index].values

        # Ensure the engine has at least enough rows to form one sequence
        if len(X) < time_steps:
            continue

        # The target for a sequence of length T should be the RUL at the final step of that window
        targets = y[time_steps - 1:]
        
        unit_ds = tf.keras.utils.timeseries_dataset_from_array(
            data=X,
            targets=targets,
            sequence_length=time_steps,
            batch_size=batch_size
        )

        if combined_dataset is None:
            combined_dataset = unit_ds
        else:
            combined_dataset = combined_dataset.concatenate(unit_ds)

    # Prefetch for performance optimization during training
    return combined_dataset.cache().prefetch(tf.data.AUTOTUNE)

class Objective:
    # 1. Initialize with raw arrays/dataframes, NOT pre-built tf.datasets
    def __init__(self, split_dict, train_ref, val_ref):
        self.X_train = split_dict["X_train_scaled"]
        self.y_train = split_dict["y_train_scaled"]
        self.ref_train = train_ref
        
        self.X_val = split_dict["X_val_scaled"]
        self.y_val = split_dict["y_val_scaled"]
        self.ref_val = val_ref

    def __call__(self, trial):
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Hyperparameters
            num_neurons = trial.suggest_int("num_neurons", 32, 256, step=32)
            epochs = trial.suggest_int("epochs", 10, 100, step=10)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
            
            # Data-level Hyperparameters
            timesteps = trial.suggest_int("time_steps", 1, 10, step=1)
            batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
            
            # Log ALL params (added timesteps and batch_size)
            mlflow.log_params({
                "num_neurons": num_neurons,
                "epochs": epochs,
                "dropout_rate": dropout_rate,
                "time_steps": timesteps,
                "batch_size": batch_size
            })

            # 2. Build the datasets dynamically for THIS specific trial
            train_ds = build_combined_tf_dataset(
                self.X_train, self.y_train, self.ref_train, 
                time_steps=timesteps, batch_size=batch_size
            )
            
            val_ds = build_combined_tf_dataset(
                self.X_val, self.y_val, self.ref_val, 
                time_steps=timesteps, batch_size=batch_size
            )

            # Safely grab the feature dimension from the newly created dataset
            num_features = train_ds.element_spec[0].shape[-1] 

            # Build Model
            model = Sequential([
                LSTM(num_neurons, input_shape=(None, num_features)),
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(loss='mae', optimizer='adam', jit_compile=True)

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,  
                restore_best_weights=True,
                verbose=1
            )

            history = model.fit(
                train_ds, 
                epochs=epochs, 
                verbose=1, 
                validation_data=val_ds, 
                shuffle=False, 
                callbacks=[early_stopping]
            )
            
            metrics, predictions = ear.evaluate_model_numerics(model, self.X_val, self.y_val)
            mlflow.log_metrics(metrics)

            res_metrics = ear.residuals_analysis_numerics(self.y_val, predictions)
            mlflow.log_metrics(res_metrics)

            best_val_loss = min(history.history['val_loss'])
            
            mlflow.log_metric("val_loss", best_val_loss)
            mlflow.tensorflow.log_model(model, artifact_path="model")

            return best_val_loss