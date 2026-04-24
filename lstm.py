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
    return combined_dataset.prefetch(tf.data.AUTOTUNE)

class Objective:
    def __init__(self, train, val):
        self.train = train
        self.val = val

    def __call__(self, trial):
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # 1. Hyperparameters
            num_neurons = trial.suggest_int("num_neurons", 32, 128, step=32)
            epochs = trial.suggest_int("epochs", 10, 100, step=10)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
            
            # Log params
            mlflow.log_params({
                "num_neurons": num_neurons,
                "epochs": epochs,
                "dropout_rate": dropout_rate
            })

            # Fix 1: Use self.train and safely grab the feature dimension
            num_features = self.train.element_spec[0].shape[-1] 

            # 2. Build Model
            model = Sequential([
                LSTM(num_neurons, input_shape=(None, num_features)),
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(loss='mae', optimizer='adam')

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,  
                restore_best_weights=True,
                verbose=1
            )

            history = model.fit(
                self.train, 
                epochs=epochs, 
                verbose=1, 
                validation_data=self.val, 
                shuffle=False, 
                callbacks=[early_stopping]
            )

            best_val_loss = min(history.history['val_loss'])
            
            # Log the metric to MLflow
            mlflow.log_metric("val_loss", best_val_loss)

            # HOW TO LOG THE MODEL:
            # This saves the model architecture, weights, and training configuration
            mlflow.tensorflow.log_model(model, artifact_path="model")

            # Fix 4: Return the target metric for Optuna
            return best_val_loss