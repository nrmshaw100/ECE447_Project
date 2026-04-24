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
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
import optuna
import evaluation_and_residual as ear
import keras

def build_combined_tf_dataset(df_X: np.ndarray, df_Y: np.ndarray, ref_df: pd.DataFrame, time_steps: int, batch_size: int = 32, shuffle: bool = False) -> tf.data.Dataset:
    """
    Builds a combined tf.data.Dataset across multiple engine units to prevent data leakage.
    Assumes the DataFrame is already scaled and contains a 'Unit Number' column.
    """ 
    combined_dataset = None
    X_list = []
    y_list = []
    
    for unit_id, group in ref_df.groupby(["Unit Number", "Dataset"]):
        if isinstance(df_X, pd.DataFrame):
            X = df_X.loc[group.index].values
            y = df_Y.loc[group.index].values
        else:
            # Safe indexing for NumPy arrays
            X = df_X[group.index]
            y = df_Y[group.index]

        # Ensure the engine has at least enough rows to form one sequence
        if len(X) < time_steps:
            continue

        # The target for a sequence of length T should be the RUL at the final step of that window
        targets = y[time_steps - 1:]
        num_seqs = len(X) - time_steps + 1
        X_seq = np.array([X[i : i + time_steps] for i in range(num_seqs)], dtype=np.float32)
        y_seq = y[time_steps - 1:].astype(np.float32)
        
        unit_ds = tf.keras.utils.timeseries_dataset_from_array(
            data=X,
            targets=targets,
            sequence_length=time_steps,
            batch_size=64 # Arbitrary initial batch size for generation
        )
        X_list.append(X_seq)
        y_list.append(y_seq)

    if not X_list:
        return None
        
        # Unbatch immediately so we can globally shuffle individual sequences later
        unit_ds = unit_ds.unbatch()
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

        if combined_dataset is None:
            combined_dataset = unit_ds
        else:
            combined_dataset = combined_dataset.concatenate(unit_ds)
    combined_dataset = tf.data.Dataset.from_tensor_slices((X_all, y_all))

    if shuffle and combined_dataset is not None:
        combined_dataset = combined_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

    if combined_dataset is not None:
        combined_dataset = combined_dataset.batch(batch_size)

    # Prefetch for performance optimization during training
    return combined_dataset.cache().prefetch(tf.data.AUTOTUNE)

class Objective:
    # 1. Initialize with raw arrays/dataframes, NOT pre-built tf.datasets
    def __init__(self, X_t, y_t, X_val, y_val, train_ref, val_ref):
        self.X_train = X_t
        self.y_train = y_t
        self.ref_train = train_ref
        
        self.X_val = X_val
        self.y_val = y_val
        self.ref_val = val_ref

    def __call__(self, trial):
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Hyperparameters
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            num_neurons = trial.suggest_int("num_neurons", 32, 256, step=32)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
            
            # const
            epochs = 50
            timesteps = 30
            batch_size = 128
            
            # Log ALL params (added timesteps and batch_size)
            mlflow.log_params({
                "num_neurons": num_neurons,
                "dropout_rate": dropout_rate,
                "learning_rate": lr
            })

            # 2. Build the datasets dynamically for THIS specific trial
            train_ds = build_combined_tf_dataset(
                self.X_train, self.y_train, self.ref_train, 
                time_steps=timesteps, batch_size=batch_size, shuffle=True
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
                Input(shape=(None, num_features)),
                LSTM(num_neurons),
                Dropout(dropout_rate),
                Dense(1)
            ])


            opt = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(
                loss='mse', 
                optimizer=opt, 
                jit_compile=True,
                metrics=[
                    tf.keras.metrics.MeanAbsoluteError(name="mae"),
                    tf.keras.metrics.R2Score(name="r2")
    ]
)

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
                callbacks=[early_stopping]
            )


            # return_dict=True outputs something like: {'loss': 0.042, 'mae': 0.15, 'r2': 0.88}
            val_metrics = model.evaluate(val_ds, return_dict=True)
            mlflow_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            mlflow.log_metrics(mlflow_metrics)
            mlflow.tensorflow.log_model(model, artifact_path="model")

            return val_metrics["loss"]