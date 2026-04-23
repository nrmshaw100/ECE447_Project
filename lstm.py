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
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def build_combined_tf_dataset(df: pd.DataFrame, feature_cols: list, target_col: str, time_steps: int, batch_size: int = 32) -> tf.data.Dataset:
    """
    Builds a combined tf.data.Dataset across multiple engine units to prevent data leakage.
    Assumes the DataFrame is already scaled and contains a 'Unit Number' column.
    """
    combined_dataset = None

    for unit_id, group in df.groupby("Unit Number"):
        X = group[feature_cols].values
        y = group[target_col].values

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

def train_lstm_model(data_dict: Mapping[Hashable, pd.DataFrame], epochs: int, num_neurons: int, timesteps: int):
    model = Sequential([
        LSTM(num_neurons, input_shape=(timesteps)),
        Dense(1)
    ])
    model.compile(loss='mae', optimizer='adam')
    # train the model
    model.fit(data_dict["X_train"], data_dict["y_train"], epochs=epochs, verbose=1)
    return model, data_dict["X_test"], data_dict["y_test"]

# train the model using model.fit
def fit_model(model, train_x, train_y, epochs):
    model.fit(train_x, train_y, epochs=epochs, verbose=1)
    return model
