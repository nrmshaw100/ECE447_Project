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

# could convert dataset to a tf.data.Dataset

def get_lstm_model(neurons, timesteps, features):
    model = Sequential([
        LSTM(neurons, input_shape=(timesteps, features)),
        Dense(1)
    ])
    model.compile(loss='mae', optimizer='adam')
    return model

def train_model(train, test, epochs):
    train_x = train["X"]
# function that takes in the training data and attaches the cycle count

# train the model using model.fit
def fit_model(model, train_x, train_y, epochs):
    model.fit(train_x, train_y, epochs=epochs, verbose=1)
    return model

