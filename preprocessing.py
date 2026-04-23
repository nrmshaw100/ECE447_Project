from collections.abc import Hashable, Mapping
import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for future preprocessing pipeline steps."""
    return df.copy()


def _coefficient_of_variation(series: pd.Series) -> float:
    """Return abs(std / mean), or infinity when the mean is effectively zero."""
    mean = series.mean()
    if pd.isna(mean) or np.isclose(mean, 0.0):
        return np.inf

    std = series.std()
    if pd.isna(std):
        return np.inf

    return float(abs(std / mean))


def drop_low_cv_sensors(
    data_dict: Mapping[Hashable, pd.DataFrame],
    threshold: float = 0.1,
) -> tuple[dict[Hashable, pd.DataFrame], list[str]]:
    """Drop sensors whose coefficient of variation is below the threshold across all datasets.

    Args:
        data_dict: Mapping of dataset ids to pandas DataFrames. Sensor columns are
            expected to follow the "Sensor <n>" naming convention, while all other
            columns such as unit ids, cycle counts, settings, RUL, and dataset labels
            are preserved.
        threshold: Minimum coefficient of variation required to keep a sensor.

    Returns:
        A tuple containing:
        - a copied dictionary of DataFrames with globally low-CV sensor columns removed
        - a sorted list of sensor column names that were dropped

    A sensor is removed only if its coefficient of variation, computed as abs(std / mean),
    is below ``threshold`` in every DataFrame where that sensor exists.
    """
    filtered_data = {key: df.copy() for key, df in data_dict.items()}

    sensor_cols = sorted(
        {
            col
            for df in filtered_data.values()
            for col in df.columns
            if col.startswith("Sensor")
        },
        key=lambda name: int(name.split()[1]),
    )

    sensors_to_drop: list[str] = []

    for sensor in sensor_cols:
        cv_values = [
            _coefficient_of_variation(df[sensor])
            for df in filtered_data.values()
            if sensor in df.columns
        ]

        if cv_values and all(cv < threshold for cv in cv_values):
            sensors_to_drop.append(sensor)

    for df in filtered_data.values():
        present_sensors = [sensor for sensor in sensors_to_drop if sensor in df.columns]
        if present_sensors:
            df.drop(columns=present_sensors, inplace=True)

    return filtered_data, sensors_to_drop

def compute_RUL(
    data_dict: Mapping[Hashable, pd.DataFrame]
) -> dict[Hashable, pd.DataFrame]:
    """Compute the RUL for each row in the datasets and add it as a new column.
        Args:
        data_dict: Mapping of dataset ids to pandas DataFrames.

    Returns:
        - a copied dictionary of DataFrames with RUL column
     RUL column is computed for each time step of each unit.
    """
    for i in data_dict:
        df = data_dict[i].copy()
        df["RUL"] = df.groupby("Unit Number")["Time, In Cycles"].transform(lambda x: x.max() - x)
        df["Dataset"] = f"FD00{i}" # adding a column to identify which dataset each row belongs to for combined analysis
        data_dict[i] = df
    return data_dict

def compute_lags(
        data_dict: Mapping[Hashable, pd.DataFrame],
        sensor_cols: list[str],
        lags: list[int]
) -> dict[Hashable, pd.DataFrame]:
        """Compute lag features for specified sensor columns and lags, and add them as new columns.
            Args:
            data_dict: Mapping of dataset ids to pandas DataFrames.
            sensor_cols: List of sensor column names to compute lags for.
            lags: List of integer lag values to compute.

        Returns:
            A dictionary of DataFrames with the computed lag features added as new columns.
        Lag features are computed for each time step of each unit, and lagged values are aligned with the current time step."""

        lagged_data = {key: df.copy() for key, df in data_dict.items()}
        for i in data_dict:
            df = data_dict[i].copy()
            # creating lag features for each unit to avoid data leaks across units
            for lag in lags:
                df_lag = df.groupby("Unit Number")[sensor_cols].shift(lag)
                df_lag.columns = [f"{col}_lag{lag}" for col in sensor_cols]

                df = df.join(df_lag)
            lagged_data[i] = df.dropna()
        return lagged_data

def compute_window_features(
    data_dict: Mapping[Hashable, pd.DataFrame],
    sensor_cols: list[str],
    window_size: int
) -> dict[Hashable, pd.DataFrame]:
    """Compute strictly historical rolling window features for each sensor.
        Args:
        data_dict: Mapping of dataset ids to pandas DataFrames.
        sensor_cols: List of sensor column names to compute window features for.
        window_size: Integer size of the rolling window.
    Returns:
    A dictionary of DataFrames with the computed window features added as new columns.
    Window features are computed for each time step of each unit, and rolling calculations are performed separately for each unit to avoid data leaks across units.
     Each row only uses prior time steps; the current time step is excluded from its own window.
     The new columns are named in the format "<sensor>_window{window_size}_mean" and "<sensor>_window{window_size}_std" for the mean and standard deviation features, respectively.
    """
    windowed_data = {key: df.copy() for key, df in data_dict.items()}
    for i in data_dict:
        df = data_dict[i].copy()
        # creating rolling window features for each unit to avoid data leaks across units
        for sensor in sensor_cols:
            history = df.groupby("Unit Number")[sensor].shift(1)
            df[f"{sensor}_window{window_size}_mean"] = history.groupby(df["Unit Number"]).transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
            )
            df[f"{sensor}_window{window_size}_std"] = history.groupby(df["Unit Number"]).transform(
                lambda x: x.rolling(window=window_size, min_periods=1).std()
            )
        windowed_data[i] = df.dropna()
    return windowed_data

def parse_data() -> dict[int, pd.DataFrame]:
    """Parse the original CMAPSS datasets from the text files and return a dictionary of DataFrames.
    """
    data_dict = {}
    for i in range(1,5):
        df = pd.read_csv(f"CMAPSSData/train_FD00{str(i)}.txt", sep=" ", header=None)
        df = df.drop(columns=[26, 27])  # Remove the last two empty columns
        df.columns = ["Unit Number", "Time, In Cycles", "Setting 1", "Setting 2", "Setting 3"] + [f"Sensor {i}" for i in range(1, 22)]
        data_dict[i] = df
    return data_dict

def clip_RUL(data_dict: Mapping[Hashable, pd.DataFrame], max_RUL: int = 125) -> dict[Hashable, pd.DataFrame]:
    """Clip the RUL values in the datasets to a maximum value.

    Args:
        data_dict: Mapping of dataset ids to pandas DataFrames, each containing an "RUL" column.
        max_RUL: Maximum RUL value to clip to. Any RUL values above this will be set to max_RUL.

    Returns:
        A dictionary of DataFrames with the RUL values clipped to the specified maximum.
    """
    clipped_data = {key: df.copy() for key, df in data_dict.items()}
    for key, df in clipped_data.items():
        if "RUL" in df.columns:
            df["RUL"] = df["RUL"].clip(upper=max_RUL)
    return clipped_data

def pipeline_A(data_dict: Mapping[Hashable, pd.DataFrame]) -> dict[Hashable, pd.DataFrame]:
    """Example pipeline that applies the preprocessing steps in sequence."""
    processed_data, dropped_sensors = drop_low_cv_sensors(data_dict, threshold=0.05)
    processed_data = compute_RUL(processed_data)
    sensor_cols = [col for col in data_dict[1].columns.drop(dropped_sensors) if col.startswith("Sensor")]
    processed_data = compute_lags(processed_data, sensor_cols=sensor_cols, lags=[1, 2, 3])
    processed_data = compute_window_features(processed_data, sensor_cols=sensor_cols, window_size=5)
    processed_data = clip_RUL(processed_data, max_RUL=125)
    return processed_data