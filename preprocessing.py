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
