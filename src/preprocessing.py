"""
Preprocessing and sequence pipeline for Multivariate Weather Forecasting.
Preserves original behavior while offering leakage-free chronological scaling.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TARGET_COLUMN = 'Temperature (C)'
DROP_COLUMNS = ["Summary", "Daily Summary", "Precip Type", "Wind Bearing (degrees)"]

# Indices of physical sensor telemetry vs calendar timestamps
# Feature order: ['Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 
#                 'Visibility (km)', 'Cloud Cover', 'Pressure (millibars)', 
#                 'year', 'month', 'day', 'hour']
SENSOR_FEATURE_INDICES = [0, 1, 2, 3, 4, 5]
CALENDAR_FEATURE_INDICES = [6, 7, 8, 9]


def load_and_clean_data(csv_path="weatherHistory.xls"):
    """
    Loads and cleans the weather dataset matching the original notebook pipeline.
    """
    if not os.path.exists(csv_path):
        alt_path = "weatherHistory.csv"
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find dataset at {csv_path} or {alt_path}")

    df = pd.read_csv(csv_path)

    # Rename incorrectly named column
    if 'Loud Cover' in df.columns:
        df.rename(columns={'Loud Cover': 'Cloud Cover'}, inplace=True)

    # Drop non-numerical text columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    # Parse and extract datetime features
    df_clean['Formatted Date'] = pd.to_datetime(df_clean['Formatted Date'], utc=True)
    df_clean['year'] = df_clean['Formatted Date'].dt.year
    df_clean['month'] = df_clean['Formatted Date'].dt.month
    df_clean['day'] = df_clean['Formatted Date'].dt.day
    df_clean['hour'] = df_clean['Formatted Date'].dt.hour
    df_clean = df_clean.drop(columns=['Formatted Date'])

    # Drop missing rows
    df_clean = df_clean.dropna().reset_index(drop=True)
    return df_clean


def prepare_datasets(
    df_clean,
    sequence_length=24,
    train_ratio=0.70,
    val_ratio=0.15,
    leakage_free=False,
    scaler_dir="saved_models"
):
    """
    Normalizes data and builds sliding window sequences.
    
    Parameters:
    - leakage_free: if False, reproduces the original notebook (fitting on whole dataset).
                    if True, fits scalers strictly on the training partition.
    """
    features = df_clean.drop(columns=[TARGET_COLUMN])
    target = df_clean[[TARGET_COLUMN]]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    if not leakage_free:
        # Original pipeline: Fit scalers on the entire dataset
        scaled_features = feature_scaler.fit_transform(features)
        scaled_target = target_scaler.fit_transform(target)

        # Sliding window sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i - sequence_length:i])
            y.append(scaled_target[i])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        total_samples = X.shape[0]
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    else:
        # Leakage-free pipeline: Chronological train boundary first
        total_rows = len(df_clean)
        train_rows = int(train_ratio * total_rows)
        val_rows = int(val_ratio * total_rows)

        train_feat = features.iloc[:train_rows]
        train_targ = target.iloc[:train_rows]

        feature_scaler.fit(train_feat)
        target_scaler.fit(train_targ)

        scaled_features = feature_scaler.transform(features)
        scaled_target = target_scaler.transform(target)

        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i - sequence_length:i])
            y.append(scaled_target[i])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        total_samples = X.shape[0]
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Persist scalers if directory provided
    if scaler_dir:
        os.makedirs(scaler_dir, exist_ok=True)
        joblib.dump(feature_scaler, os.path.join(scaler_dir, "feature_scaler.joblib"))
        joblib.dump(target_scaler, os.path.join(scaler_dir, "target_scaler.joblib"))

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_names": list(features.columns)
    }

