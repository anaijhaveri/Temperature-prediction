"""
Inference module for Date-Time based Temperature Prediction.
Preserves the user's original interactive prediction capability while 
adding optional cybersecurity attack simulation.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf


def predict_temperature_by_datetime(
    df_clean,
    lstm_model,
    rnn_model,
    feature_scaler,
    target_scaler,
    year=None,
    month=None,
    day=None,
    hour=None,
    epsilon=0.0,
    verbose=True
):
    """
    Predicts next-hour temperature using the 24 preceding hours from df_clean.
    If year, month, day, hour are None, prompts the user via input() (matching original notebook).
    
    Parameters:
    -----------
    epsilon : float
        If 0.0, runs normal clean prediction.
        If > 0.0, simulates an adversarial attack on the input telemetry before prediction.
    """
    try:
        if year is None:
            year = int(input("Enter year (e.g., 2016): "))
        if month is None:
            month = int(input("Enter month (1-12): "))
        if day is None:
            day = int(input("Enter day (1-31): "))
        if hour is None:
            hour = int(input("Enter hour (0-23): "))

        # Ensure datetime index on df_clean
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            df_with_idx = df_clean.copy()
            df_with_idx.index = pd.to_datetime(df_clean[['year', 'month', 'day', 'hour']])
        else:
            df_with_idx = df_clean

        input_time = datetime(year, month, day, hour)
        start_time = input_time - timedelta(hours=24)

        # Filter the 24 previous hours
        mask = (df_with_idx.index >= start_time) & (df_with_idx.index < input_time)
        input_data = df_with_idx.loc[mask].drop(columns=['Temperature (C)'])

        if input_data.shape[0] < 24:
            if verbose:
                print(f"Not enough historical data to make a prediction (found {input_data.shape[0]} rows, expected 24).")
            return None

        scaled_input = feature_scaler.transform(input_data)
        X_input = np.expand_dims(scaled_input, axis=0).astype(np.float32)

        # If epsilon > 0, inject adversarial perturbation on sensor features
        if epsilon > 0.0:
            from src.attacks.fgsm import generate_fgsm_attack
            # Use pseudo-target or dummy ground truth for gradient calculation
            dummy_target = np.array([[0.5]], dtype=np.float32)
            X_input, _ = generate_fgsm_attack(
                model=lstm_model,
                X=X_input,
                y=dummy_target,
                epsilon=epsilon,
                sensor_only=True
            )

        pred_lstm = lstm_model.predict(X_input, verbose=0)
        pred_rnn = rnn_model.predict(X_input, verbose=0)
        pred_avg = (pred_lstm + pred_rnn) / 2.0

        temperature_pred = float(target_scaler.inverse_transform(pred_avg)[0][0])
        lstm_pred_deg = float(target_scaler.inverse_transform(pred_lstm)[0][0])
        rnn_pred_deg = float(target_scaler.inverse_transform(pred_rnn)[0][0])

        actual_temp = None
        if input_time in df_with_idx.index:
            actual_row = df_with_idx.loc[input_time]
            if isinstance(actual_row, pd.DataFrame):
                actual_temp = float(actual_row['Temperature (C)'].iloc[0])
            else:
                actual_temp = float(actual_row['Temperature (C)'])

        if verbose:
            tag = f"[ATTACKED ε={epsilon}]" if epsilon > 0 else "[CLEAN FORECAST]"
            print(f"\n{tag} Prediction for {input_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"LSTM Forecast:          {lstm_pred_deg:.2f}°C")
            print(f"RNN Forecast:           {rnn_pred_deg:.2f}°C")
            print(f"Ensemble Forecast:      {temperature_pred:.2f}°C")
            if actual_temp is not None:
                abs_error = abs(actual_temp - temperature_pred)
                accuracy = 100 - (abs_error / abs(actual_temp)) * 100 if actual_temp != 0 else 0
                print(f"Actual Ground Truth:    {actual_temp:.2f}°C")
                print(f"Absolute Error:         {abs_error:.2f}°C")
                print(f"Accuracy Metric:        {accuracy:.2f}%")
            else:
                print("Notice: Actual temperature not recorded at this exact timestamp.")

        return {
            "prediction": temperature_pred,
            "lstm_prediction": lstm_pred_deg,
            "rnn_prediction": rnn_pred_deg,
            "actual": actual_temp
        }

    except Exception as e:
        if verbose:
            print("Invalid input or prediction error:", str(e))
        return None

