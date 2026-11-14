"""
Cybersecurity Research Experiment: Adversarial Attacks on Multivariate Temperature Forecasting.

Course Context: AI for Cyber Security
------------------------------------
This script evaluates the adversarial vulnerability of recurrent deep learning architectures 
(LSTM and SimpleRNN) against evasion attacks. It crafts Fast Gradient Sign Method (FGSM) 
adversarial perturbations strictly against local multivariate weather test sequences, 
measuring forecast degradation under varying perturbation budgets (epsilon).

Safety and Scope:
-----------------
- Strictly offline and local.
- Operates only on local test splits and saved local neural network weights.
- Preserves the clean test dataset intact and persists attacked datasets separately.
"""

import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import src FIRST to configure MSVC runtime preloading for Windows TensorFlow
import src

import random
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Set fixed random seeds for complete experimental reproducibility
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
from src.preprocessing import load_and_clean_data, prepare_datasets
from src.models import (
    build_lstm_model,
    build_rnn_model,
    train_lstm,
    train_rnn,
    save_trained_model,
    load_trained_model
)
from src.attacks.fgsm import generate_fgsm_attack
from src.evaluation import (
    compute_metrics,
    compute_perturbation_stats,
    compute_attack_impact,
    plot_clean_vs_attacked_trajectory,
    plot_error_vs_epsilon,
    plot_prediction_degradation,
    plot_perturbation_magnitude
)
from src.inference import predict_temperature_by_datetime


def main():
    print("=" * 80)
    print("CYBERSECURITY EXPERIMENT: ADVERSARIAL ATTACKS ON TIME-SERIES FORECASTING")
    print("=" * 80)

    # 1. Dataset Loading and Preparation
    print("\n[Step 1/5] Loading and preparing chronological datasets...")
    df_clean = load_and_clean_data("weatherHistory.xls")
    print(f"Loaded dataset: {df_clean.shape[0]} samples, {df_clean.shape[1]} columns.")

    # We use leakage_free=False here to maintain exact baseline continuity with the original notebook
    dataset_dict = prepare_datasets(df_clean, sequence_length=24, leakage_free=False, scaler_dir="saved_models")

    X_train, y_train = dataset_dict["X_train"], dataset_dict["y_train"]
    X_val, y_val = dataset_dict["X_val"], dataset_dict["y_val"]
    X_test, y_test = dataset_dict["X_test"], dataset_dict["y_test"]
    target_scaler = dataset_dict["target_scaler"]
    feature_scaler = dataset_dict["feature_scaler"]

    # Persist clean test set separately to ensure data integrity
    os.makedirs("data/clean", exist_ok=True)
    np.save("data/clean/X_test_clean.npy", X_test)
    np.save("data/clean/y_test_clean.npy", y_test)
    print(f"Clean test sequences saved: {X_test.shape} samples.")

    # 2. Model Initialization / Loading
    print("\n[Step 2/5] Initializing / Loading forecasting models...")
    lstm_path = "saved_models/lstm_model.keras"
    rnn_path = "saved_models/rnn_model.keras"

    if os.path.exists(lstm_path):
        print("Loading pre-trained LSTM model...")
        lstm_model = load_trained_model(lstm_path)
    else:
        print("Training baseline LSTM model (preserving notebook parameters)...")
        lstm_model = build_lstm_model(input_shape=(24, 10))
        train_lstm(lstm_model, X_train, y_train, X_val, y_val, epochs=12, batch_size=256, patience=4)
        save_trained_model(lstm_model, lstm_path)

    if os.path.exists(rnn_path):
        print("Loading pre-trained RNN model...")
        rnn_model = load_trained_model(rnn_path)
    else:
        print("Training baseline RNN model (preserving notebook parameters)...")
        rnn_model = build_rnn_model(input_shape=(24, 10))
        train_rnn(rnn_model, X_train, y_train, X_val, y_val, epochs=12, batch_size=256, patience=4)
        save_trained_model(rnn_model, rnn_path)

    # 3. Clean Baseline Evaluation
    print("\n[Step 3/5] Evaluating clean baseline performance...")
    y_test_inv = target_scaler.inverse_transform(y_test)

    # Clean predictions
    y_pred_lstm_norm = lstm_model.predict(X_test, batch_size=256, verbose=0)
    y_pred_lstm_clean = target_scaler.inverse_transform(y_pred_lstm_norm)
    lstm_clean_metrics = compute_metrics(y_test_inv, y_pred_lstm_clean)

    y_pred_rnn_norm = rnn_model.predict(X_test, batch_size=256, verbose=0)
    y_pred_rnn_clean = target_scaler.inverse_transform(y_pred_rnn_norm)
    rnn_clean_metrics = compute_metrics(y_test_inv, y_pred_rnn_clean)

    print("--- Clean Baseline Metrics ---")
    print(f"LSTM -> MSE: {lstm_clean_metrics['MSE']:.4f}, MAE: {lstm_clean_metrics['MAE']:.4f}, RMSE: {lstm_clean_metrics['RMSE']:.4f}, R²: {lstm_clean_metrics['R2']:.4f}")
    print(f"RNN  -> MSE: {rnn_clean_metrics['MSE']:.4f}, MAE: {rnn_clean_metrics['MAE']:.4f}, RMSE: {rnn_clean_metrics['RMSE']:.4f}, R²: {rnn_clean_metrics['R2']:.4f}")

    # 4. Adversarial Attack Evaluation Across Perturbation Strengths
    epsilons = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    print(f"\n[Step 4/5] Running FGSM Attack across epsilon budgets: {epsilons}...")

    models = [("LSTM", lstm_model, lstm_clean_metrics, y_pred_lstm_clean),
              ("RNN", rnn_model, rnn_clean_metrics, y_pred_rnn_clean)]

    results_records = []
    os.makedirs("data/attacked", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Store predictions for plotting
    plotted_predictions = {}

    for model_name, model_obj, clean_metrics, clean_preds in models:
        print(f"\n>>> Executing Adversarial Attack on {model_name} Model <<<")
        for eps in epsilons:
            t0 = time.time()
            # Generate FGSM attack
            X_adv, perturbation = generate_fgsm_attack(
                model=model_obj,
                X=X_test,
                y=y_test,
                epsilon=eps,
                sensor_only=True,
                batch_size=256
            )
            attack_time = time.time() - t0

            # Perturbation statistics
            pert_stats = compute_perturbation_stats(perturbation)

            # Save separate attacked test dataset
            adv_filename = f"data/attacked/X_test_adv_{model_name.lower()}_eps_{eps}.npy"
            np.save(adv_filename, X_adv)

            # Model inference on perturbed input
            y_pred_adv_norm = model_obj.predict(X_adv, batch_size=256, verbose=0)
            y_pred_adv = target_scaler.inverse_transform(y_pred_adv_norm)

            # Store predictions at key epsilon for time-series visualization
            if eps in [0.01, 0.05]:
                plotted_predictions[(model_name, eps)] = y_pred_adv

            # Calculate error metrics
            adv_metrics = compute_metrics(y_test_inv, y_pred_adv)

            # Calculate attack impact compared with clean baseline
            impact = compute_attack_impact(clean_metrics, adv_metrics)

            record = {
                "Model": model_name,
                "Epsilon": eps,
                "MSE": adv_metrics["MSE"],
                "MAE": adv_metrics["MAE"],
                "RMSE": adv_metrics["RMSE"],
                "R2": adv_metrics["R2"],
                "Delta_MSE": impact["Delta_MSE"],
                "Pct_Inc_MSE": impact["Pct_Inc_MSE"],
                "Delta_MAE": impact["Delta_MAE"],
                "Pct_Inc_MAE": impact["Pct_Inc_MAE"],
                "Delta_RMSE": impact["Delta_RMSE"],
                "Pct_Inc_RMSE": impact["Pct_Inc_RMSE"],
                "R2_Degradation": impact["R2_Degradation"],
                "L_inf": pert_stats["L_inf"],
                "L_1_mean": pert_stats["L_1_mean"],
                "L_2_mean": pert_stats["L_2_mean"],
                "Gen_Time_Sec": round(attack_time, 2)
            }
            results_records.append(record)

            print(f"  [ε = {eps:0.3f}] MSE: {adv_metrics['MSE']:.4f} (+{impact['Pct_Inc_MSE']:6.1f}%) | "
                  f"MAE: {adv_metrics['MAE']:.4f} | RMSE: {adv_metrics['RMSE']:.4f} | R²: {adv_metrics['R2']:.4f} | "
                  f"Mean L2: {pert_stats['L_2_mean']:.4f}")

    results_df = pd.DataFrame(results_records)
    csv_path = "results/adversarial_attack_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nAll attack results saved to: {csv_path}")

    # 5. Visualization Generation
    print("\n[Step 5/5] Generating research visualization plots...")
    for model_name in ["LSTM", "RNN"]:
        for eps in [0.01, 0.05]:
            if (model_name, eps) in plotted_predictions:
                y_adv = plotted_predictions[(model_name, eps)]
                clean_pred = y_pred_lstm_clean if model_name == "LSTM" else y_pred_rnn_clean
                plot_clean_vs_attacked_trajectory(
                    y_true=y_test_inv,
                    y_clean=clean_pred,
                    y_adv=y_adv,
                    epsilon=eps,
                    model_name=model_name,
                    sample_count=120,
                    save_path=f"results/plots/timeseries_{model_name.lower()}_eps_{eps}.png"
                )

    plot_error_vs_epsilon(results_df, save_path="results/plots/error_vs_epsilon.png")
    plot_prediction_degradation(results_df, save_path="results/plots/prediction_degradation.png")
    plot_perturbation_magnitude(results_df, save_path="results/plots/perturbation_magnitude.png")
    print("Saved all 4 plot types to: results/plots/")

    # 6. Verification of the Interactive Clean Pipeline
    print("\n--- Verifying Original User Date-Time Pipeline ---")
    test_sample = predict_temperature_by_datetime(
        df_clean=df_clean,
        lstm_model=lstm_model,
        rnn_model=rnn_model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        year=2009,
        month=11,
        day=24,
        hour=12,
        epsilon=0.0,
        verbose=True
    )

    print("\n--- Testing Same Timestamp Under Simulated Adversarial Perturbation (ε = 0.02) ---")
    attack_sample = predict_temperature_by_datetime(
        df_clean=df_clean,
        lstm_model=lstm_model,
        rnn_model=rnn_model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        year=2009,
        month=11,
        day=24,
        hour=12,
        epsilon=0.02,
        verbose=True
    )

    print("\nAdversarial Attack Experiment successfully completed.")


if __name__ == "__main__":
    main()

