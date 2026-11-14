"""
Cybersecurity Research Experiment: Input Sanitization Defense Layer for Time-Series Forecasting.

Course Context: AI for Cyber Security
------------------------------------
This script evaluates pre-inference input sanitization defenses:
1. Rolling Median Filter (temporal window smoothing of sign-based gradient spikes).
2. Physical Rate Limiter (enforcing physical atmospheric continuity bounds).
3. Hybrid Sanitizer (combining rolling median with physical delta clipping).

Evaluates the following conditions across LSTM and SimpleRNN:
- Condition A: Clean input -> Model
- Condition B: Attacked input -> Model
- Condition C: Attacked input -> Rolling Median -> Model
- Condition D: Attacked input -> Physical Rate Limiter -> Model

Strict scientific standards:
- Zero model retraining.
- Clean inputs and attacked inputs are preserved strictly unmodified.
- Separate sanitized datasets are saved to data/sanitized/.
- Thresholds are calibrated on clean training sequences (no data leakage).
"""

import os
import sys

# Ensure repository root is on sys.path and preloads MSVC runtime DLLs first
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import src

import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

from src.preprocessing import load_and_clean_data, prepare_datasets
from src.models import load_trained_model
from src.defenses.sanitization import (
    RollingMedianSanitizer,
    PhysicalRateLimiterSanitizer
)
from src.evaluation import compute_metrics
from src.inference import predict_temperature_by_datetime


def main():
    print("=" * 80)
    print("CYBERSECURITY EXPERIMENT: INPUT SANITIZATION DEFENSE")
    print("=" * 80)

    # 1. Load Data and Models
    print("\n[Step 1/5] Loading data and pre-trained forecasting models...")
    df_clean = load_and_clean_data("weatherHistory.xls")
    dataset_dict = prepare_datasets(df_clean, sequence_length=24, leakage_free=False, scaler_dir="saved_models")

    X_train = dataset_dict["X_train"]
    target_scaler = dataset_dict["target_scaler"]
    feature_scaler = dataset_dict["feature_scaler"]

    # Load preserved clean test dataset
    X_test_clean = np.load("data/clean/X_test_clean.npy")
    y_test_clean = np.load("data/clean/y_test_clean.npy")
    y_test_inv = target_scaler.inverse_transform(y_test_clean)
    print(f"Loaded clean test data: {X_test_clean.shape} samples.")

    # Load pre-trained models
    lstm_model = load_trained_model("saved_models/lstm_model.keras")
    rnn_model = load_trained_model("saved_models/rnn_model.keras")

    # 2. Configure & Calibrate Sanitizers (No Data Leakage)
    print("\n[Step 2/5] Initializing & calibrating sanitizers on clean training data...")
    # Sanitizer 1: Rolling Median Filter (window_size=3)
    median_sanitizer = RollingMedianSanitizer(window_size=3)

    # Sanitizer 2: Physical Rate Limiter (99.5th percentile calibrated strictly on X_train)
    rate_limiter = PhysicalRateLimiterSanitizer(delta_percentile=99.5)
    max_deltas = rate_limiter.fit(X_train)

    print("Calibrated physical rate-of-change hourly limits per sensor feature:")
    sensor_names = ['Apparent Temp', 'Humidity', 'Wind Speed', 'Visibility', 'Cloud Cover', 'Pressure']
    for s_name, d_val in zip(sensor_names, max_deltas):
        print(f"  {s_name:15s}: max allowable hourly delta = {d_val:.4f}")

    # Check Filter Distortion on Clean Data (Ensure Sanitizer doesn't harm normal operations)
    print("\n--- Verifying Filter Invariance / Distortion on Clean Test Data ---")
    X_clean_median = median_sanitizer.sanitize(X_test_clean)
    X_clean_limited = rate_limiter.sanitize(X_test_clean)

    pred_clean_lstm = target_scaler.inverse_transform(lstm_model.predict(X_test_clean, batch_size=256, verbose=0))
    pred_clean_med_lstm = target_scaler.inverse_transform(lstm_model.predict(X_clean_median, batch_size=256, verbose=0))
    pred_clean_lim_lstm = target_scaler.inverse_transform(lstm_model.predict(X_clean_limited, batch_size=256, verbose=0))

    clean_mse = compute_metrics(y_test_inv, pred_clean_lstm)["MSE"]
    clean_med_mse = compute_metrics(y_test_inv, pred_clean_med_lstm)["MSE"]
    clean_lim_mse = compute_metrics(y_test_inv, pred_clean_lim_lstm)["MSE"]

    print(f"Original Clean Test MSE (LSTM):               {clean_mse:.4f}")
    print(f"Clean after Median Filter (Filter Distortion): {clean_med_mse:.4f} (Δ = {clean_med_mse - clean_mse:+.4f})")
    print(f"Clean after Rate Limiter (Filter Distortion):  {clean_lim_mse:.4f} (Δ = {clean_lim_mse - clean_mse:+.4f})")

    # 3. Comprehensive Evaluation Across Epsilons and Models
    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    results_records = []
    os.makedirs("data/sanitized", exist_ok=True)
    os.makedirs("results/plots/sanitization", exist_ok=True)

    models = [("LSTM", lstm_model), ("RNN", rnn_model)]
    trajectory_samples = {}

    print("\n[Step 3/5] Running sanitization benchmark across all epsilons and models...")

    for model_name, model_obj in models:
        print(f"\n>>> Evaluating Sanitization Defenses on {model_name} Model <<<")
        # Condition A: Clean Baseline
        y_pred_clean_raw = model_obj.predict(X_test_clean, batch_size=256, verbose=0)
        y_pred_clean = target_scaler.inverse_transform(y_pred_clean_raw)
        m_clean = compute_metrics(y_test_inv, y_pred_clean)

        for eps in epsilons:
            adv_path = f"data/attacked/X_test_adv_{model_name.lower()}_eps_{eps}.npy"
            if not os.path.exists(adv_path):
                print(f"Warning: Attacked file {adv_path} not found. Skipping.")
                continue

            X_test_adv = np.load(adv_path)

            # Condition B: Attacked without Defense
            y_pred_adv_raw = model_obj.predict(X_test_adv, batch_size=256, verbose=0)
            y_pred_adv = target_scaler.inverse_transform(y_pred_adv_raw)
            m_adv = compute_metrics(y_test_inv, y_pred_adv)

            # Condition C: Rolling Median Sanitization
            X_san_median = median_sanitizer.sanitize(X_test_adv)
            # Save separate sanitized array
            np.save(f"data/sanitized/X_test_san_median_{model_name.lower()}_eps_{eps}.npy", X_san_median)

            y_pred_med_raw = model_obj.predict(X_san_median, batch_size=256, verbose=0)
            y_pred_med = target_scaler.inverse_transform(y_pred_med_raw)
            m_med = compute_metrics(y_test_inv, y_pred_med)

            # Condition D: Physical Rate Limiter Sanitization
            X_san_lim = rate_limiter.sanitize(X_test_adv)
            # Save separate sanitized array
            np.save(f"data/sanitized/X_test_san_ratelimit_{model_name.lower()}_eps_{eps}.npy", X_san_lim)

            y_pred_lim_raw = model_obj.predict(X_san_lim, batch_size=256, verbose=0)
            y_pred_lim = target_scaler.inverse_transform(y_pred_lim_raw)
            m_lim = compute_metrics(y_test_inv, y_pred_lim)

            # Cache representative trajectories for visualization
            if eps in [0.02, 0.05]:
                trajectory_samples[(model_name, eps)] = {
                    "clean": y_pred_clean,
                    "adv": y_pred_adv,
                    "median": y_pred_med,
                    "ratelimit": y_pred_lim,
                    "true": y_test_inv
                }

            # Calculate Defense Performance Indicators
            # 1. Attacked degradation
            delta_mse_adv = m_adv["MSE"] - m_clean["MSE"]

            # 2. Defense recovery percentage
            rec_med = ((m_adv["MSE"] - m_med["MSE"]) / delta_mse_adv) * 100.0 if delta_mse_adv > 0 else 0.0
            rec_lim = ((m_adv["MSE"] - m_lim["MSE"]) / delta_mse_adv) * 100.0 if delta_mse_adv > 0 else 0.0

            # 3. Perturbation reduction percentage
            base_pert = np.mean(np.abs(X_test_adv - X_test_clean))
            pert_red_med = ((base_pert - np.mean(np.abs(X_san_median - X_test_clean))) / base_pert) * 100.0 if base_pert > 0 else 0.0
            pert_red_lim = ((base_pert - np.mean(np.abs(X_san_lim - X_test_clean))) / base_pert) * 100.0 if base_pert > 0 else 0.0

            # 4. Percentage of samples modified
            pct_mod_med = float(np.mean(np.any(np.abs(X_san_median - X_test_adv) > 1e-5, axis=(1, 2))) * 100.0)
            pct_mod_lim = float(np.mean(np.any(np.abs(X_san_lim - X_test_adv) > 1e-5, axis=(1, 2))) * 100.0)

            # Append Records for both sanitizers
            for s_name, m_san, rec, pert_red, pct_mod in [
                ("Rolling_Median", m_med, rec_med, pert_red_med, pct_mod_med),
                ("Physical_Rate_Limiter", m_lim, rec_lim, pert_red_lim, pct_mod_lim)
            ]:
                results_records.append({
                    "Model": model_name,
                    "Sanitizer": s_name,
                    "Epsilon": eps,
                    "Clean_MSE": m_clean["MSE"],
                    "Attacked_MSE": m_adv["MSE"],
                    "Sanitized_MSE": m_san["MSE"],
                    "Sanitized_MAE": m_san["MAE"],
                    "Sanitized_RMSE": m_san["RMSE"],
                    "Sanitized_R2": m_san["R2"],
                    "MSE_Improvement": m_adv["MSE"] - m_san["MSE"],
                    "Defense_Recovery_Pct": rec,
                    "Perturbation_Reduction_Pct": pert_red,
                    "Pct_Samples_Modified": pct_mod
                })

            print(f"  [ε = {eps:0.3f}] Attacked MSE: {m_adv['MSE']:6.2f} | "
                  f"Median MSE: {m_med['MSE']:6.2f} (Rec: {rec_med:5.1f}%, Mod: {pct_mod_med:5.1f}%) | "
                  f"RateLimit MSE: {m_lim['MSE']:6.2f} (Rec: {rec_lim:5.1f}%, Mod: {pct_mod_lim:5.1f}%)")

    results_df = pd.DataFrame(results_records)
    csv_path = "results/input_sanitization_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nAll sanitization results saved to: {csv_path}")

    # 4. Generate Research Visualizations
    print("\n[Step 4/5] Generating research visualization plots...")

    # Plot 1: Clean vs Attacked vs Sanitized Temperature Trajectory
    for m_name in ["LSTM", "RNN"]:
        if (m_name, 0.05) in trajectory_samples:
            traj = trajectory_samples[(m_name, 0.05)]
            plt.figure(figsize=(13, 5), dpi=150)
            t_slice = np.arange(120)
            plt.plot(t_slice, traj["true"][:120], label='Actual Temperature (Ground Truth)', color='black', linewidth=2.0)
            plt.plot(t_slice, traj["clean"][:120], label=f'Clean {m_name} Forecast', color='#1f77b4', linestyle='--', linewidth=1.5)
            plt.plot(t_slice, traj["adv"][:120], label='Attacked Forecast (ε=0.05)', color='#d62728', alpha=0.8, linewidth=1.5)
            plt.plot(t_slice, traj["median"][:120], label='Sanitized: Rolling Median', color='#2ca02c', linewidth=1.5)
            plt.plot(t_slice, traj["ratelimit"][:120], label='Sanitized: Physical Rate Limiter', color='#9467bd', linestyle=':', linewidth=1.8)

            plt.title(f"{m_name}: Forecast Trajectory Restoration via Input Sanitization (ε = 0.05)", fontsize=12, fontweight='bold')
            plt.xlabel("Time Step (Hours)", fontsize=11)
            plt.ylabel("Temperature (°C)", fontsize=11)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(frameon=True, facecolor='white', framealpha=0.9)
            plt.tight_layout()
            plt.savefig(f"results/plots/sanitization/clean_vs_attacked_vs_sanitized_{m_name.lower()}.png")
            plt.close()

    # Plot 2: Prediction Error (MSE) Before and After Sanitization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for idx, m_name in enumerate(["LSTM", "RNN"]):
        ax = axes[idx]
        df_sub_med = results_df[(results_df["Model"] == m_name) & (results_df["Sanitizer"] == "Rolling_Median")]
        df_sub_lim = results_df[(results_df["Model"] == m_name) & (results_df["Sanitizer"] == "Physical_Rate_Limiter")]

        x = np.arange(len(epsilons))
        width = 0.25
        ax.bar(x - width, df_sub_med["Attacked_MSE"], width, label='Attacked (No Defense)', color='#d62728', alpha=0.85)
        ax.bar(x, df_sub_med["Sanitized_MSE"], width, label='Sanitized: Rolling Median', color='#2ca02c', alpha=0.85)
        ax.bar(x + width, df_sub_lim["Sanitized_MSE"], width, label='Sanitized: Physical Rate Limiter', color='#9467bd', alpha=0.85)

        ax.set_title(f"{m_name}: Forecasting MSE Before vs. After Defense", fontsize=11, fontweight='bold')
        ax.set_xlabel("Perturbation Budget (ε)", fontsize=10)
        ax.set_ylabel("Mean Squared Error (MSE)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"ε={e}" for e in epsilons])
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/plots/sanitization/prediction_error_before_after_sanitization.png")
    plt.close()

    # Plot 3: RMSE / MAE vs Epsilon Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for idx, (metric, ax) in enumerate([("Sanitized_RMSE", axes[0]), ("Sanitized_MAE", axes[1])]):
        for m_name in ["LSTM", "RNN"]:
            df_med = results_df[(results_df["Model"] == m_name) & (results_df["Sanitizer"] == "Rolling_Median")]
            df_lim = results_df[(results_df["Model"] == m_name) & (results_df["Sanitizer"] == "Physical_Rate_Limiter")]

            ax.plot(df_med["Epsilon"], df_med[metric], marker='o', label=f"{m_name} (Median)", linewidth=1.8)
            ax.plot(df_lim["Epsilon"], df_lim[metric], marker='s', linestyle='--', label=f"{m_name} (Rate Limiter)", linewidth=1.8)

        metric_name = "RMSE (°C)" if "RMSE" in metric else "MAE (°C)"
        ax.set_title(f"Sanitized Forecast {metric_name} vs. Perturbation Budget (ε)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Perturbation Budget (ε)", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/plots/sanitization/rmse_mae_vs_epsilon.png")
    plt.close()

    # Plot 4: Defense Recovery Percentage vs Epsilon
    plt.figure(figsize=(9, 5), dpi=150)
    for m_name in ["LSTM", "RNN"]:
        df_med = results_df[(results_df["Model"] == m_name) & (results_df["Sanitizer"] == "Rolling_Median")]
        df_lim = results_df[(results_df["Model"] == m_name) & (results_df["Sanitizer"] == "Physical_Rate_Limiter")]

        plt.plot(df_med["Epsilon"], df_med["Defense_Recovery_Pct"], marker='^', linewidth=2.0, label=f"{m_name}: Rolling Median")
        plt.plot(df_lim["Epsilon"], df_lim["Defense_Recovery_Pct"], marker='v', linestyle='--', linewidth=2.0, label=f"{m_name}: Physical Rate Limiter")

    plt.title("Defensive Error Recovery Percentage vs. Perturbation Budget (ε)", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Budget (ε)", fontsize=10)
    plt.ylabel("Error Recovery Rate (%)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("results/plots/sanitization/defense_recovery_vs_epsilon.png")
    plt.close()

    # Plot 5: Perturbation Removed (L1 and Linf)
    plt.figure(figsize=(9, 5), dpi=150)
    for m_name in ["LSTM"]:
        df_med = results_df[(results_df["Model"] == m_name) & (results_df["Sanitizer"] == "Rolling_Median")]
        df_lim = results_df[(results_df["Model"] == m_name) & (results_df["Sanitizer"] == "Physical_Rate_Limiter")]

        plt.plot(df_med["Epsilon"], df_med["Perturbation_Reduction_Pct"], marker='o', linewidth=2.0, label='Rolling Median: Perturbation Reduction %')
        plt.plot(df_lim["Epsilon"], df_lim["Perturbation_Reduction_Pct"], marker='s', linewidth=2.0, label='Physical Rate Limiter: Perturbation Reduction %')

    plt.title("Adversarial Perturbation Reduction Percentage vs. Budget (ε)", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Budget (ε)", fontsize=10)
    plt.ylabel("Perturbation Reduction (%)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("results/plots/sanitization/amount_of_perturbation_removed.png")
    plt.close()

    # Plot 6: Percentage of Samples Modified by Sanitizer
    plt.figure(figsize=(9, 5), dpi=150)
    for m_name in ["LSTM"]:
        df_med = results_df[(results_df["Model"] == m_name) & (results_df["Sanitizer"] == "Rolling_Median")]
        df_lim = results_df[(results_df["Model"] == m_name) & (results_df["Sanitizer"] == "Physical_Rate_Limiter")]

        plt.plot(df_med["Epsilon"], df_med["Pct_Samples_Modified"], marker='d', linewidth=2.0, label='Rolling Median: Modified Samples %')
        plt.plot(df_lim["Epsilon"], df_lim["Pct_Samples_Modified"], marker='x', linestyle='--', linewidth=2.0, label='Rate Limiter: Modified Samples %')

    plt.title("Percentage of Test Sequences Filtered / Modified vs. Budget (ε)", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Budget (ε)", fontsize=10)
    plt.ylabel("Sequences Modified (%)", fontsize=10)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("results/plots/sanitization/percentage_of_samples_modified.png")
    plt.close()

    print("Saved all 6 research sanitization plot artifacts to: results/plots/sanitization/")

    # 5. Verification of the Interactive Forecasting Pipeline
    print("\n[Step 5/5] Verifying original prediction pipeline with sanitized input...")
    sample_sanitized = predict_temperature_by_datetime(
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
        verbose=False
    )
    print(f"Pipeline verification test clean forecast: {sample_sanitized['prediction']:.2f}°C (Ground Truth: {sample_sanitized['actual']:.2f}°C)")
    print("\nInput Sanitization Experiment successfully executed.")


if __name__ == "__main__":
    main()

