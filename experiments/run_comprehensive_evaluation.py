"""
Cybersecurity Research: Comprehensive Evaluation of Attacks & Defenses on Time-Series Forecasting.

Course Context: AI for Cyber Security
------------------------------------
This is the master benchmark script comparing all stages of the research:
1. Clean Baseline (Original model on clean weather sequences)
2. Under Adversarial Attack (FGSM evasion attack without defense)
3. Input Sanitization Defense (Rolling Median & Physical Rate Limiter)
4. Adversarial Training Defense (Retrained robust LSTM and SimpleRNN)
5. Defense-in-Depth (Sanitization + Adversarial Training combined)

Outputs:
- Persisted robust models to saved_models/
- Master benchmark metrics table to results/comprehensive_cybersecurity_benchmark.csv
- Publication-quality comparative plots to results/plots/comparative/
- Real-time verified interactive date-time prediction pipeline
"""

import os
import sys

# Preload MSVC runtime DLLs first
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import src

import random
import time
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
from src.models import (
    build_lstm_model,
    build_rnn_model,
    load_trained_model,
    save_trained_model
)
from src.defenses.sanitization import (
    RollingMedianSanitizer,
    PhysicalRateLimiterSanitizer
)
from src.defenses.adversarial_training import train_adversarially
from src.evaluation import compute_metrics
from src.inference import predict_temperature_by_datetime


def main():
    print("=" * 80)
    print("CYBERSECURITY RESEARCH: COMPREHENSIVE ATTACK & DEFENSE BENCHMARK")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1/6] Loading data and preparing datasets...")
    df_clean = load_and_clean_data("weatherHistory.xls")
    dataset_dict = prepare_datasets(df_clean, sequence_length=24, leakage_free=False, scaler_dir="saved_models")

    X_train, y_train = dataset_dict["X_train"], dataset_dict["y_train"]
    X_val, y_val = dataset_dict["X_val"], dataset_dict["y_val"]
    target_scaler = dataset_dict["target_scaler"]
    feature_scaler = dataset_dict["feature_scaler"]

    X_test_clean = np.load("data/clean/X_test_clean.npy")
    y_test_clean = np.load("data/clean/y_test_clean.npy")
    y_test_inv = target_scaler.inverse_transform(y_test_clean)
    print(f"Loaded clean evaluation dataset: {X_test_clean.shape} sequences.")

    # 2. Load Clean Baseline Models
    print("\n[Step 2/6] Loading clean baseline models...")
    lstm_clean = load_trained_model("saved_models/lstm_model.keras")
    rnn_clean = load_trained_model("saved_models/rnn_model.keras")

    # 3. Train or Load Adversarially Trained Models
    print("\n[Step 3/6] Initializing / Loading adversarially trained models...")
    lstm_adv_path = "saved_models/lstm_adv_trained.keras"
    rnn_adv_path = "saved_models/rnn_adv_trained.keras"

    if os.path.exists(lstm_adv_path):
        print("Loading pre-trained robust LSTM model...")
        lstm_robust = load_trained_model(lstm_adv_path)
    else:
        print("Training robust LSTM with online adversarial training (ε_train=0.02, α=0.5)...")
        lstm_robust = build_lstm_model(input_shape=(24, 10))
        train_adversarially(lstm_robust, X_train, y_train, X_val, y_val, epsilon_train=0.02, alpha=0.5, epochs=6, batch_size=256, patience=3)
        save_trained_model(lstm_robust, lstm_adv_path)

    if os.path.exists(rnn_adv_path):
        print("Loading pre-trained robust RNN model...")
        rnn_robust = load_trained_model(rnn_adv_path)
    else:
        print("Training robust RNN with online adversarial training (ε_train=0.02, α=0.5)...")
        rnn_robust = build_rnn_model(input_shape=(24, 10))
        train_adversarially(rnn_robust, X_train, y_train, X_val, y_val, epsilon_train=0.02, alpha=0.5, epochs=6, batch_size=256, patience=3)
        save_trained_model(rnn_robust, rnn_adv_path)

    # 4. Initialize Sanitizers
    median_sanitizer = RollingMedianSanitizer(window_size=3)
    rate_limiter = PhysicalRateLimiterSanitizer(delta_percentile=99.5)
    rate_limiter.fit(X_train)

    # 5. Master Comparative Evaluation Across All Scenarios
    print("\n[Step 4/6] Running master multi-scenario evaluation...")
    epsilons = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    records = []
    trajectories = {}

    models_suite = [
        ("LSTM", lstm_clean, lstm_robust),
        ("RNN", rnn_clean, rnn_robust)
    ]

    for model_name, clean_m, robust_m in models_suite:
        print(f"\n>>> Benchmarking All Scenarios for {model_name} <<<")
        # Clean baseline metrics
        pred_clean_base = target_scaler.inverse_transform(clean_m.predict(X_test_clean, batch_size=256, verbose=0))
        m_base_clean = compute_metrics(y_test_inv, pred_clean_base)

        for eps in epsilons:
            if eps == 0.0:
                X_adv = np.copy(X_test_clean)
            else:
                X_adv = np.load(f"data/attacked/X_test_adv_{model_name.lower()}_eps_{eps}.npy")

            # 1. Undefended Baseline Model
            pred_adv = target_scaler.inverse_transform(clean_m.predict(X_adv, batch_size=256, verbose=0))
            m_adv = compute_metrics(y_test_inv, pred_adv)

            # 2. Input Sanitization: Rolling Median
            X_med = median_sanitizer.sanitize(X_adv)
            pred_med = target_scaler.inverse_transform(clean_m.predict(X_med, batch_size=256, verbose=0))
            m_med = compute_metrics(y_test_inv, pred_med)

            # 3. Input Sanitization: Physical Rate Limiter
            X_lim = rate_limiter.sanitize(X_adv)
            pred_lim = target_scaler.inverse_transform(clean_m.predict(X_lim, batch_size=256, verbose=0))
            m_lim = compute_metrics(y_test_inv, pred_lim)

            # 4. Adversarial Training Defense
            pred_rob = target_scaler.inverse_transform(robust_m.predict(X_adv, batch_size=256, verbose=0))
            m_rob = compute_metrics(y_test_inv, pred_rob)

            # 5. Defense-in-Depth (Sanitization + Adversarial Training)
            pred_did = target_scaler.inverse_transform(robust_m.predict(X_lim, batch_size=256, verbose=0))
            m_did = compute_metrics(y_test_inv, pred_did)

            # Cache trajectory at eps=0.05 for publication plot
            if eps == 0.05:
                trajectories[model_name] = {
                    "true": y_test_inv,
                    "clean": pred_clean_base,
                    "adv": pred_adv,
                    "median": pred_med,
                    "ratelimit": pred_lim,
                    "adv_trained": pred_rob,
                    "defense_in_depth": pred_did
                }

            # Calculate Recovery % relative to attack degradation
            delta_attack = m_adv["MSE"] - m_base_clean["MSE"]

            def get_rec(mse_def):
                if delta_attack <= 0:
                    return 0.0
                return ((m_adv["MSE"] - mse_def) / delta_attack) * 100.0

            scenarios = [
                ("Attacked (No Defense)", m_adv, 0.0),
                ("Defense: Rolling Median", m_med, get_rec(m_med["MSE"])),
                ("Defense: Physical Rate Limiter", m_lim, get_rec(m_lim["MSE"])),
                ("Defense: Adversarial Training", m_rob, get_rec(m_rob["MSE"])),
                ("Defense: Defense-in-Depth (Hybrid)", m_did, get_rec(m_did["MSE"]))
            ]

            for s_name, m_dict, rec_pct in scenarios:
                records.append({
                    "Model": model_name,
                    "Scenario": s_name,
                    "Epsilon": eps,
                    "MSE": m_dict["MSE"],
                    "MAE": m_dict["MAE"],
                    "RMSE": m_dict["RMSE"],
                    "R2": m_dict["R2"],
                    "Defense_Recovery_Pct": rec_pct
                })

            print(f"  [ε = {eps:0.3f}] Attacked MSE: {m_adv['MSE']:6.2f} | "
                  f"Median: {m_med['MSE']:6.2f} (Rec: {get_rec(m_med['MSE']):5.1f}%) | "
                  f"AdvTrain: {m_rob['MSE']:6.2f} (Rec: {get_rec(m_rob['MSE']):5.1f}%) | "
                  f"DefenseInDepth: {m_did['MSE']:6.2f} (Rec: {get_rec(m_did['MSE']):5.1f}%)")

    results_df = pd.DataFrame(records)
    csv_path = "results/comprehensive_cybersecurity_benchmark.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nMaster benchmark metrics saved to: {csv_path}")

    # 6. Generate Comparative Visualizations
    print("\n[Step 5/6] Generating publication-quality comparative visualizations...")
    os.makedirs("results/plots/comparative", exist_ok=True)

    # Plot 1: Comprehensive Multi-Scenario Trajectory Restoration
    for m_name in ["LSTM", "RNN"]:
        if m_name in trajectories:
            traj = trajectories[m_name]
            plt.figure(figsize=(14, 5.5), dpi=150)
            t_steps = np.arange(120)

            plt.plot(t_steps, traj["true"][:120], label='Actual Temperature (Ground Truth)', color='black', linewidth=2.2)
            plt.plot(t_steps, traj["clean"][:120], label='Clean Baseline Forecast', color='#1f77b4', linestyle='--', linewidth=1.6)
            plt.plot(t_steps, traj["adv"][:120], label='Attacked (ε=0.05, Undefended)', color='#d62728', alpha=0.8, linewidth=1.5)
            plt.plot(t_steps, traj["ratelimit"][:120], label='Defended: Physical Rate Limiter', color='#ff7f0e', linestyle=':', linewidth=1.8)
            plt.plot(t_steps, traj["adv_trained"][:120], label='Defended: Adversarial Training', color='#2ca02c', linewidth=1.8)
            plt.plot(t_steps, traj["defense_in_depth"][:120], label='Defended: Defense-in-Depth (Hybrid)', color='#9467bd', linewidth=1.8)

            plt.title(f"{m_name}: Complete Multi-Scenario Forecast Trajectory (Clean vs Attacked vs Defenses @ ε=0.05)", fontsize=12, fontweight='bold')
            plt.xlabel("Time Step (Hours)", fontsize=11)
            plt.ylabel("Temperature (°C)", fontsize=11)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
            plt.tight_layout()
            plt.savefig(f"results/plots/comparative/trajectory_comparison_{m_name.lower()}.png")
            plt.close()

    # Plot 2: MSE Scaling Across Epsilon (All Defenses Compared)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
    for idx, m_name in enumerate(["LSTM", "RNN"]):
        ax = axes[idx]
        for s_name, color, style in [
            ("Attacked (No Defense)", "#d62728", "-"),
            ("Defense: Rolling Median", "#ff7f0e", "--"),
            ("Defense: Physical Rate Limiter", "#8c564b", ":"),
            ("Defense: Adversarial Training", "#2ca02c", "-"),
            ("Defense: Defense-in-Depth (Hybrid)", "#9467bd", "-.")
        ]:
            sub = results_df[(results_df["Model"] == m_name) & (results_df["Scenario"] == s_name)]
            ax.plot(sub["Epsilon"], sub["MSE"], marker='o', label=s_name, color=color, linestyle=style, linewidth=1.8)

        ax.set_title(f"{m_name}: Forecasting MSE Across Attack Budgets (ε)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Perturbation Budget (ε)", fontsize=10)
        ax.set_ylabel("Mean Squared Error (MSE)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig("results/plots/comparative/mse_across_all_defenses.png")
    plt.close()

    # Plot 3: R2 Retention Across All Scenarios
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
    for idx, m_name in enumerate(["LSTM", "RNN"]):
        ax = axes[idx]
        for s_name, color, style in [
            ("Attacked (No Defense)", "#d62728", "-"),
            ("Defense: Rolling Median", "#ff7f0e", "--"),
            ("Defense: Physical Rate Limiter", "#8c564b", ":"),
            ("Defense: Adversarial Training", "#2ca02c", "-"),
            ("Defense: Defense-in-Depth (Hybrid)", "#9467bd", "-.")
        ]:
            sub = results_df[(results_df["Model"] == m_name) & (results_df["Scenario"] == s_name)]
            ax.plot(sub["Epsilon"], sub["R2"], marker='s', label=s_name, color=color, linestyle=style, linewidth=1.8)

        ax.set_title(f"{m_name}: R² Explained Variance Retention", fontsize=11, fontweight='bold')
        ax.set_xlabel("Perturbation Budget (ε)", fontsize=10)
        ax.set_ylabel("R² Score", fontsize=10)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig("results/plots/comparative/r2_retention_all_defenses.png")
    plt.close()

    # Plot 4: Defense Recovery Percentage Comparison Bar Chart
    plt.figure(figsize=(12, 5), dpi=150)
    eval_eps = [0.01, 0.05, 0.10]
    sub_bar = results_df[(results_df["Epsilon"].isin(eval_eps)) & (results_df["Scenario"] != "Attacked (No Defense)")]

    palette = {"Defense: Rolling Median": "#ff7f0e",
               "Defense: Physical Rate Limiter": "#8c564b",
               "Defense: Adversarial Training": "#2ca02c",
               "Defense: Defense-in-Depth (Hybrid)": "#9467bd"}

    sns.barplot(
        data=sub_bar,
        x="Epsilon",
        y="Defense_Recovery_Pct",
        hue="Scenario",
        palette=palette,
        errorbar=None
    )
    plt.title("Defensive Recovery Percentage Across Key Threat Budgets (ε)", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Budget (ε)", fontsize=11)
    plt.ylabel("Error Recovery Rate (%)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5, axis='y')
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/plots/comparative/defense_recovery_comparison_barchart.png")
    plt.close()

    print("Saved all 4 comparative plot artifacts to: results/plots/comparative/")

    # 7. Verification of the Interactive Cyber-Aware Prediction Pipeline
    print("\n[Step 6/6] Verifying integrated cyber-aware forecasting pipeline...")
    sample_res = predict_temperature_by_datetime(
        df_clean=df_clean,
        lstm_model=lstm_robust,
        rnn_model=rnn_robust,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        year=2009,
        month=11,
        day=24,
        hour=12,
        epsilon=0.02,
        verbose=True
    )
    print("\nMaster Cybersecurity Research Benchmark successfully completed!")


if __name__ == "__main__":
    main()

