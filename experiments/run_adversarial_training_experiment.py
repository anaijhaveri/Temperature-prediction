"""
Cybersecurity Research Experiment: Adversarial Training Defense for Time-Series Forecasting.

Course Context: AI for Cyber Security
------------------------------------
This script implements Defense Component 3: Adversarial Training.
It trains robust counterparts of the SimpleRNN and LSTM models using online 
clean-adversarial training batches (50% clean, 50% FGSM adversarial).

Strict Chronological Data Separation:
-------------------------------------
- Training data (X_train, y_train): strictly used for training clean and robust models.
- Validation data (X_val, y_val): strictly used for early-stopping and checkpointing.
- Final test data (X_test, y_test): strictly held out for the final security evaluation.

Deliverables:
-------------
- Persisted robust models: saved_models/lstm_robust.keras, saved_models/rnn_robust.keras
- Configuration saved to: results/adversarial_training_config.json
- Final comparison CSVs: results/final_adversarial_training_comparison.csv
                         results/final_security_comparison_table.csv
- 8 Academic plots saved to: results/plots/adversarial_training/
"""

import os
import sys

# Force immediate unbuffered console logging
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Ensure repo root is on sys.path and preloads MSVC runtime DLLs BEFORE any other C-extensions
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Preload MSVC runtime before numpy, scipy, matplotlib, pandas or tensorflow
import src

import json
import random
import time
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
from src.defenses.adversarial_training import train_adversarially
from src.evaluation import compute_metrics
from src.inference import predict_temperature_by_datetime


def main():
    print("=" * 80)
    print("CYBERSECURITY EXPERIMENT: ADVERSARIAL TRAINING DEFENSE (COMPONENT 3)")
    print("=" * 80)

    # 1. Dataset Loading and Strict Partitioning
    print("\n[Step 1/6] Loading data with strict chronological separation...")
    df_clean = load_and_clean_data("weatherHistory.xls")
    dataset_dict = prepare_datasets(df_clean, sequence_length=24, leakage_free=False, scaler_dir="saved_models")

    X_train, y_train = dataset_dict["X_train"], dataset_dict["y_train"]
    X_val, y_val = dataset_dict["X_val"], dataset_dict["y_val"]
    target_scaler = dataset_dict["target_scaler"]
    feature_scaler = dataset_dict["feature_scaler"]

    # Load preserved clean test dataset (UNTOUCHED during training)
    X_test_clean = np.load("data/clean/X_test_clean.npy")
    y_test_clean = np.load("data/clean/y_test_clean.npy")
    y_test_inv = target_scaler.inverse_transform(y_test_clean)

    print(f"Data partitioning verification:")
    print(f"  Training partition (strictly for model training):   {X_train.shape[0]} sequences")
    print(f"  Validation partition (strictly for early stopping): {X_val.shape[0]} sequences")
    print(f"  Final test partition (unseen evaluation holdout):   {X_test_clean.shape[0]} sequences")

    # 2. Save Experiment Configuration
    config = {
        "random_seed": RANDOM_SEED,
        "epsilon_train": 0.02,
        "adversarial_ratio": 0.50,
        "alpha_clean_loss": 0.50,
        "alpha_adversarial_loss": 0.50,
        "learning_rate": 0.001,
        "batch_size": 256,
        "max_epochs": 8,
        "early_stopping_patience": 3,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test_clean.shape[0]),
        "evaluation_epsilons": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    }
    os.makedirs("results", exist_ok=True)
    with open("results/adversarial_training_config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("Saved training configuration to: results/adversarial_training_config.json")

    # 3. Load Original Models (Preserved Baseline)
    print("\n[Step 2/6] Loading preserved original clean models (Baseline)...")
    lstm_orig = load_trained_model("saved_models/lstm_model.keras")
    rnn_orig = load_trained_model("saved_models/rnn_model.keras")

    # 4. Train Robust Models (Adversarially Trained)
    print("\n[Step 3/6] Training / Loading robust models...")
    lstm_robust_path = "saved_models/lstm_robust.keras"
    rnn_robust_path = "saved_models/rnn_robust.keras"

    if os.path.exists(lstm_robust_path):
        print("Loading pre-trained Robust LSTM model...")
        lstm_robust = load_trained_model(lstm_robust_path)
    else:
        print("\n>>> Training Robust LSTM Model (50% Clean, 50% FGSM Adversarial) <<<")
        lstm_robust = build_lstm_model(input_shape=(24, 10))
        train_adversarially(
            model=lstm_robust,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epsilon_train=config["epsilon_train"],
            alpha=config["alpha_clean_loss"],
            epochs=config["max_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            random_seed=config["random_seed"],
            patience=config["early_stopping_patience"]
        )
        save_trained_model(lstm_robust, lstm_robust_path)

    if os.path.exists(rnn_robust_path):
        print("Loading pre-trained Robust RNN model...")
        rnn_robust = load_trained_model(rnn_robust_path)
    else:
        print("\n>>> Training Robust SimpleRNN Model (50% Clean, 50% FGSM Adversarial) <<<")
        rnn_robust = build_rnn_model(input_shape=(24, 10))
        train_adversarially(
            model=rnn_robust,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epsilon_train=config["epsilon_train"],
            alpha=config["alpha_clean_loss"],
            epochs=config["max_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            random_seed=config["random_seed"],
            patience=config["early_stopping_patience"]
        )
        save_trained_model(rnn_robust, rnn_robust_path)

    # 5. Comprehensive Security Comparison on Test Data
    print("\n[Step 4/6] Running final security evaluation across all epsilon budgets...")
    epsilons = config["evaluation_epsilons"]
    records = []
    trajectories = {}

    models_eval = [
        ("LSTM", lstm_orig, lstm_robust),
        ("RNN", rnn_orig, rnn_robust)
    ]

    for model_name, m_orig, m_robust in models_eval:
        print(f"\n--- Final Security Evaluation for {model_name} ---")

        # Evaluate clean baseline on original and robust models
        clean_orig_pred = target_scaler.inverse_transform(m_orig.predict(X_test_clean, batch_size=256, verbose=0))
        clean_rob_pred = target_scaler.inverse_transform(m_robust.predict(X_test_clean, batch_size=256, verbose=0))

        met_clean_orig = compute_metrics(y_test_inv, clean_orig_pred)
        met_clean_rob = compute_metrics(y_test_inv, clean_rob_pred)

        clean_perf_delta = met_clean_rob["MSE"] - met_clean_orig["MSE"]
        clean_perf_pct = (clean_perf_delta / met_clean_orig["MSE"]) * 100.0

        print(f"Clean Performance Trade-off:")
        print(f"  Original {model_name} Clean MSE: {met_clean_orig['MSE']:.4f} (R² = {met_clean_orig['R2']:.4f})")
        print(f"  Robust   {model_name} Clean MSE: {met_clean_rob['MSE']:.4f} (R² = {met_clean_rob['R2']:.4f})")
        print(f"  Clean Accuracy Trade-off Δ:      {clean_perf_delta:+.4f} ({clean_perf_pct:+.2f}%)")

        for eps in epsilons:
            if eps == 0.0:
                X_adv = np.copy(X_test_clean)
            else:
                adv_path = f"data/attacked/X_test_adv_{model_name.lower()}_eps_{eps}.npy"
                X_adv = np.load(adv_path)

            # Predict with Original Model
            pred_orig = target_scaler.inverse_transform(m_orig.predict(X_adv, batch_size=256, verbose=0))
            m_orig_curr = compute_metrics(y_test_inv, pred_orig)

            # Predict with Robust Model
            pred_rob = target_scaler.inverse_transform(m_robust.predict(X_adv, batch_size=256, verbose=0))
            m_rob_curr = compute_metrics(y_test_inv, pred_rob)

            if eps == 0.05:
                trajectories[model_name] = {
                    "true": y_test_inv,
                    "clean_orig": clean_orig_pred,
                    "clean_rob": clean_rob_pred,
                    "att_orig": pred_orig,
                    "att_rob": pred_rob
                }

            # Calculations
            attack_degradation = m_orig_curr["MSE"] - met_clean_orig["MSE"]
            robustness_improvement = m_orig_curr["MSE"] - m_rob_curr["MSE"]

            recovery_pct = (robustness_improvement / attack_degradation * 100.0) if attack_degradation > 0 else 0.0
            rel_rmse_imp = ((m_orig_curr["RMSE"] - m_rob_curr["RMSE"]) / m_orig_curr["RMSE"]) * 100.0
            rel_mae_imp = ((m_orig_curr["MAE"] - m_rob_curr["MAE"]) / m_orig_curr["MAE"]) * 100.0

            # Record Original Model
            records.append({
                "Model": model_name,
                "Model_Type": f"Original {model_name}",
                "Condition": "Clean" if eps == 0.0 else "Attacked",
                "Epsilon": eps,
                "MAE": m_orig_curr["MAE"],
                "MSE": m_orig_curr["MSE"],
                "RMSE": m_orig_curr["RMSE"],
                "R2": m_orig_curr["R2"],
                "Attack_Degradation": attack_degradation,
                "Robustness_Improvement": 0.0,
                "Defense_Recovery_Pct": 0.0,
                "Rel_RMSE_Improvement_Pct": 0.0,
                "Rel_MAE_Improvement_Pct": 0.0
            })

            # Record Robust Model
            records.append({
                "Model": model_name,
                "Model_Type": f"Robust {model_name}",
                "Condition": "Clean" if eps == 0.0 else "Attacked (Adv Trained)",
                "Epsilon": eps,
                "MAE": m_rob_curr["MAE"],
                "MSE": m_rob_curr["MSE"],
                "RMSE": m_rob_curr["RMSE"],
                "R2": m_rob_curr["R2"],
                "Attack_Degradation": m_rob_curr["MSE"] - met_clean_rob["MSE"],
                "Robustness_Improvement": robustness_improvement,
                "Defense_Recovery_Pct": recovery_pct,
                "Rel_RMSE_Improvement_Pct": rel_rmse_imp,
                "Rel_MAE_Improvement_Pct": rel_mae_imp
            })

            print(f"  [ε = {eps:0.3f}] Orig MSE: {m_orig_curr['MSE']:6.2f} | "
                  f"Robust MSE: {m_rob_curr['MSE']:6.2f} | "
                  f"Improvement: {robustness_improvement:6.2f} (Rec: {recovery_pct:5.1f}%, RMSE Imp: {rel_rmse_imp:4.1f}%)")

    results_df = pd.DataFrame(records)
    csv_path = "results/final_adversarial_training_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nFinal comparison dataset saved to: {csv_path}")

    # Create Clean Summary Table
    table_records = []
    for m_type in ["Original RNN", "Robust RNN", "Original LSTM", "Robust LSTM"]:
        sub = results_df[results_df["Model_Type"] == m_type]
        for _, row in sub.iterrows():
            table_records.append({
                "Model": row["Model_Type"],
                "Condition": row["Condition"],
                "Epsilon": row["Epsilon"],
                "MAE": round(row["MAE"], 4),
                "MSE": round(row["MSE"], 4),
                "RMSE": round(row["RMSE"], 4),
                "R2": round(row["R2"], 4)
            })
    table_df = pd.DataFrame(table_records)
    table_csv = "results/final_security_comparison_table.csv"
    table_df.to_csv(table_csv, index=False)
    print(f"Final security comparison table saved to: {table_csv}")

    # 6. Generate 8 Academic Visualization Plots
    print("\n[Step 5/6] Generating 8 academic-quality visualization plots...")
    os.makedirs("results/plots/adversarial_training", exist_ok=True)

    # Plot 1: Clean vs Attacked Performance (Bar Chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for idx, m_name in enumerate(["LSTM", "RNN"]):
        ax = axes[idx]
        df_orig = results_df[(results_df["Model"] == m_name) & (results_df["Model_Type"].str.startswith("Original"))]
        df_rob = results_df[(results_df["Model"] == m_name) & (results_df["Model_Type"].str.startswith("Robust"))]

        x = np.arange(len(epsilons))
        w = 0.35
        ax.bar(x - w/2, df_orig["MSE"], w, label=f"Original {m_name}", color="#d62728", alpha=0.85)
        ax.bar(x + w/2, df_rob["MSE"], w, label=f"Robust {m_name} (Adv Trained)", color="#2ca02c", alpha=0.85)

        ax.set_title(f"{m_name}: Clean vs. Attacked Forecasting MSE", fontsize=11, fontweight='bold')
        ax.set_xlabel("Perturbation Budget (ε)", fontsize=10)
        ax.set_ylabel("Mean Squared Error (MSE)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"ε={e}" for e in epsilons])
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/plots/adversarial_training/1_clean_vs_attacked_performance.png")
    plt.close()

    # Plot 2: Original vs Adversarially Trained Trajectory (Time-Series restoration)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=150)
    for idx, m_name in enumerate(["LSTM", "RNN"]):
        ax = axes[idx]
        traj = trajectories[m_name]
        t = np.arange(120)
        ax.plot(t, traj["true"][:120], label='Actual Temperature (Ground Truth)', color='black', linewidth=2.0)
        ax.plot(t, traj["clean_orig"][:120], label=f'Original {m_name} (Clean)', color='#1f77b4', linestyle='--', linewidth=1.5)
        ax.plot(t, traj["att_orig"][:120], label=f'Original {m_name} (Attacked ε=0.05)', color='#d62728', alpha=0.8, linewidth=1.5)
        ax.plot(t, traj["att_rob"][:120], label=f'Robust {m_name} (Adv Trained ε=0.05)', color='#2ca02c', linewidth=1.8)

        ax.set_title(f"{m_name}: Forecast Trajectory Under Attack vs. Adversarially Trained Robustness (ε = 0.05)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time Step (Hours)", fontsize=10)
        ax.set_ylabel("Temperature (°C)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper left', frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/plots/adversarial_training/2_original_vs_robust_trajectory.png")
    plt.close()

    # Plot 3: RMSE vs Epsilon Curves
    plt.figure(figsize=(9, 5), dpi=150)
    for m_name in ["LSTM", "RNN"]:
        df_orig = results_df[(results_df["Model"] == m_name) & (results_df["Model_Type"].str.startswith("Original"))]
        df_rob = results_df[(results_df["Model"] == m_name) & (results_df["Model_Type"].str.startswith("Robust"))]

        plt.plot(df_orig["Epsilon"], df_orig["RMSE"], marker='o', label=f"Original {m_name}", linewidth=1.8)
        plt.plot(df_rob["Epsilon"], df_rob["RMSE"], marker='s', linestyle='--', label=f"Robust {m_name}", linewidth=1.8)

    plt.title("Root Mean Squared Error (RMSE) vs. Perturbation Budget (ε)", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Budget (ε)", fontsize=11)
    plt.ylabel("RMSE (°C)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("results/plots/adversarial_training/3_rmse_vs_epsilon.png")
    plt.close()

    # Plot 4: MAE vs Epsilon Curves
    plt.figure(figsize=(9, 5), dpi=150)
    for m_name in ["LSTM", "RNN"]:
        df_orig = results_df[(results_df["Model"] == m_name) & (results_df["Model_Type"].str.startswith("Original"))]
        df_rob = results_df[(results_df["Model"] == m_name) & (results_df["Model_Type"].str.startswith("Robust"))]

        plt.plot(df_orig["Epsilon"], df_orig["MAE"], marker='^', label=f"Original {m_name}", linewidth=1.8)
        plt.plot(df_rob["Epsilon"], df_rob["MAE"], marker='v', linestyle='--', label=f"Robust {m_name}", linewidth=1.8)

    plt.title("Mean Absolute Error (MAE) vs. Perturbation Budget (ε)", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Budget (ε)", fontsize=11)
    plt.ylabel("MAE (°C)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("results/plots/adversarial_training/4_mae_vs_epsilon.png")
    plt.close()

    # Plot 5: R2 vs Epsilon Curves
    plt.figure(figsize=(9, 5), dpi=150)
    for m_name in ["LSTM", "RNN"]:
        df_orig = results_df[(results_df["Model"] == m_name) & (results_df["Model_Type"].str.startswith("Original"))]
        df_rob = results_df[(results_df["Model"] == m_name) & (results_df["Model_Type"].str.startswith("Robust"))]

        plt.plot(df_orig["Epsilon"], df_orig["R2"], marker='d', label=f"Original {m_name}", linewidth=1.8)
        plt.plot(df_rob["Epsilon"], df_rob["R2"], marker='x', linestyle='--', label=f"Robust {m_name}", linewidth=1.8)

    plt.title("Explained Variance (R² Score) Retention vs. Perturbation Budget (ε)", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Budget (ε)", fontsize=11)
    plt.ylabel("R² Score", fontsize=11)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("results/plots/adversarial_training/5_r2_vs_epsilon.png")
    plt.close()

    # Plot 6: Defense Recovery Percentage vs Epsilon
    plt.figure(figsize=(9, 5), dpi=150)
    for m_name in ["LSTM", "RNN"]:
        df_rob = results_df[(results_df["Model"] == m_name) & (results_df["Model_Type"].str.startswith("Robust")) & (results_df["Epsilon"] > 0)]
        plt.plot(df_rob["Epsilon"], df_rob["Defense_Recovery_Pct"], marker='o', linewidth=2.0, label=f"Robust {m_name} Recovery %")

    plt.title("Adversarial Training Defense Recovery Percentage vs. Budget (ε)", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Budget (ε)", fontsize=11)
    plt.ylabel("Error Recovery Rate (%)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("results/plots/adversarial_training/6_defense_recovery_vs_epsilon.png")
    plt.close()

    # Plot 7: Clean Accuracy Comparison (Trade-off Analysis)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    clean_sub = results_df[results_df["Epsilon"] == 0.0]
    bars = ax.bar(clean_sub["Model_Type"], clean_sub["MSE"], color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"], width=0.5, alpha=0.85)
    ax.set_title("Clean-Data Forecast Accuracy Comparison (Trade-Off Analysis)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Clean Test MSE", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/plots/adversarial_training/7_clean_accuracy_comparison.png")
    plt.close()

    # Plot 8: Robustness Improvement Comparison (MSE reduction at key threat levels)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    key_eps = [0.01, 0.02, 0.05, 0.10]
    sub_imp = results_df[(results_df["Epsilon"].isin(key_eps)) & (results_df["Model_Type"].str.startswith("Robust"))]
    sns.barplot(data=sub_imp, x="Epsilon", y="Robustness_Improvement", hue="Model", palette={"LSTM": "#1f77b4", "RNN": "#2ca02c"}, ax=ax)
    ax.set_title("Absolute Robustness Improvement (MSE Reduction) under Attack", fontsize=12, fontweight='bold')
    ax.set_xlabel("Perturbation Budget (ε)", fontsize=11)
    ax.set_ylabel("MSE Reduction (Error Saved)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    ax.legend(title="Model Architecture")
    plt.tight_layout()
    plt.savefig("results/plots/adversarial_training/8_robustness_improvement_comparison.png")
    plt.close()

    print("Saved all 8 academic-quality plot artifacts to: results/plots/adversarial_training/")

    # 7. Verification of Interactive Pipeline
    print("\n[Step 6/6] Verifying operational forecasting pipeline with robust models...")
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
    print("\nAdversarial Training Defense Experiment successfully executed.")


if __name__ == "__main__":
    main()
