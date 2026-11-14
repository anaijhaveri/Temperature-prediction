"""
Cybersecurity Research Experiment: Attack Detection Layer for Time-Series Forecasting.

Course Context: AI for Cyber Security
------------------------------------
This script evaluates two complementary intrusion detection approaches:
1. Statistical Rate-of-Change Detection (analyzing physical hourly continuity).
2. Prediction Residual Anomaly Detection (analyzing forecasting residual deviations).

The experiment benchmarks both detectors on:
- Clean baseline test sequences
- Adversarially attacked test sequences across epsilon in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
- Evaluated separately for LSTM and SimpleRNN models.

Zero model re-training is performed. All pre-generated attack datasets are preserved.
"""

import os
import sys

# Ensure repo root is on sys.path and preloads MSVC runtime DLLs BEFORE any other C-extensions
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Preload MSVC runtime before numpy, scipy, matplotlib or tensorflow
import src

import random
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
from src.defenses.detection import (
    StatisticalRateOfChangeDetector,
    ResidualAnomalyDetector,
    evaluate_detector
)
from src.inference import predict_temperature_by_datetime


def main():
    print("=" * 80)
    print("CYBERSECURITY EXPERIMENT: ADVERSARIAL ATTACK DETECTION")
    print("=" * 80)

    # 1. Load Data and Pre-Trained Models
    print("\n[Step 1/5] Loading datasets and pre-trained models...")
    df_clean = load_and_clean_data("weatherHistory.xls")
    dataset_dict = prepare_datasets(df_clean, sequence_length=24, leakage_free=False, scaler_dir="saved_models")

    X_val, y_val = dataset_dict["X_val"], dataset_dict["y_val"]
    feature_scaler = dataset_dict["feature_scaler"]
    target_scaler = dataset_dict["target_scaler"]

    # Load preserved clean test dataset
    X_test_clean = np.load("data/clean/X_test_clean.npy")
    y_test_clean = np.load("data/clean/y_test_clean.npy")
    print(f"Loaded clean test data: {X_test_clean.shape} samples.")

    # Load preserved models
    lstm_model = load_trained_model("saved_models/lstm_model.keras")
    rnn_model = load_trained_model("saved_models/rnn_model.keras")

    # 2. Calibrate Detectors on Clean Validation Data
    print("\n[Step 2/5] Calibrating detection thresholds on clean validation sequences...")
    # Calibrate Statistical Rate-of-Change Detector at 99.0th percentile of clean validation sequences
    rate_detector = StatisticalRateOfChangeDetector(percentile=99.0)
    thresh_rate = rate_detector.fit(X_val)
    print(f"  [Rate-of-Change Detector] Calibrated 99.0th percentile threshold: {thresh_rate:.4f}")

    # Calibrate Residual Anomaly Detector at 98.0th percentile for LSTM and RNN
    residual_detector_lstm = ResidualAnomalyDetector(percentile=98.0)
    thresh_res_lstm = residual_detector_lstm.fit(lstm_model, X_val, y_val)
    print(f"  [Residual Detector - LSTM] Calibrated 98.0th percentile threshold: {thresh_res_lstm:.4f}")

    residual_detector_rnn = ResidualAnomalyDetector(percentile=98.0)
    thresh_res_rnn = residual_detector_rnn.fit(rnn_model, X_val, y_val)
    print(f"  [Residual Detector - RNN]  Calibrated 98.0th percentile threshold: {thresh_res_rnn:.4f}")

    # Verify False Positive Rate on Clean Test Data
    print("\n--- Baseline False Positive Rate on Clean Test Data ---")
    clean_flags_rate = rate_detector.predict(X_test_clean)
    fpr_clean_rate = np.mean(clean_flags_rate) * 100.0

    clean_flags_res_lstm = residual_detector_lstm.predict(lstm_model, X_test_clean, y_test_clean)
    fpr_clean_res_lstm = np.mean(clean_flags_res_lstm) * 100.0

    clean_flags_res_rnn = residual_detector_rnn.predict(rnn_model, X_test_clean, y_test_clean)
    fpr_clean_res_rnn = np.mean(clean_flags_res_rnn) * 100.0

    print(f"Rate-of-Change Clean False Positive Rate: {fpr_clean_rate:.2f}% (Expected ~1.0%)")
    print(f"LSTM Residual Clean False Positive Rate:   {fpr_clean_res_lstm:.2f}% (Expected ~2.0%)")
    print(f"RNN Residual Clean False Positive Rate:    {fpr_clean_res_rnn:.2f}% (Expected ~2.0%)")

    # 3. Evaluate Detection Across All Epsilon Values
    print("\n[Step 3/5] Evaluating detection across attack strengths on preserved attacked data...")
    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    results_records = []
    confusion_matrices = {}
    score_samples = {}

    for model_name, model_obj, res_detector in [
        ("LSTM", lstm_model, residual_detector_lstm),
        ("RNN", rnn_model, residual_detector_rnn)
    ]:
        print(f"\n>>> Running Detection Evaluation on {model_name} Model <<<")
        for eps in epsilons:
            adv_path = f"data/attacked/X_test_adv_{model_name.lower()}_eps_{eps}.npy"
            if not os.path.exists(adv_path):
                print(f"Warning: Attacked file {adv_path} not found. Skipping.")
                continue

            X_test_adv = np.load(adv_path)

            # Balanced benchmark: 50% clean (label 0), 50% attacked (label 1)
            # Use matching number of samples for rigorous evaluation
            n_samples = len(X_test_clean)
            X_eval = np.concatenate([X_test_clean, X_test_adv], axis=0)
            y_eval_true = np.concatenate([np.zeros(n_samples, dtype=int), np.ones(n_samples, dtype=int)], axis=0)
            y_eval_targets = np.concatenate([y_test_clean, y_test_clean], axis=0)

            # 1. Rate-of-Change Detector Evaluation
            y_pred_rate = rate_detector.predict(X_eval)
            metrics_rate = evaluate_detector(y_eval_true, y_pred_rate)

            # 2. Residual Anomaly Detector Evaluation
            y_pred_res = res_detector.predict(model_obj, X_eval, y_eval_targets)
            metrics_res = evaluate_detector(y_eval_true, y_pred_res)

            # Record results
            results_records.append({
                "Model": model_name,
                "Detector": "Rate_of_Change",
                "Epsilon": eps,
                "Threshold": thresh_rate,
                "TP": metrics_rate["TP"],
                "TN": metrics_rate["TN"],
                "FP": metrics_rate["FP"],
                "FN": metrics_rate["FN"],
                "Precision": metrics_rate["Precision"],
                "Recall_TPR": metrics_rate["Recall"],
                "FPR": metrics_rate["FPR"],
                "F1_Score": metrics_rate["F1_Score"]
            })

            results_records.append({
                "Model": model_name,
                "Detector": "Residual_Anomaly",
                "Epsilon": eps,
                "Threshold": res_detector.threshold,
                "TP": metrics_res["TP"],
                "TN": metrics_res["TN"],
                "FP": metrics_res["FP"],
                "FN": metrics_res["FN"],
                "Precision": metrics_res["Precision"],
                "Recall_TPR": metrics_res["Recall"],
                "FPR": metrics_res["FPR"],
                "F1_Score": metrics_res["F1_Score"]
            })

            # Save confusion matrices at eps=0.01 and eps=0.05
            if eps in [0.01, 0.05]:
                confusion_matrices[(model_name, "Rate_of_Change", eps)] = metrics_rate
                confusion_matrices[(model_name, "Residual_Anomaly", eps)] = metrics_res

            # Cache score samples for distribution visualization
            if eps in [0.01, 0.05]:
                score_samples[(model_name, "Rate_of_Change", eps)] = (
                    rate_detector.compute_anomaly_scores(X_test_clean[:2000]),
                    rate_detector.compute_anomaly_scores(X_test_adv[:2000])
                )
                score_samples[(model_name, "Residual_Anomaly", eps)] = (
                    res_detector.compute_residuals(model_obj, X_test_clean[:2000], y_test_clean[:2000]),
                    res_detector.compute_residuals(model_obj, X_test_adv[:2000], y_test_clean[:2000])
                )

            print(f"  [ε = {eps:0.3f}] Rate-of-Change -> TPR: {metrics_rate['Recall']*100:5.1f}% | FPR: {metrics_rate['FPR']*100:4.1f}% | F1: {metrics_rate['F1_Score']:.4f}")
            print(f"             Residual-Anomaly -> TPR: {metrics_res['Recall']*100:5.1f}% | FPR: {metrics_res['FPR']*100:4.1f}% | F1: {metrics_res['F1_Score']:.4f}")

    results_df = pd.DataFrame(results_records)
    csv_path = "results/attack_detection_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nAll detection results saved to: {csv_path}")

    # 4. Generate Research Plots
    print("\n[Step 4/5] Generating detection visualization plots...")
    os.makedirs("results/plots/detection", exist_ok=True)

    # Plot 1A: Confusion Matrices for LSTM at epsilon=0.01 and 0.05
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    for idx, (d_name, eps, ax) in enumerate([
        ("Residual_Anomaly", 0.01, axes[0]),
        ("Residual_Anomaly", 0.05, axes[1]),
    ]):
        cm_data = confusion_matrices[("LSTM", d_name, eps)]
        cm_matrix = np.array([[cm_data["TN"], cm_data["FP"]],
                              [cm_data["FN"], cm_data["TP"]]])
        sns.heatmap(
            cm_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=["Pred Clean", "Pred Attack"],
            yticklabels=["Actual Clean", "Actual Attack"]
        )
        ax.set_title(f"LSTM: {d_name.replace('_', ' ')} (ε = {eps})\nF1: {cm_data['F1_Score']:.3f} | TPR: {cm_data['Recall']*100:.1f}% | FPR: {cm_data['FPR']*100:.1f}%", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/plots/detection/confusion_matrices_lstm.png")
    plt.close()

    # Plot 1B: Confusion Matrices for RNN at epsilon=0.01 and 0.05
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    for idx, (d_name, eps, ax) in enumerate([
        ("Residual_Anomaly", 0.01, axes[0]),
        ("Residual_Anomaly", 0.05, axes[1]),
    ]):
        cm_data = confusion_matrices[("RNN", d_name, eps)]
        cm_matrix = np.array([[cm_data["TN"], cm_data["FP"]],
                              [cm_data["FN"], cm_data["TP"]]])
        sns.heatmap(
            cm_matrix, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax,
            xticklabels=["Pred Clean", "Pred Attack"],
            yticklabels=["Actual Clean", "Actual Attack"]
        )
        ax.set_title(f"RNN: {d_name.replace('_', ' ')} (ε = {eps})\nF1: {cm_data['F1_Score']:.3f} | TPR: {cm_data['Recall']*100:.1f}% | FPR: {cm_data['FPR']*100:.1f}%", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/plots/detection/confusion_matrices_rnn.png")
    plt.close()

    # Plot 2: Clean vs Attacked Detection Rate (Bar Chart for both LSTM & RNN)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for idx, (m_name, fpr_res, ax) in enumerate([("LSTM", fpr_clean_res_lstm, axes[0]), ("RNN", fpr_clean_res_rnn, axes[1])]):
        m_rate = results_df[(results_df["Model"] == m_name) & (results_df["Detector"] == "Rate_of_Change")]
        m_res = results_df[(results_df["Model"] == m_name) & (results_df["Detector"] == "Residual_Anomaly")]

        x_labels = ["Clean\n(FPR)"] + [f"ε={e}" for e in epsilons]
        rates_roc = [fpr_clean_rate] + [r * 100 for r in m_rate["Recall_TPR"]]
        rates_res = [fpr_res] + [r * 100 for r in m_res["Recall_TPR"]]

        x = np.arange(len(x_labels))
        width = 0.35
        ax.bar(x - width/2, rates_roc, width, label='Statistical Rate-of-Change', color='#1f77b4', alpha=0.9)
        ax.bar(x + width/2, rates_res, width, label='Prediction Residual Anomaly', color='#d62728', alpha=0.9)
        ax.set_title(f"{m_name}: Clean False Alarm vs. Attack Detection Rate", fontsize=11, fontweight='bold')
        ax.set_ylabel("Detection / Alert Rate (%)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 105)
        ax.axhline(5, color='gray', linestyle=':', label='5% Max False Alarm Target')
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/plots/detection/clean_vs_attacked_detection_rate.png")
    plt.close()

    # Plot 3: Detection Score Distribution (Clean vs Attacked Histograms)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    clean_res_l, adv_res_l_01 = score_samples[("LSTM", "Residual_Anomaly", 0.01)]
    _, adv_res_l_05 = score_samples[("LSTM", "Residual_Anomaly", 0.05)]
    sns.histplot(clean_res_l, color="green", label="Clean Sequences", kde=True, stat="density", bins=40, alpha=0.3, ax=axes[0])
    sns.histplot(adv_res_l_01, color="orange", label="Attacked (ε=0.01)", kde=True, stat="density", bins=40, alpha=0.3, ax=axes[0])
    sns.histplot(adv_res_l_05, color="red", label="Attacked (ε=0.05)", kde=True, stat="density", bins=40, alpha=0.3, ax=axes[0])
    axes[0].axvline(thresh_res_lstm, color="black", linestyle="--", linewidth=2.0, label=f"Threshold ({thresh_res_lstm:.3f})")
    axes[0].set_title("LSTM Prediction Residual Distribution Across Attack Strengths", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Absolute Normalized Residual |y_pred - y_true|", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # Plot 4: Threshold Visualization (Clean Density + Cutoff Area)
    clean_scores_all = rate_detector.compute_anomaly_scores(X_val)
    sns.kdeplot(clean_scores_all, color="#1f77b4", fill=True, alpha=0.3, label="Clean Validation Distribution", ax=axes[1])
    axes[1].axvline(thresh_rate, color="crimson", linestyle="--", linewidth=2.5, label=f"99.0th Percentile Cutoff ({thresh_rate:.3f})")
    axes[1].set_title("Statistical Rate-of-Change Threshold Calibration", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Rate-of-Change Anomaly Score", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/plots/detection/score_distributions_and_thresholds.png")
    plt.close()

    # Plot 5: Detection Performance (TPR, FPR, F1) vs Epsilon Curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for idx, m_name in enumerate(["LSTM", "RNN"]):
        ax = axes[idx]
        df_sub_roc = results_df[(results_df["Model"] == m_name) & (results_df["Detector"] == "Rate_of_Change")]
        df_sub_res = results_df[(results_df["Model"] == m_name) & (results_df["Detector"] == "Residual_Anomaly")]

        ax.plot(df_sub_roc["Epsilon"], df_sub_roc["Recall_TPR"] * 100, marker='o', label="Rate-of-Change (TPR)", color="#1f77b4", linewidth=2)
        ax.plot(df_sub_roc["Epsilon"], df_sub_roc["F1_Score"] * 100, marker='s', linestyle='--', label="Rate-of-Change (F1)", color="#1f77b4", linewidth=1.5)

        ax.plot(df_sub_res["Epsilon"], df_sub_res["Recall_TPR"] * 100, marker='^', label="Residual-Anomaly (TPR)", color="#d62728", linewidth=2)
        ax.plot(df_sub_res["Epsilon"], df_sub_res["F1_Score"] * 100, marker='d', linestyle='--', label="Residual-Anomaly (F1)", color="#d62728", linewidth=1.5)

        ax.set_title(f"{m_name}: Detection Metrics vs. Attack Budget (ε)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Perturbation Budget (ε)", fontsize=10)
        ax.set_ylabel("Metric Score (%)", fontsize=10)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("results/plots/detection/detection_metrics_vs_epsilon.png")
    plt.close()

    print("Saved all 5 detection plot artifacts to: results/plots/detection/")

    # 5. Verify Original Prediction Pipeline Operation
    print("\n[Step 5/5] Verifying original prediction pipeline integrity with integrated detection...")
    sample_clean = predict_temperature_by_datetime(
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
    print(f"Pipeline verification test clean forecast: {sample_clean['prediction']:.2f}°C (Ground Truth: {sample_clean['actual']:.2f}°C)")
    print("\nAttack Detection Experiment successfully executed.")


if __name__ == "__main__":
    main()
