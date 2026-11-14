"""
Generates high-resolution system architecture diagram and baseline forecast plots
for inclusion in the final academic project report.
"""

import os
import sys

# Preload MSVC runtime
import src

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import joblib

from src.models import load_trained_model
from src.evaluation import compute_metrics

os.makedirs("results/plots", exist_ok=True)

# ------------------------------------------------------------------------------
# 1. System Architecture Diagram
# ------------------------------------------------------------------------------
def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 10), dpi=200)
    ax.axis('off')

    # Color Palette
    c_data = '#E3F2FD'     # Light blue
    c_model = '#E8F5E9'    # Light green
    c_attack = '#FFEBEE'   # Light red
    c_defense = '#FFF8E1'  # Light amber
    c_border = '#37474F'   # Dark slate

    def draw_box(x, y, w, h, title, subtitle, fill_c, border_c=c_border, lw=1.5):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=fill_c, edgecolor=border_c, linewidth=lw
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#1A237E')
        ax.text(x + w/2, y + h*0.30, subtitle, ha='center', va='center', fontsize=8.5, color='#37474F', style='italic')

    def draw_arrow(x1, y1, x2, y2, label="", color='#263238'):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", color=color, lw=2.0)
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx + 0.02, my, label, fontsize=8, color=color, fontweight='bold')

    # Section Labels
    ax.text(0.5, 0.98, "SECURE TIME-SERIES FORECASTING: END-TO-END SYSTEM ARCHITECTURE", 
            ha='center', va='top', fontsize=14, fontweight='bold', color='#0B2545')
    ax.text(0.5, 0.95, "Multivariate Weather Telemetry Pipeline with 3-Tier Cybersecurity Defense Hierarchy", 
            ha='center', va='top', fontsize=10, style='italic', color='#555555')

    # Column 1: Preprocessing & Data Flow (Left)
    draw_box(0.04, 0.78, 0.26, 0.12, "Raw Weather Dataset", "weatherHistory.xls (96,453 hourly records)\n7 atmospheric features + Date", c_data)
    draw_box(0.04, 0.58, 0.26, 0.12, "Leakage-Free Preprocessing", "Chronological Split: 70% Train, 15% Val, 15% Test\nMinMaxScaler fitted strictly on Train only", c_data)
    draw_box(0.04, 0.38, 0.26, 0.12, "Sliding Window Sequences", "Lookback Horizon T = 24 hours, Features D = 10\nTarget y = Next-Hour Temperature (°C)", c_data)

    draw_arrow(0.17, 0.78, 0.17, 0.70)
    draw_arrow(0.17, 0.58, 0.17, 0.50)

    # Column 2: Baseline Models & Inference (Center-Left)
    draw_box(0.38, 0.70, 0.26, 0.12, "Clean Baseline Models", "Stacked SimpleRNN & LSTM (128-64-32 units)\nAdam Optimizer, MSE Loss, Dropout p=0.2", c_model)
    draw_box(0.38, 0.48, 0.26, 0.12, "Baseline Forecast Output", "SimpleRNN: MSE = 7.59, R² = 0.905\nLSTM: MSE = 6.78, R² = 0.915", c_model)

    draw_arrow(0.30, 0.44, 0.34, 0.44)
    draw_arrow(0.34, 0.44, 0.34, 0.76)
    draw_arrow(0.34, 0.76, 0.38, 0.76)
    draw_arrow(0.51, 0.70, 0.51, 0.60)

    # Column 3: Adversarial Threat Modeling (Center-Right)
    draw_box(0.38, 0.24, 0.26, 0.14, "Adversarial Evasion Engine", "White-Box FGSM & BIM / PGD (ε in [0.001, 0.10])\nSensor Mask: Weather sensors corrupted\nTimestamps (Yr/Mo/Day/Hr) immutable", c_attack, border_c='#D32F2F', lw=2.0)
    draw_arrow(0.17, 0.38, 0.17, 0.31)
    draw_arrow(0.17, 0.31, 0.38, 0.31, "Telemetry Ingestion", color='#D32F2F')

    # Column 4: 3-Tier Defense Hierarchy (Right)
    # Tier 1
    draw_box(0.70, 0.76, 0.26, 0.14, "Tier 1: Attack Detection", "Rate-of-Change & Residual Anomaly Detectors\nThresholds calibrated on Clean Validation\nTPR: 83.1% to 100% at ε >= 0.05, FPR < 4.1%", c_defense, border_c='#F57C00')
    # Tier 2
    draw_box(0.70, 0.50, 0.26, 0.14, "Tier 2: Input Sanitization", "Rolling Median (w=3) & Physical Rate Limiter\nNon-parametric signal pre-filtering (No retraining)\nRecovers up to 46.2% error on SimpleRNN", c_defense, border_c='#F57C00')
    # Tier 3
    draw_box(0.70, 0.24, 0.26, 0.14, "Tier 3: Adversarial Training", "Robust Models (lstm_robust, rnn_robust)\n50% Clean + 50% Online FGSM (ε=0.02) batches\nRecovers 72.1% error on SimpleRNN at ε=0.10", c_defense, border_c='#388E3C', lw=2.0)

    # Connections into Defenses
    draw_arrow(0.64, 0.31, 0.67, 0.31)
    draw_arrow(0.67, 0.31, 0.67, 0.83)
    draw_arrow(0.67, 0.83, 0.70, 0.83, "Flag Hostile")

    draw_arrow(0.67, 0.57, 0.70, 0.57, "Filter In-Place")
    draw_arrow(0.67, 0.31, 0.70, 0.31, "Min-Max Retrain")

    # Bottom Banner: Secure Output
    rect_out = patches.FancyBboxPatch(
        (0.20, 0.04), 0.60, 0.12,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor='#ECEFF1', edgecolor='#0B2545', linewidth=2.0
    )
    ax.add_patch(rect_out)
    ax.text(0.50, 0.12, "DEFENSE-IN-DEPTH FORECASTING ENGINE", ha='center', va='center', fontsize=11, fontweight='bold', color='#0B2545')
    ax.text(0.50, 0.07, "Couples Pre-Inference Sanitization, Real-Time Residual Anomaly Monitoring,\nand Hardened Recurrent Weights for Resilient Cyber-Physical Infrastructure", 
            ha='center', va='center', fontsize=8.5, color='#263238')

    draw_arrow(0.83, 0.24, 0.83, 0.10)
    draw_arrow(0.83, 0.10, 0.80, 0.10)

    plt.tight_layout()
    plt.savefig("results/plots/system_architecture_diagram.png", dpi=300)
    plt.close()
    print("[OK] Created results/plots/system_architecture_diagram.png")

# ------------------------------------------------------------------------------
# 2. Baseline Forecast Comparison Plot
# ------------------------------------------------------------------------------
def create_baseline_plot():
    target_scaler = joblib.load("saved_models/target_scaler.joblib")
    X_test_clean = np.load("data/clean/X_test_clean.npy")
    y_test_clean = np.load("data/clean/y_test_clean.npy")
    y_test_true = target_scaler.inverse_transform(y_test_clean).flatten()

    lstm_model = load_trained_model("saved_models/lstm_model.keras")
    rnn_model = load_trained_model("saved_models/rnn_model.keras")

    y_pred_lstm = target_scaler.inverse_transform(lstm_model.predict(X_test_clean, batch_size=256, verbose=0)).flatten()
    y_pred_rnn = target_scaler.inverse_transform(rnn_model.predict(X_test_clean, batch_size=256, verbose=0)).flatten()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=200)
    hours = 120
    t = np.arange(hours)

    # Panel 1: Trajectory
    axes[0].plot(t, y_test_true[:hours], 'k-', linewidth=2.0, label='Actual Ground Truth Temperature')
    axes[0].plot(t, y_pred_lstm[:hours], color='#1f77b4', linestyle='--', linewidth=1.6, label='Clean LSTM Forecast (MSE: 6.78, R²: 0.915)')
    axes[0].plot(t, y_pred_rnn[:hours], color='#2ca02c', linestyle='-.', linewidth=1.6, label='Clean SimpleRNN Forecast (MSE: 7.59, R²: 0.905)')
    axes[0].set_title("Clean Baseline Temperature Forecast vs. Actual Temperature (First 120 Test Hours)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Temperature (°C)", fontsize=11)
    axes[0].set_xlabel("Time (Hours)", fontsize=11)
    axes[0].legend(loc='upper right', frameon=True, framealpha=0.95)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Error Residual Distribution
    res_lstm = y_test_true - y_pred_lstm
    res_rnn = y_test_true - y_pred_rnn
    sns.kdeplot(res_lstm, ax=axes[1], color='#1f77b4', label=f'LSTM Residuals (Mean: {np.mean(res_lstm):.2f}°C, Std: {np.std(res_lstm):.2f}°C)', fill=True, alpha=0.25)
    sns.kdeplot(res_rnn, ax=axes[1], color='#2ca02c', label=f'SimpleRNN Residuals (Mean: {np.mean(res_rnn):.2f}°C, Std: {np.std(res_rnn):.2f}°C)', fill=True, alpha=0.25)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1.0)
    axes[1].set_title("Forecast Residual Error Distribution (y_true - y_pred) on Clean Test Set (N = 14,465)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Residual Error (°C)", fontsize=11)
    axes[1].set_ylabel("Probability Density", fontsize=11)
    axes[1].set_xlim(-10, 10)
    axes[1].legend(loc='upper right', frameon=True, framealpha=0.95)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("results/plots/baseline_forecast_comparison.png", dpi=300)
    plt.close()
    print("[OK] Created results/plots/baseline_forecast_comparison.png")

if __name__ == "__main__":
    create_architecture_diagram()
    create_baseline_plot()

