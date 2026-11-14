"""
Evaluation metrics, attack impact analysis, and publication-quality visualization tools.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred):
    """
    Computes MAE, MSE, RMSE, and R2 score.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def compute_perturbation_stats(perturbation):
    """
    Calculates L_infinity, L_1, and L_2 norms of adversarial perturbation.
    """
    l_inf = float(np.max(np.abs(perturbation)))
    l_1_mean = float(np.mean(np.abs(perturbation)))
    l_2_mean = float(np.mean(np.sqrt(np.sum(np.square(perturbation), axis=(1, 2)))))

    return {
        "L_inf": l_inf,
        "L_1_mean": l_1_mean,
        "L_2_mean": l_2_mean
    }


def compute_attack_impact(clean_metrics, adv_metrics):
    """
    Calculates absolute and percentage degradation in metrics due to adversarial perturbation.
    """
    delta_mse = adv_metrics["MSE"] - clean_metrics["MSE"]
    pct_mse = (delta_mse / clean_metrics["MSE"]) * 100.0 if clean_metrics["MSE"] != 0 else 0.0

    delta_mae = adv_metrics["MAE"] - clean_metrics["MAE"]
    pct_mae = (delta_mae / clean_metrics["MAE"]) * 100.0 if clean_metrics["MAE"] != 0 else 0.0

    delta_rmse = adv_metrics["RMSE"] - clean_metrics["RMSE"]
    pct_rmse = (delta_rmse / clean_metrics["RMSE"]) * 100.0 if clean_metrics["RMSE"] != 0 else 0.0

    delta_r2 = clean_metrics["R2"] - adv_metrics["R2"]

    return {
        "Delta_MSE": delta_mse,
        "Pct_Inc_MSE": pct_mse,
        "Delta_MAE": delta_mae,
        "Pct_Inc_MAE": pct_mae,
        "Delta_RMSE": delta_rmse,
        "Pct_Inc_RMSE": pct_rmse,
        "R2_Degradation": delta_r2
    }


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_clean_vs_attacked_trajectory(
    y_true,
    y_clean,
    y_adv,
    epsilon,
    model_name="LSTM",
    sample_count=120,
    save_path=None
):
    """
    Plots ground truth, clean model prediction, and attacked model prediction.
    """
    plt.figure(figsize=(12, 5), dpi=150)
    time_steps = np.arange(sample_count)

    plt.plot(time_steps, y_true[:sample_count], label='Actual Temperature', color='black', linewidth=1.8)
    plt.plot(time_steps, y_clean[:sample_count], label=f'Clean {model_name} Prediction', color='#1f77b4', linestyle='--', linewidth=1.5)
    plt.plot(time_steps, y_adv[:sample_count], label=f'Attacked (ε={epsilon})', color='#d62728', alpha=0.85, linewidth=1.5)

    plt.title(f"{model_name}: Clean vs. FGSM Attacked Temperature Forecast (ε = {epsilon})", fontsize=13, fontweight='bold')
    plt.xlabel("Time Step (Hours)", fontsize=11)
    plt.ylabel("Temperature (°C)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_error_vs_epsilon(results_df, save_path=None):
    """
    Plots MAE, MSE, and RMSE against epsilon for both LSTM and RNN models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)
    metrics = ["MAE", "MSE", "RMSE"]
    colors = {"LSTM": "#1f77b4", "RNN": "#2ca02c"}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model_name in results_df["Model"].unique():
            df_m = results_df[results_df["Model"] == model_name]
            ax.plot(
                df_m["Epsilon"],
                df_m[metric],
                marker='o',
                linewidth=2.0,
                label=model_name,
                color=colors.get(model_name, "#333333")
            )
        ax.set_title(f"{metric} vs. Perturbation Budget (ε)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Perturbation Budget (ε)", fontsize=10)
        ax.set_ylabel(f"{metric} (°C)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_prediction_degradation(results_df, save_path=None):
    """
    Plots R2 degradation and percentage error amplification under increasing attack budgets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    colors = {"LSTM": "#1f77b4", "RNN": "#2ca02c"}

    # Left plot: R2 vs Epsilon
    ax1 = axes[0]
    for model_name in results_df["Model"].unique():
        df_m = results_df[results_df["Model"] == model_name]
        ax1.plot(
            df_m["Epsilon"],
            df_m["R2"],
            marker='s',
            linewidth=2.0,
            label=f"{model_name} R²",
            color=colors.get(model_name, "#333333")
        )
    ax1.set_title("Coefficient of Determination (R²) Degradation", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Perturbation Budget (ε)", fontsize=10)
    ax1.set_ylabel("R² Score", fontsize=10)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Right plot: Percentage MSE Inflation
    ax2 = axes[1]
    for model_name in results_df["Model"].unique():
        df_m = results_df[results_df["Model"] == model_name]
        ax2.plot(
            df_m["Epsilon"],
            df_m["Pct_Inc_MSE"],
            marker='^',
            linewidth=2.0,
            label=f"{model_name} % MSE Increase",
            color=colors.get(model_name, "#333333")
        )
    ax2.set_title("Forecast Error Amplification Factor (% MSE Increase)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Perturbation Budget (ε)", fontsize=10)
    ax2.set_ylabel("Error Increase (%)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_perturbation_magnitude(results_df, save_path=None):
    """
    Plots perturbation magnitude norms (L_inf and L_2) against epsilon.
    """
    plt.figure(figsize=(8, 5), dpi=150)
    for model_name in results_df["Model"].unique():
        df_m = results_df[results_df["Model"] == model_name]
        plt.plot(
            df_m["Epsilon"],
            df_m["L_2_mean"],
            marker='d',
            linewidth=1.8,
            label=f"{model_name} Mean L2 Perturbation"
        )
        plt.plot(
            df_m["Epsilon"],
            df_m["L_inf"],
            marker='x',
            linestyle='--',
            linewidth=1.5,
            label=f"{model_name} Max L_inf Perturbation"
        )

    plt.title("Adversarial Perturbation Magnitude vs. Budget (ε)", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Budget (ε)", fontsize=10)
    plt.ylabel("Perturbation Norm (Normalized Scale)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

