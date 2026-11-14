"""
Cybersecurity Defense: Attack Detection for Multivariate Time-Series Forecasting.

Cybersecurity Context:
----------------------
In industrial IoT, weather telemetry, or smart infrastructure, an intrusion detection 
system (IDS) must flag adversarial evasion attacks before corrupted predictions propagate.

This module implements two complementary detection mechanisms:
1. Statistical Rate-of-Change Detector:
   Examines consecutive hourly differences across physical sensor telemetry.
   Attacks that inject high-frequency sign perturbations violate atmospheric continuity bounds.
2. Prediction Residual Anomaly Detector:
   Examines forecast discrepancy / residual error relative to clean historical distributions.
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
from src.preprocessing import SENSOR_FEATURE_INDICES


class StatisticalRateOfChangeDetector:
    """
    Detects adversarial perturbations by analyzing consecutive time-step deltas
    and high-frequency variations in multivariate sensor sequences.
    """

    def __init__(self, percentile=99.0, custom_threshold=None):
        """
        Parameters:
        -----------
        percentile : float
            Percentile of clean training/validation scores used to set the detection threshold.
        custom_threshold : float or None
            If provided, overrides percentile calibration with an explicit threshold.
        """
        self.percentile = percentile
        self.custom_threshold = custom_threshold
        self.sensor_indices = SENSOR_FEATURE_INDICES
        self.threshold = None
        self.is_calibrated = False

    def compute_anomaly_scores(self, X):
        """
        Computes rate-of-change anomaly score for sequences of shape (N, 24, D).
        Uses maximum absolute consecutive step jump across physical sensor channels.
        """
        # Focus on physical sensor channels: shape (N, 24, num_sensors)
        X_sensors = X[:, :, self.sensor_indices]

        # First-order difference: (N, 23, num_sensors)
        deltas = np.diff(X_sensors, axis=1)

        # Max absolute delta across time and sensor features: (N,)
        max_delta = np.max(np.abs(deltas), axis=(1, 2))

        # Second-order difference (acceleration / roughness): (N, 22, num_sensors)
        diff2 = np.diff(deltas, axis=1)
        mean_roughness = np.mean(np.abs(diff2), axis=(1, 2))

        # Composite score combining maximum step jump and local roughness
        scores = max_delta + mean_roughness
        return scores

    def fit(self, X_clean):
        """
        Calibrates the detection threshold on clean training/validation sequences.
        """
        scores = self.compute_anomaly_scores(X_clean)
        if self.custom_threshold is not None:
            self.threshold = float(self.custom_threshold)
        else:
            self.threshold = float(np.percentile(scores, self.percentile))
        self.is_calibrated = True
        return self.threshold

    def predict(self, X):
        """
        Flags sequences as suspicious: 0 = Clean, 1 = Suspicious / Attacked.
        """
        if not self.is_calibrated:
            raise RuntimeError("Detector must be calibrated with fit() before calling predict().")
        scores = self.compute_anomaly_scores(X)
        return (scores > self.threshold).astype(int)


class ResidualAnomalyDetector:
    """
    Detects adversarial attacks by analyzing forecasting prediction residuals.
    Under adversarial perturbations, prediction residuals deviate significantly 
    from the clean validation baseline.
    """

    def __init__(self, percentile=98.0, custom_threshold=None):
        """
        Parameters:
        -----------
        percentile : float
            Percentile of clean validation residuals used to establish threshold.
        custom_threshold : float or None
            If provided, overrides percentile calibration with explicit threshold.
        """
        self.percentile = percentile
        self.custom_threshold = custom_threshold
        self.threshold = None
        self.is_calibrated = False

    def compute_residuals(self, model, X, y_true=None):
        """
        Calculates prediction residuals.
        If y_true is provided (supervised evaluation): residual = |y_pred - y_true|.
        If y_true is None (unsupervised streaming): residual = |y_pred - last_temperature_step|.
        """
        y_pred = model.predict(X, batch_size=256, verbose=0)
        
        if y_true is not None:
            # Supervised prediction residual
            residuals = np.abs(y_pred.flatten() - y_true.flatten())
        else:
            # Temporal persistence discrepancy: compare forecast against the sequence's last temperature reading
            # Note: in normalized space, temperature target corresponds to target_scaler
            last_reading = X[:, -1, 0]  # First feature is Apparent Temperature
            residuals = np.abs(y_pred.flatten() - last_reading)

        return residuals

    def fit(self, model, X_clean, y_clean):
        """
        Calibrates residual threshold on clean validation data.
        """
        residuals = self.compute_residuals(model, X_clean, y_clean)
        if self.custom_threshold is not None:
            self.threshold = float(self.custom_threshold)
        else:
            self.threshold = float(np.percentile(residuals, self.percentile))
        self.is_calibrated = True
        return self.threshold

    def predict(self, model, X, y_true):
        """
        Flags sequences as suspicious: 0 = Clean, 1 = Suspicious / Attacked.
        """
        if not self.is_calibrated:
            raise RuntimeError("Residual detector must be calibrated with fit() before predict().")
        residuals = self.compute_residuals(model, X, y_true)
        return (residuals > self.threshold).astype(int)


def evaluate_detector(y_true_binary, y_pred_binary):
    """
    Calculates cybersecurity intrusion detection metrics:
    TP, TN, FP, FN, Precision, Recall, F1, FPR, TPR.
    """
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": float(precision),
        "Recall": float(recall),
        "TPR": float(tpr),
        "FPR": float(fpr),
        "F1_Score": float(f1)
    }
