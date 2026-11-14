"""
Cybersecurity Defense: Input Sanitization for Multivariate Time-Series Forecasting.

Cybersecurity Context:
----------------------
Input Sanitization serves as an active pre-inference defensive firewall. 
When sensor telemetry arrives at the edge or ingestion gateway, the sanitizer 
filters high-frequency adversarial gradient noise and enforces physical 
rate-of-change continuity before feeding the sequences into the forecasting neural network.

Sanitization Algorithms Implemented:
------------------------------------
1. Rolling Median Filter (RollingMedianSanitizer):
   Non-linear temporal sliding-window filter that strips out alternating sign-based 
   gradient spikes (characteristic of FGSM/BIM) while preserving genuine diurnal weather trends.
2. Physical Rate Limiter (PhysicalRateLimiterSanitizer):
   Enforces maximum plausible physical rate of change between consecutive hourly time steps,
   clipping sudden unphysical sensor jumps exceeding atmospheric domain bounds.
"""

import numpy as np
from scipy.ndimage import median_filter
from src.preprocessing import SENSOR_FEATURE_INDICES


class RollingMedianSanitizer:
    """
    Applies a temporal rolling median filter across physical sensor dimensions.
    Leaves non-sensor deterministic calendar features (year, month, day, hour) untouched.
    """

    def __init__(self, window_size=3):
        """
        Parameters:
        -----------
        window_size : int
            Temporal window size (must be an odd integer >= 3, default: 3).
        """
        if window_size % 2 == 0 or window_size < 1:
            raise ValueError("window_size must be an odd positive integer (e.g., 3, 5).")
        self.window_size = window_size
        self.sensor_indices = SENSOR_FEATURE_INDICES

    def sanitize(self, X):
        """
        Sanitizes input sequences X of shape (N, 24, D).
        Returns a separate sanitized array without modifying the input.
        """
        X_sanitized = np.copy(X)

        # Apply 1D rolling median filter along the time axis (axis=1) for sensor channels only
        for feat_idx in self.sensor_indices:
            # size=(1, window_size) processes each sequence independently across time
            X_sanitized[:, :, feat_idx] = median_filter(
                X_sanitized[:, :, feat_idx],
                size=(1, self.window_size),
                mode='nearest'
            )

        # Ensure valid normalized box bounds [0, 1]
        X_sanitized = np.clip(X_sanitized, 0.0, 1.0)
        return X_sanitized


class PhysicalRateLimiterSanitizer:
    """
    Enforces physical atmospheric rate-of-change continuity constraints.
    Clips abrupt step variations between consecutive hours exceeding calibrated physical limits.
    """

    def __init__(self, delta_percentile=99.5, custom_limits=None):
        """
        Parameters:
        -----------
        delta_percentile : float
            Percentile of clean training deltas used to establish maximum physical hourly delta.
            (Default: 99.5th percentile captures 99.5% of natural clean transitions).
        custom_limits : dict or list or None
            Optional explicit domain limits per sensor feature.
        """
        self.delta_percentile = delta_percentile
        self.custom_limits = custom_limits
        self.sensor_indices = SENSOR_FEATURE_INDICES
        self.max_allowed_deltas = None
        self.is_calibrated = False

    def fit(self, X_clean_train):
        """
        Calibrates maximum physical hourly deltas strictly on clean training sequences.
        Prevents test-set data leakage.
        """
        # X_clean_train: shape (N, 24, D)
        X_sensors = X_clean_train[:, :, self.sensor_indices]
        deltas = np.abs(np.diff(X_sensors, axis=1))  # shape: (N, 23, num_sensors)

        if self.custom_limits is not None:
            self.max_allowed_deltas = np.array(self.custom_limits, dtype=np.float32)
        else:
            # Calibrate 99.5th percentile per sensor feature across clean training sequences
            self.max_allowed_deltas = np.percentile(deltas, self.delta_percentile, axis=(0, 1))

        self.is_calibrated = True
        return self.max_allowed_deltas

    def sanitize(self, X):
        """
        Enforces step-to-step physical continuity constraints.
        Returns a separate sanitized array without modifying the input.
        """
        if not self.is_calibrated:
            raise RuntimeError("PhysicalRateLimiterSanitizer must be calibrated with fit() before sanitizing.")

        X_sanitized = np.copy(X)

        for ch_idx, feat_idx in enumerate(self.sensor_indices):
            limit = self.max_allowed_deltas[ch_idx]
            # Sequential forward pass enforcing |X_t - X_{t-1}| <= limit
            for t in range(1, X_sanitized.shape[1]):
                step_delta = X_sanitized[:, t, feat_idx] - X_sanitized[:, t - 1, feat_idx]
                clipped_delta = np.clip(step_delta, -limit, limit)
                X_sanitized[:, t, feat_idx] = X_sanitized[:, t - 1, feat_idx] + clipped_delta

        X_sanitized = np.clip(X_sanitized, 0.0, 1.0)
        return X_sanitized
