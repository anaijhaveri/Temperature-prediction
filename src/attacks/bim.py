"""
Basic Iterative Method (BIM / PGD) for Multivariate Time-Series Forecasting.

Cybersecurity Context:
----------------------
Threat Model: Multi-step gradient-based evasion attack. While single-step FGSM 
applies one linear approximation, BIM iterates K times with smaller step size alpha, 
finding stronger adversarial examples within the L_infinity epsilon-ball.

Mathematical Formulation:
--------------------------
    X^(0) = X
    X^(t+1) = Clip_{X, eps}( Clip_{[0, 1]}( X^(t) + alpha * sign(grad_{X^(t)} L) * M_sensor ) )
"""

import numpy as np
import tensorflow as tf
from src.preprocessing import SENSOR_FEATURE_INDICES


def generate_bim_attack(
    model,
    X,
    y,
    epsilon=0.02,
    alpha=0.005,
    num_iter=10,
    sensor_only=True,
    clip_min=0.0,
    clip_max=1.0,
    batch_size=256
):
    """
    Generates iterative adversarial perturbations using BIM / PGD.

    Parameters:
    -----------
    model : tf.keras.Model
        Trained forecasting model.
    X : np.ndarray
        Clean input sequences (N, sequence_length, num_features).
    y : np.ndarray
        True targets (N, 1).
    epsilon : float
        Maximum total L_infinity perturbation budget.
    alpha : float
        Step size per iteration (typically epsilon / num_iter or small constant).
    num_iter : int
        Number of gradient iterations (default: 10).
    sensor_only : bool
        If True, restricts perturbations strictly to physical sensor channels.
    """
    if epsilon == 0.0 or epsilon == 0:
        return np.copy(X), np.zeros_like(X)

    N = X.shape[0]
    X_adv_list = []

    # Sensor feature mask
    feature_mask = np.ones((1, 1, X.shape[2]), dtype=np.float32)
    if sensor_only:
        feature_mask[:] = 0.0
        feature_mask[0, 0, SENSOR_FEATURE_INDICES] = 1.0
    mask_tensor = tf.constant(feature_mask, dtype=tf.float32)

    loss_fn = tf.keras.losses.MeanSquaredError()

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        X_orig_batch = tf.convert_to_tensor(X[start_idx:end_idx], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y[start_idx:end_idx], dtype=tf.float32)

        # Initialize adversarial sample at clean input
        X_adv_batch = tf.identity(X_orig_batch)

        for _ in range(num_iter):
            with tf.GradientTape() as tape:
                tape.watch(X_adv_batch)
                y_pred = model(X_adv_batch, training=False)
                loss_val = loss_fn(y_batch, y_pred)

            grad = tape.gradient(loss_val, X_adv_batch)
            step_pert = tf.cast(alpha, tf.float32) * tf.sign(grad) * mask_tensor

            # Step and project to [X - eps, X + eps]
            X_adv_batch = X_adv_batch + step_pert
            X_adv_batch = tf.clip_by_value(X_adv_batch, X_orig_batch - epsilon, X_orig_batch + epsilon)
            # Project to valid data range [clip_min, clip_max]
            X_adv_batch = tf.clip_by_value(X_adv_batch, clip_min, clip_max)

        X_adv_list.append(X_adv_batch.numpy())

    X_adv = np.concatenate(X_adv_list, axis=0)
    perturbation = X_adv - X
    return X_adv, perturbation

