"""
Fast Gradient Sign Method (FGSM) for Multivariate Time-Series Forecasting.

Cybersecurity Context:
----------------------
Threat Model: Test-time evasion attack against neural forecasting pipelines.
The adversary crafts minimal additive perturbations to multivariate sensor sequences 
to maximize prediction error while remaining stealthy under L_infinity budget epsilon.

Mathematical Formulation:
--------------------------
    X_adv = Clip(X + epsilon * sign(grad_X(Loss(f(X), y))), min=0.0, max=1.0)

Safety:
-------
This module runs strictly offline on local evaluation sets. It does not interface 
with any live sensor, SCADA network, or external API.
"""

import os
import numpy as np
import tensorflow as tf

from src.preprocessing import SENSOR_FEATURE_INDICES


def generate_fgsm_attack(
    model,
    X,
    y,
    epsilon=0.01,
    sensor_only=True,
    clip_min=0.0,
    clip_max=1.0,
    batch_size=256
):
    """
    Generates FGSM adversarial perturbations for a batch of time-series sequences.

    Parameters:
    -----------
    model : tf.keras.Model
        Trained forecasting model (LSTM or RNN).
    X : np.ndarray
        Clean input sequences of shape (N, sequence_length, num_features).
    y : np.ndarray
        True target values of shape (N, 1).
    epsilon : float
        Perturbation magnitude (L_infinity budget). If epsilon == 0, returns identical copy of X.
    sensor_only : bool
        If True, restricts perturbations strictly to physical sensor features
        (preserving deterministic calendar timestamp fields: year, month, day, hour).
    clip_min : float
        Minimum valid normalized feature value (default 0.0).
    clip_max : float
        Maximum valid normalized feature value (default 1.0).
    batch_size : int
        Batch size for gradient computation to prevent GPU/CPU memory exhaustion.

    Returns:
    --------
    X_adv : np.ndarray
        Adversarially perturbed input sequences of same shape as X.
    perturbation : np.ndarray
        The additive perturbation array (X_adv - X).
    """
    if epsilon == 0.0 or epsilon == 0:
        # Epsilon 0 represents clean unattacked baseline
        return np.copy(X), np.zeros_like(X)

    N = X.shape[0]
    X_adv_list = []

    # Create feature mask (1 for perturbable features, 0 for invariant calendar features)
    feature_mask = np.ones((1, 1, X.shape[2]), dtype=np.float32)
    if sensor_only:
        feature_mask[:] = 0.0
        feature_mask[0, 0, SENSOR_FEATURE_INDICES] = 1.0
    mask_tensor = tf.constant(feature_mask, dtype=tf.float32)

    loss_fn = tf.keras.losses.MeanSquaredError()

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        X_batch = tf.convert_to_tensor(X[start_idx:end_idx], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y[start_idx:end_idx], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_batch)
            y_pred = model(X_batch, training=False)
            loss_val = loss_fn(y_batch, y_pred)

        # Calculate loss gradient with respect to input sequences: grad_X L(f(X), y)
        gradients = tape.gradient(loss_val, X_batch)

        # Apply sign of the gradient
        signed_grad = tf.sign(gradients)

        # Apply sensor mask to prevent altering clock/calendar features
        effective_perturbation = tf.cast(epsilon, tf.float32) * signed_grad * mask_tensor

        # Add perturbation and project back into valid normalized box [clip_min, clip_max]
        X_batch_adv = tf.clip_by_value(X_batch + effective_perturbation, clip_min, clip_max)
        X_adv_list.append(X_batch_adv.numpy())

    X_adv = np.concatenate(X_adv_list, axis=0)
    perturbation = X_adv - X
    return X_adv, perturbation

