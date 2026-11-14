"""
Cybersecurity Defense: Adversarial Training for Time-Series Forecasting.

Cybersecurity Context:
----------------------
Adversarial Training is a model-level defense that embeds adversarial perturbations 
directly into the loss optimization objective. Rather than relying solely on pre-inference 
filters, the neural network adjusts its internal recurrent weights and cell states to 
minimize loss on both clean weather sequences and worst-case gradient-perturbed inputs.

Formulation:
------------
    L_total(theta) = alpha * L(f_theta(X_clean), y) + (1 - alpha) * L(f_theta(X_adv), y)
    where X_adv = Clip_{[0, 1]}(X_clean + eps_train * sign(grad_X L) * M_sensor)
"""

import os
import time
import numpy as np
import tensorflow as tf
from src.preprocessing import SENSOR_FEATURE_INDICES


def train_adversarially(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epsilon_train=0.02,
    alpha=0.5,
    epochs=8,
    batch_size=256,
    learning_rate=0.001,
    random_seed=42,
    patience=3,
    verbose=1
):
    """
    Trains a forecasting model using online adversarial training.

    Parameters:
    -----------
    model : tf.keras.Model
        The un-trained (or fresh) model instance.
    X_train, y_train : np.ndarray
        Clean training sequences and targets.
    X_val, y_val : np.ndarray
        Clean validation sequences and targets.
    epsilon_train : float
        Adversarial training budget (default: 0.02).
    alpha : float
        Weight assigned to clean loss (1 - alpha assigned to adversarial loss, default: 0.5).
    epochs : int
        Maximum number of training epochs (default: 8).
    batch_size : int
        Batch size (default: 256).
    learning_rate : float
        Adam optimizer learning rate (default: 0.001).
    random_seed : int
        Seed for reproducibility.
    patience : int
        Early stopping patience based on validation loss.

    Returns:
    --------
    history : dict
        Training and validation loss progression.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Pre-build sensor mask tensor
    feature_mask = np.zeros((1, 1, X_train.shape[2]), dtype=np.float32)
    feature_mask[0, 0, SENSOR_FEATURE_INDICES] = 1.0
    mask_tensor = tf.constant(feature_mask, dtype=tf.float32)

    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))

    # Prepare validation tensor
    X_val_tf = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val_tf = tf.convert_to_tensor(y_val, dtype=tf.float32)

    history = {"loss": [], "val_loss": [], "clean_loss": [], "adv_loss": []}
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    print(f"Beginning Adversarial Training (ε_train={epsilon_train}, α={alpha}, epochs={epochs})...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        epoch_clean_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_total_loss = 0.0

        for b in range(num_batches):
            start_i = b * batch_size
            end_i = min(start_i + batch_size, num_samples)

            X_b = tf.convert_to_tensor(X_train[start_i:end_i], dtype=tf.float32)
            y_b = tf.convert_to_tensor(y_train[start_i:end_i], dtype=tf.float32)

            # 1. Compute loss gradient w.r.t input to generate online FGSM sample
            with tf.GradientTape() as input_tape:
                input_tape.watch(X_b)
                pred_eval = model(X_b, training=False)
                loss_eval = loss_fn(y_b, pred_eval)

            grad_X = input_tape.gradient(loss_eval, X_b)
            # Create online adversarial sample with sensor masking
            pert = tf.cast(epsilon_train, tf.float32) * tf.sign(grad_X) * mask_tensor
            X_b_adv = tf.clip_by_value(X_b + pert, 0.0, 1.0)

            # 2. Joint forward pass and weight gradient optimization
            with tf.GradientTape() as tape:
                pred_clean = model(X_b, training=True)
                l_clean = loss_fn(y_b, pred_clean)

                pred_adv = model(X_b_adv, training=True)
                l_adv = loss_fn(y_b, pred_adv)

                l_total = tf.cast(alpha, tf.float32) * l_clean + tf.cast(1.0 - alpha, tf.float32) * l_adv

            grads = tape.gradient(l_total, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_clean_loss += float(l_clean)
            epoch_adv_loss += float(l_adv)
            epoch_total_loss += float(l_total)

        epoch_clean_loss /= num_batches
        epoch_adv_loss /= num_batches
        epoch_total_loss /= num_batches

        # Validation evaluation
        val_pred = model(X_val_tf, training=False)
        val_loss = float(loss_fn(y_val_tf, val_pred))
        epoch_time = time.time() - t0

        history["loss"].append(epoch_total_loss)
        history["clean_loss"].append(epoch_clean_loss)
        history["adv_loss"].append(epoch_adv_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"Epoch {epoch}/{epochs} ({epoch_time:4.1f}s) - "
                  f"Total Loss: {epoch_total_loss:.4f} (Clean: {epoch_clean_loss:.4f}, Adv: {epoch_adv_loss:.4f}) - "
                  f"Val Loss: {val_loss:.4f}", flush=True)

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Restoring best model weights.", flush=True)
                break

    if best_weights is not None:
        model.set_weights(best_weights)

    return history
