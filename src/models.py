"""
Model architectures and training routines for LSTM and SimpleRNN.
Exact structural replication of original notebook baseline models.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_lstm_model(input_shape=(24, 10)):
    """Replicates the 3-layer LSTM architecture from the notebook."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ], name="LSTM_Forecaster")
    model.compile(optimizer='adam', loss='mse')
    return model


def build_rnn_model(input_shape=(24, 10)):
    """Replicates the 3-layer SimpleRNN architecture from the notebook."""
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(128, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(64, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ], name="RNN_Forecaster")
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, patience=7, verbose=1):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        shuffle=False,
        verbose=verbose
    )
    return history


def train_rnn(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, patience=5, verbose=1):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=verbose
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        shuffle=False,
        verbose=verbose
    )
    return history


def save_trained_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"Model saved successfully to: {filepath}")


def load_trained_model(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trained model not found at: {filepath}")
    model = keras_load_model(filepath)
    print(f"Model loaded successfully from: {filepath}")
    return model

