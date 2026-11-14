# Adversarial Attacks and Defenses on Multivariate Time-Series Forecasting

**Course**: AI for Cyber Security (AICS)  
**Project**: Robust Weather & Temperature Forecasting using Recurrent Neural Networks (LSTM & SimpleRNN)

---

## 1. Project Overview

This project investigates the vulnerability of multivariate time-series temperature forecasting models (LSTM and SimpleRNN) to adversarial evasion attacks and evaluates empirical defense mechanisms across the entire ML security lifecycle.

### Research Scope
1. **Baseline Time-Series Forecasting**: Multivariate weather forecasting over hourly sequences ($T=24$, $D=10$) predicting next-hour temperature.
2. **Adversarial Evasion Attacks**: Fast Gradient Sign Method (FGSM) and Basic Iterative Method (BIM/PGD) with domain-aware physical sensor masking.
3. **Attack Detection**: Pre-inference temporal rate-of-change and prediction residual anomaly detection.
4. **Input Sanitization**: Pre-inference non-parametric defense via Rolling Median filtering and Physical Rate Limiting.
5. **Adversarial Training**: Robust model retraining with online clean/adversarial mixture batches.
6. **Clean-Accuracy vs. Robustness Trade-off**: Quantitative assessment of generalization loss vs. attack resilience.

---

## 2. Repository Structure

```
Temperature-prediction/
├── Temperature Prediction New.ipynb       # Original preserved baseline notebook
├── weatherHistory.xls                     # Historical meteorological dataset (96,453 hourly records)
├── requirements.txt                       # Project dependencies
├── README.md                              # Comprehensive project documentation
│
├── src/                                   # Core modular codebase
│   ├── __init__.py                        # Environment setup & MSVC runtime preloading
│   ├── preprocessing.py                   # Data cleaning, sequence building, and leakage-safe scaling
│   ├── models.py                          # LSTM, SimpleRNN architectures and serialization
│   ├── evaluation.py                      # MAE, MSE, RMSE, R² metrics and plotting routines
│   ├── inference.py                       # Operational interactive forecast pipeline
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── fgsm.py                        # Time-series FGSM attack with sensor masking
│   │   └── bim.py                         # Basic Iterative Method (BIM/PGD)
│   └── defenses/
│       ├── __init__.py
│       ├── detection.py                   # Rate-of-Change & Residual Anomaly Detectors
│       ├── sanitization.py                # Rolling Median & Physical Rate Limiter
│       └── adversarial_training.py        # Online adversarial training harness
│
├── saved_models/                          # Checkpointed neural network models & scalers
│   ├── lstm_model.keras                   # Clean baseline LSTM
│   ├── rnn_model.keras                    # Clean baseline SimpleRNN
│   ├── lstm_robust.keras                  # Adversarially trained LSTM
│   ├── rnn_robust.keras                   # Adversarially trained SimpleRNN
│   ├── feature_scaler.joblib              # Fitted feature MinMaxScaler
│   └── target_scaler.joblib               # Fitted target MinMaxScaler
│
├── data/                                  # Isolated evaluation partitions
│   ├── clean/                             # Clean test sequences (X_test_clean, y_test_clean)
│   ├── attacked/                          # Attacked test sequences across all epsilons
│   └── sanitized/                         # Defended/filtered sequences
│
├── experiments/                           # Reproducible experimental runners
│   ├── run_adversarial_attack.py          # Attack benchmarks across all epsilons
│   ├── run_attack_detection.py            # Detection calibration and confusion matrix evaluation
│   ├── run_input_sanitization.py          # Rolling median and rate limiter benchmarks
│   └── run_adversarial_training_experiment.py # Robust model training and final security comparison
│
└── results/                               # Empirical logs, tables, and visualization artifacts
    ├── adversarial_training_config.json   # Exact hyperparameters and training configuration
    ├── final_security_comparison_table.csv# Final clean vs. attacked vs. robust evaluation table
    ├── adversarial_attack_results.csv     # Attack impact data
    ├── attack_detection_results.csv       # Intrusion detection metrics (TPR, FPR, F1)
    ├── input_sanitization_results.csv     # Sanitization recovery data
    └── plots/                             # Academic report visualizations
        ├── adversarial_training/          # 8 Plots for Adversarial Training
        ├── detection/                     # 5 Plots for Attack Detection
        ├── sanitization/                  # 6 Plots for Input Sanitization
        └── comparative/                   # Multi-scenario trajectory comparisons
```

---

## 3. Threat Model & Attack Formulation

* **Attacker Capability**: White-box evasion attack at inference time. The attacker computes loss gradients $\nabla_X \mathcal{L}(f_\theta(X), y)$ with respect to sensor telemetry.
* **Scope & Boundaries**: Strictly local and offline. Operates on local test arrays without touching external networks or SCADA interfaces.
* **Physical Sensor Masking**: Perturbations are restricted to physical sensor channels (`Apparent Temperature`, `Humidity`, `Wind Speed`, `Visibility`, `Cloud Cover`, `Pressure`). Calendar timestamp invariants (`year`, `month`, `day`, `hour`) remain unaltered:
  $$X_{adv} = \text{Clip}_{[0, 1]}\Big(X + \epsilon \cdot \text{sign}\big(\nabla_X \mathcal{L}(f_\theta(X), y)\big) \odot \mathbf{M}_{sensor}\Big)$$

---

## 4. Final Security Comparison Results

Evaluated on $N = 14,465$ unseen test sequences ($T=24$, $D=10$):

| Model | Condition | Epsilon ($\epsilon$) | MAE (°C) | MSE | RMSE (°C) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Original RNN** | Clean | 0.000 | 2.3462 | 7.5873 | 2.7545 | 0.9050 |
| Original RNN | Attacked | 0.001 | 2.4725 | 8.1947 | 2.8626 | 0.8974 |
| Original RNN | Attacked | 0.005 | 2.9787 | 10.9528 | 3.3095 | 0.8629 |
| Original RNN | Attacked | 0.010 | 3.6141 | 15.1443 | 3.8916 | 0.8104 |
| Original RNN | Attacked | 0.020 | 4.8887 | 25.9996 | 5.0990 | 0.6744 |
| Original RNN | Attacked | 0.050 | 8.5880 | 76.0149 | 8.7186 | 0.0482 |
| Original RNN | Attacked | 0.100 | 14.5086 | 213.8411 | 14.6233 | -1.6776 |
| **Robust RNN** | Clean | 0.000 | 2.4508 | 8.6948 | 2.9487 | 0.8911 |
| Robust RNN | Attacked (Adv Trained) | 0.001 | 2.4972 | 8.9560 | 2.9927 | 0.8879 |
| Robust RNN | Attacked (Adv Trained) | 0.005 | 2.6897 | 10.0681 | 3.1730 | 0.8739 |
| Robust RNN | Attacked (Adv Trained) | 0.010 | 2.9425 | 11.6100 | 3.4074 | 0.8546 |
| Robust RNN | Attacked (Adv Trained) | 0.020 | 3.4754 | 15.1962 | 3.8982 | 0.8097 |
| Robust RNN | Attacked (Adv Trained) | 0.050 | 5.1180 | 29.4352 | 5.4254 | 0.6314 |
| Robust RNN | Attacked (Adv Trained) | 0.100 | 7.8649 | 65.1834 | 8.0736 | 0.1838 |
| **Original LSTM** | Clean | 0.000 | 2.1829 | 6.7784 | 2.6035 | 0.9151 |
| Original LSTM | Attacked | 0.001 | 2.2659 | 7.1482 | 2.6736 | 0.9105 |
| Original LSTM | Attacked | 0.005 | 2.5983 | 8.7700 | 2.9614 | 0.8902 |
| Original LSTM | Attacked | 0.010 | 3.0150 | 11.1194 | 3.3346 | 0.8608 |
| Original LSTM | Attacked | 0.020 | 3.8495 | 16.8795 | 4.1085 | 0.7886 |
| Original LSTM | Attacked | 0.050 | 6.2572 | 41.3713 | 6.4321 | 0.4820 |
| Original LSTM | Attacked | 0.100 | 10.1470 | 105.7859 | 10.2852 | -0.3246 |
| **Robust LSTM** | Clean | 0.000 | 3.3843 | 16.7237 | 4.0895 | 0.7906 |
| Robust LSTM | Attacked (Adv Trained) | 0.001 | 3.4266 | 17.0687 | 4.1314 | 0.7863 |
| Robust LSTM | Attacked (Adv Trained) | 0.005 | 3.5996 | 18.5061 | 4.3019 | 0.7683 |
| Robust LSTM | Attacked (Adv Trained) | 0.010 | 3.8245 | 20.4306 | 4.5200 | 0.7442 |
| Robust LSTM | Attacked (Adv Trained) | 0.020 | 4.2973 | 24.7016 | 4.9701 | 0.6907 |
| Robust LSTM | Attacked (Adv Trained) | 0.050 | 5.8055 | 40.8331 | 6.3901 | 0.4887 |
| Robust LSTM | Attacked (Adv Trained) | 0.100 | 8.3653 | 78.4311 | 8.8561 | 0.0179 |

---

## 5. Key Findings & Course Conclusions

1. **Adversarial Training Breakthrough on SimpleRNN**:
   * SimpleRNN achieved a dramatic **72.1% error recovery** under severe attack ($\epsilon = 0.10$), cutting MSE from **213.84 down to 65.18** ($\Delta\text{MSE} = -148.66$).
   * Clean forecasting accuracy was preserved with only a minor trade-off (MSE: $7.59 \rightarrow 8.69$, $R^2: 0.905 \rightarrow 0.891$).
2. **Clean-Accuracy vs. Robustness Trade-off (Tsipras et al.)**:
   * For LSTM, adversarial training improved worst-case robustness at $\epsilon = 0.10$ ($105.79 \rightarrow 78.43$, **27.6% recovery**), but incurred a clean-data accuracy penalty (MSE: $6.78 \rightarrow 16.72$).
   * This empirically demonstrates the fundamental theorem that optimizing for adversarial perturbation envelopes forces models to sacrifice fine-grained clean margin fitting.
3. **Defense-in-Depth Paradigm**:
   * Combining **Residual Anomaly Detection** ($100\%$ detection at $\epsilon \ge 0.05$) with **Input Sanitization** (for un-retrained systems) and **Adversarial Training** (for resilient architectures) provides comprehensive defense across all threat intensities.

---

## 6. How to Reproduce All Experiments

Ensure Python 3.11/3.12 with dependencies installed:
```bash
pip install -r requirements.txt
```

Run experiments sequentially:
```bash
# 1. Adversarial Attacks (FGSM & metrics across all epsilons)
python experiments/run_adversarial_attack.py

# 2. Attack Detection (Rate-of-Change & Residual detectors)
python experiments/run_attack_detection.py

# 3. Input Sanitization (Rolling Median & Physical Rate Limiter)
python experiments/run_input_sanitization.py

# 4. Adversarial Training (Robust models & final security comparison)
python experiments/run_adversarial_training_experiment.py
```

