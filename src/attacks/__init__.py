"""
Adversarial attack modules for multivariate time-series forecasting.
"""

from src.attacks.fgsm import generate_fgsm_attack
from src.attacks.bim import generate_bim_attack

__all__ = ["generate_fgsm_attack", "generate_bim_attack"]
