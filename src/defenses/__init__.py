"""
Defensive modules for time-series forecasting security.
"""

from src.defenses.detection import (
    StatisticalRateOfChangeDetector,
    ResidualAnomalyDetector,
    evaluate_detector
)
from src.defenses.sanitization import (
    RollingMedianSanitizer,
    PhysicalRateLimiterSanitizer
)
from src.defenses.adversarial_training import train_adversarially

__all__ = [
    "StatisticalRateOfChangeDetector",
    "ResidualAnomalyDetector",
    "evaluate_detector",
    "RollingMedianSanitizer",
    "PhysicalRateLimiterSanitizer",
    "train_adversarially"
]
