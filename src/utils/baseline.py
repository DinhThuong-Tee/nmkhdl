"""
Persistence Baseline for Time-Series Forecasting.

A persistence (naïve) model predicts that the future will be identical
to the most recent observation.  It serves as the minimum-performance
benchmark: any useful forecasting model should beat persistence.
"""

import pandas as pd
import numpy as np


def persistence_forecast(history, n_quarters, features):
    """
    Dự báo bằng phương pháp lặp lại giá trị cuối cùng (Persistence / Naïve Baseline).
    """
    last_obs = history[features].iloc[-1].to_dict()
    return pd.DataFrame([last_obs] * n_quarters)
