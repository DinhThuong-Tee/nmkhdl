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


def seasonal_persistence_forecast(history, n_quarters, features):
    """
    Dự báo bằng phương pháp lặp lại giá trị cùng kỳ năm trước
    (Seasonal Persistence / Seasonal Naïve).

    Yêu cầu ít nhất 4 quý lịch sử (để có 1 chu kỳ năm hoàn chỉnh).
    """
    cycle_len = min(4, len(history))
    rows = []
    for i in range(n_quarters):
        idx = -(cycle_len - (i % cycle_len))
        rows.append(history[features].iloc[idx].to_dict())
    return pd.DataFrame(rows)
