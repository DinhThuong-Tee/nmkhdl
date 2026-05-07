"""
Persistence Baseline for Time-Series Forecasting.

A persistence (naïve) model predicts that the future will be identical
to the most recent observation.  It serves as the minimum-performance
benchmark: any useful forecasting model should beat persistence.

Usage
-----
    from utils.baseline import persistence_forecast, evaluate_persistence

    df_pers = persistence_forecast(history_df, n_quarters=4, features=FEATURES)
    rmse_dict = evaluate_persistence(actual_df, features=FEATURES)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def persistence_forecast(history, n_quarters, features):
    """
    Dự báo bằng phương pháp lặp lại giá trị cuối cùng (Persistence / Naïve Baseline).

    Parameters
    ----------
    history : pd.DataFrame
        Dữ liệu lịch sử (ít nhất 1 dòng) chứa các cột trong `features`.
    n_quarters : int
        Số quý cần dự báo.
    features : list of str
        Danh sách các biến cần dự báo.

    Returns
    -------
    pd.DataFrame
        DataFrame với `n_quarters` dòng, mỗi dòng chứa giá trị bằng
        dòng cuối cùng của `history`.
    """
    last_obs = history[features].iloc[-1].to_dict()
    return pd.DataFrame([last_obs] * n_quarters)


def seasonal_persistence_forecast(history, n_quarters, features):
    """
    Dự báo bằng phương pháp lặp lại giá trị cùng kỳ năm trước
    (Seasonal Persistence / Seasonal Naïve).

    Yêu cầu ít nhất 4 quý lịch sử (để có 1 chu kỳ năm hoàn chỉnh).

    Parameters
    ----------
    history : pd.DataFrame
        Dữ liệu lịch sử (≥ 4 dòng).
    n_quarters : int
        Số quý cần dự báo.
    features : list of str
        Danh sách các biến cần dự báo.

    Returns
    -------
    pd.DataFrame
    """
    cycle_len = min(4, len(history))
    rows = []
    for i in range(n_quarters):
        idx = -(cycle_len - (i % cycle_len))
        rows.append(history[features].iloc[idx].to_dict())
    return pd.DataFrame(rows)


def evaluate_persistence(df, features, group_col="Station", date_col="Date"):
    """
    Đánh giá baseline Persistence trên tập dữ liệu đã có giá trị thực.

    Persistence dự đoán Y(t) = Y(t-1) cho mọi t.  Hàm này tính RMSE
    cho từng biến và trả về dict.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã sắp xếp theo [group_col, date_col], chứa các cột `features`.
    features : list of str
        Danh sách biến đánh giá.
    group_col : str
        Cột nhóm (ví dụ: Station).
    date_col : str
        Cột thời gian (đã được parse sang datetime).

    Returns
    -------
    dict[str, float]
        Mapping: feature_name -> RMSE (persistence).
    """
    df = df.sort_values([group_col, date_col]).copy()

    rmse_results = {}
    for col in features:
        # Persistence prediction: shift(1) within each station group
        df[f"{col}_pers"] = df.groupby(group_col)[col].shift(1)

    # Drop rows without a valid persistence prediction (first row per group)
    df_eval = df.dropna(subset=[f"{features[0]}_pers"])

    for col in features:
        actual = df_eval[col].values
        predicted = df_eval[f"{col}_pers"].values
        rmse_results[col] = np.sqrt(mean_squared_error(actual, predicted))

    return rmse_results


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"

    FEATURES = [
        'DO', 'Temperature', 'pH', 'Salinity', 'NH3', 'H2S', 'BOD5', 'COD',
        'TSS', 'Coliform', 'Alkalinity', 'Transparency',
    ]

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Quarter"], errors="coerce")
    df = df.dropna(subset=["Date"])

    rmse_dict = evaluate_persistence(df, FEATURES)

    print("\n📊 PERSISTENCE BASELINE RMSE:")
    print("-" * 40)
    for feat, rmse in rmse_dict.items():
        print(f"  {feat:<15}: {rmse:.4f}")
    print("-" * 40)
    print(f"  {'MEAN':<15}: {np.mean(list(rmse_dict.values())):.4f}")
