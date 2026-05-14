import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib


def create_lag_features(df, target_cols, lags=(1, 4)):
    df = df.sort_values("Quarter").copy()

    for col in target_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df

def temporal_train_val_split(df, date_col="Quarter", n_val_quarters=2):
    """
    Tách dữ liệu thành tập huấn luyện và tập kiểm chứng theo thời gian.
    
    Với mỗi trạm (nhóm theo X, Y), N quý cuối cùng được giữ lại
    làm tập validation. Nếu trạm có ít hơn n_val_quarters+1 quý,
    chỉ giữ lại tối đa len(group)-1 quý cho validation.
    """
    def _val_mask_for_group(dates):
        n = min(n_val_quarters, len(dates) - 1)
        if n <= 0:
            return pd.Series(False, index=dates.index)
        return dates >= dates.nlargest(n).min()

    val_mask = df.groupby(["X", "Y"])[date_col].transform(_val_mask_for_group)
    df_train = df[~val_mask].copy()
    df_val = df[val_mask].copy()
    
    print(f"📊 Train size: {len(df_train)} | Val size: {len(df_val)}")
    return df_train, df_val

