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
