import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
import os
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

OYSTER_FEATURES = [
    'DO', 'Temperature', 'pH', 'Salinity', 'NH3', 'H2S', 'BOD5', 'COD',
    'TSS', 'Coliform', 'Alkalinity', 'Transparency',
]

COBIA_FEATURES = [
    'DO', 'Temperature', 'pH', 'Salinity', 'NH3', 'PO4', 'BOD5', 'COD',
    'TSS', 'Coliform', 'Alkalinity', 'Transparency'
]

def prepare_time_series_data(csv_path, features_list, lags=[1, 4]):
    df = pd.read_csv(str(csv_path), encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df.rename(columns={df.columns[0]: 'Station'}, inplace=True)
    valid_f = [f for f in features_list if f in df.columns]
    df = df[['Station', 'Quarter'] + valid_f].copy()
    df['Date'] = pd.to_datetime(df['Quarter'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values(['Station', 'Date'])

    for col in valid_f:
        df[col] = df.groupby('Station')[col].transform(
            lambda x: x.interpolate(limit_direction='forward').fillna(x.median())
        )

    lag_cols = []
    for col in valid_f:
        for lag in lags:
            new_col_name = f"{col}_lag{lag}"
            lag_cols.append(new_col_name)
            df[new_col_name] = df.groupby('Station')[col].shift(lag)
    
    df['Quarter_Num'] = df['Date'].dt.quarter
    df_final = df.dropna().copy()
    return df_final, lag_cols + ['Quarter_Num']

def clip_percentile(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

def handle_outliers(df, features):
    df = df.copy()
    log_cols = ["Coliform", "TSS", "BOD5", "NH3"]
    for c in log_cols:
        if c in features and c in df.columns:
            df[c] = clip_percentile(df[c], 0.01, 0.99)
    return df

def temporal_train_val_split(df, n_val_quarters=2):
    val_mask = df.groupby('Station')['Date'].transform(
        lambda x: x >= x.nlargest(n_val_quarters).min()
    )
    df_train = df[~val_mask].copy()
    df_val = df[val_mask].copy()
    print(f"📊 Train size: {len(df_train)} | Val size: {len(df_val)}")
    return df_train, df_val

def train_forecast_model(csv_path, features, model_out_path, meta_out_path=None):
    model_out_path = str(model_out_path)
    df_all, input_cols = prepare_time_series_data(csv_path, features, lags=[1, 4])
    if df_all is None:
        return
    df_all = handle_outliers(df_all, features)

    df_train, df_val = temporal_train_val_split(df_all, n_val_quarters=2)
    X_train = df_train[input_cols]
    y_train = df_train[features]
    X_val = df_val[input_cols]
    y_val = df_val[features]

    model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    ))
    model.fit(X_train, y_train)

    print("\n📊 KẾT QUẢ ĐÁNH GIÁ (TRAINING SCORE):")
    print("-" * 50)
    y_train_pred = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred, multioutput='raw_values'))
    for i, col_name in enumerate(features):
        print(f"   🔹 {col_name:<15} RMSE(train): {rmse_train[i]:.4f}")

    print("\n📊 KẾT QUẢ ĐÁNH GIÁ (VALIDATION SCORE - OUT-OF-SAMPLE):")
    print("-" * 50)
    y_val_pred = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred, multioutput='raw_values'))
    for i, col_name in enumerate(features):
        print(f"   🔹 {col_name:<15} RMSE(val): {rmse_val[i]:.4f}")

    joblib.dump(model, model_out_path)
    if meta_out_path is None:
        meta_out_path = model_out_path.replace('.pkl', '_features.pkl')
    joblib.dump((input_cols, features), meta_out_path)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent 
    OUTPUT_DIR = PROJECT_DIR / "model" / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = PROJECT_DIR / "data" / "hk_water_quality"

    train_forecast_model(
        csv_path = DATA_DIR / "hk_oyster_quarterly_21vars.csv",
        features = OYSTER_FEATURES,
        model_out_path = OUTPUT_DIR / "hk_oyster_forecast_model.pkl"
    )

    train_forecast_model(
        csv_path = DATA_DIR / "hk_cobia_quarterly_21vars.csv",
        features = COBIA_FEATURES,
        model_out_path = OUTPUT_DIR / "hk_cobia_forecast_model.pkl"
    )
