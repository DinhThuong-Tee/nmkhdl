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
    val_mask = df.groupby(["X", "Y"])[date_col].transform(
        lambda x: x >= x.nlargest(n_val_quarters).min()
    )
    df_train = df[~val_mask].copy()
    df_val = df[val_mask].copy()
    print(f"📊 Train size: {len(df_train)} | Val size: {len(df_val)}")
    return df_train, df_val

def train_model_with_station_history(csv_path, model_out_path):
    df = pd.read_csv(csv_path)
    target_cols = ["CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"]
    for c in target_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Quarter"] = pd.to_datetime(df["Quarter"])
    df["year"] = df["Quarter"].dt.year
    df["quarter"] = df["Quarter"].dt.quarter

    dfs = []
    for (x, y), g in df.groupby(["X", "Y"]):
        g_lag = create_lag_features(g, target_cols, lags=(1, 4))
        dfs.append(g_lag)
    df = pd.concat(dfs, ignore_index=True)

    feature_cols = (
        [f"{c}_lag1" for c in target_cols] +
        [f"{c}_lag4" for c in target_cols] +
        ["year", "quarter"]
    )
    df = df[feature_cols + target_cols + ["X", "Y", "Quarter"]].dropna()

    df_train, df_val = temporal_train_val_split(df, date_col="Quarter", n_val_quarters=2)
    X_train = df_train[feature_cols]
    y_train = df_train[target_cols]
    X_val = df_val[feature_cols]
    y_val = df_val[target_cols]

    model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred, multioutput="raw_values"))
    print("\n📊 RMSE (TRAIN):")
    for c, r in zip(target_cols, rmse_train):
        print(f"  {c:<10}: {r:.4f}")

    y_val_pred = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred, multioutput="raw_values"))
    print("\n📊 RMSE (VALIDATION):")
    for c, r in zip(target_cols, rmse_val):
        print(f"  {c:<10}: {r:.4f}")

    joblib.dump((model, feature_cols), model_out_path)
    print(f"\n✅ Saved model: {model_out_path}")

def predict_future_for_station(model_path, df_station, start_year, start_quarter, n_quarters):
    target_cols = ["CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"]
    model, feature_cols = joblib.load(model_path)
    df_station = df_station.copy()
    df_station["Quarter"] = pd.to_datetime(df_station["Quarter"])
    df_station = df_station.sort_values("Quarter")
    for c in target_cols:
        df_station[c] = pd.to_numeric(df_station[c], errors="coerce")
    history = df_station[target_cols].iloc[-4:].copy()
    results = []
    year, quarter = start_year, start_quarter
    for _ in range(n_quarters):
        row = {}
        for c in target_cols:
            row[f"{c}_lag1"] = float(history[c].iloc[-1])
            row[f"{c}_lag4"] = float(history[c].iloc[0])
        row["year"] = int(year)
        row["quarter"] = int(quarter)
        X_pred = pd.DataFrame([row])[feature_cols]
        X_pred = X_pred.astype(float)
        y_pred = model.predict(X_pred)[0]
        result = {"year": year, "quarter": quarter}
        result.update(dict(zip(target_cols, y_pred)))
        results.append(result)
        history = pd.concat([history.iloc[1:], pd.DataFrame([y_pred], columns=target_cols)], ignore_index=True)
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
    df_future = pd.DataFrame(results)
    for c in target_cols:
        df_future[c] = df_future[c].clip(lower=0)
    return df_future

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    DATA_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"
    MODEL_PATH = PROJECT_DIR / "model" / "output" / "metal_ts_model.pkl"
    train_model_with_station_history(DATA_PATH, MODEL_PATH)
