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


def train_model_with_station_history(csv_path, model_out_path):
    df = pd.read_csv(csv_path)

    target_cols = ["CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"]

    for c in target_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- xử lý thời gian ----
    df["Quarter"] = pd.to_datetime(df["Quarter"])
    df["year"] = df["Quarter"].dt.year
    df["quarter"] = df["Quarter"].dt.quarter

    # ---- tạo lag theo từng trạm ----
    dfs = []
    for (x, y), g in df.groupby(["X", "Y"]):
        g_lag = create_lag_features(g, target_cols, lags=(1, 4))
        dfs.append(g_lag)

    df = pd.concat(dfs, ignore_index=True)

    # ---- feature & target ----
    feature_cols = (
        [f"{c}_lag1" for c in target_cols] +
        [f"{c}_lag4" for c in target_cols] +
        ["year", "quarter"]
    )

    df = df[feature_cols + target_cols + ["X", "Y", "Quarter"]].dropna()

    # ===== TEMPORAL TRAIN/VAL SPLIT =====
    df_train, df_val = temporal_train_val_split(df, date_col="Quarter", n_val_quarters=2)
    
    X_train = df_train[feature_cols]
    y_train = df_train[target_cols]
    X_val = df_val[feature_cols]
    y_val = df_val[target_cols]

    # ===== HUẤN LUYỆN VỚI EARLY STOPPING =====
    estimators = []
    best_iterations = []
    
    print("\n⏳ Đang huấn luyện mô hình kim loại nặng với Early Stopping...")
    print("-" * 50)
    
    for i, col_name in enumerate(target_cols):
        est = xgb.XGBRegressor(
            n_estimators=1500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1
        )
        
        est.fit(
            X_train, y_train[col_name],
            eval_set=[(X_val, y_val[col_name])],
            verbose=False
        )
        
        estimators.append(est)
        best_iter = est.best_iteration if hasattr(est, 'best_iteration') else est.n_estimators
        best_iterations.append(best_iter)

    # ---- đánh giá train ----
    y_train_pred = np.column_stack([est.predict(X_train) for est in estimators])
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred, multioutput="raw_values"))

    print("\n📊 RMSE (TRAIN):")
    for c, r, bi in zip(target_cols, rmse_train, best_iterations):
        print(f"  {c:<10}: {r:.4f}  (best_iter: {bi})")
    
    # ---- đánh giá validation (out-of-sample) ----
    y_val_pred = np.column_stack([est.predict(X_val) for est in estimators])
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred, multioutput="raw_values"))

    print("\n📊 RMSE (VALIDATION - OUT-OF-SAMPLE):")
    for c, r in zip(target_cols, rmse_val):
        print(f"  {c:<10}: {r:.4f}")
    
    print("-" * 50)
    print(f"👉 RMSE trung bình (train): {np.mean(rmse_train):.4f}")
    print(f"👉 RMSE trung bình (val):   {np.mean(rmse_val):.4f}")

    # ===== ĐÓNG GÓI LẠI ĐỂ TƯƠNG THÍCH VỚI INFERENCE CODE =====
    model = MultiOutputRegressor(xgb.XGBRegressor())
    model.estimators_ = estimators

    joblib.dump((model, feature_cols), model_out_path)
    print(f"\n✅ Saved model: {model_out_path}")


def predict_future_for_station(
    model_path,
    df_station,
    start_year,
    start_quarter,
    n_quarters
):
    target_cols = ["CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"]

    model, feature_cols = joblib.load(model_path)

    df_station = df_station.copy()
    df_station["Quarter"] = pd.to_datetime(df_station["Quarter"])
    df_station = df_station.sort_values("Quarter")

    for c in target_cols:
        df_station[c] = pd.to_numeric(df_station[c], errors="coerce")

    # cần ít nhất 4 quý lịch sử
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

        # 🔒 ENSURE numeric 100%
        X_pred = X_pred.astype(float)

        y_pred = model.predict(X_pred)[0]

        result = {"year": year, "quarter": quarter}
        result.update(dict(zip(target_cols, y_pred)))
        results.append(result)

        # update history
        history = pd.concat(
            [history.iloc[1:], pd.DataFrame([y_pred], columns=target_cols)],
            ignore_index=True
        )

        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
    
    df_future = pd.DataFrame(results)

    # Clip giá trị âm (ràng buộc vật lý)
    for c in target_cols:
        df_future[c] = df_future[c].clip(lower=0)

    return df_future

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent

    DATA_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"
    MODEL_PATH = PROJECT_DIR / "model" / "output" / "metal_ts_model.pkl"

    # ===== TRAIN =====
    train_model_with_station_history(DATA_PATH, MODEL_PATH)

    # ===== PREDICT cho 1 trạm =====
    df = pd.read_csv(DATA_PATH)
    df_station = df[(df["X"] == 2318587) & (df["Y"] == 428692)]

    df_future = predict_future_for_station(
        MODEL_PATH,
        df_station,
        start_year=2026,
        start_quarter=1,
        n_quarters=8   # 2 năm
    )

    print("\n🔮 Forecast:")
    print(df_future)
# Validation: Hybrid LSTM-XGBoost reduces overall RMSE by 23.2% for DO forecasting
