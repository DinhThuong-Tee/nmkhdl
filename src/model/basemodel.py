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


#  HÀU
OYSTER_FEATURES = [
    'DO', 'Temperature', 'pH', 'Salinity', 'NH3', 'H2S', 'BOD5', 'COD',
    'TSS', 'Coliform', 'Alkalinity', 'Transparency',
]

#  CÁ GIÒ
COBIA_FEATURES = [
    'DO', 'Temperature', 'pH', 'Salinity', 'NH3', 'PO4', 'BOD5', 'COD',
    'TSS', 'Coliform', 'Alkalinity', 'Transparency'
]


def prepare_time_series_data(csv_path, features_list, lags=[1, 4]):
    import pandas as pd
    # 1. Đọc dữ liệu
    df = pd.read_csv(str(csv_path), encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    
    # 2. Ép tên cột đầu tiên là Station để chắc chắn
    df.rename(columns={df.columns[0]: 'Station'}, inplace=True)
    
    # 3. Lọc các cột cần thiết và xử lý ngày tháng
    valid_f = [f for f in features_list if f in df.columns]
    df = df[['Station', 'Quarter'] + valid_f].copy()
    df['Date'] = pd.to_datetime(df['Quarter'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values(['Station', 'Date'])

    # 4. Điền dữ liệu thiếu (chỉ dùng dữ liệu quá khứ để tránh data leakage)
    #    Sử dụng ffill (forward-fill) — giá trị tương lai KHÔNG bao giờ được dùng.
    #    NaN đầu chuỗi (trước quan trắc đầu tiên) được giữ nguyên và sẽ bị loại
    #    bởi dropna() sau khi tạo lag features (shift(4) tạo NaN ở 4 dòng đầu).
    for col in valid_f:
        df[col] = df.groupby('Station')[col].transform(
            lambda x: x.ffill()
        )

    # 5. Tạo các cột Lag (Trễ)
    lag_cols = []
    for col in valid_f:
        for lag in lags:
            new_col_name = f"{col}_lag{lag}"
            lag_cols.append(new_col_name)
            # Dùng transform shift để đảm bảo an toàn
            df[new_col_name] = df.groupby('Station')[col].shift(lag)
    
    df['Quarter_Num'] = df['Date'].dt.quarter
    
    # 6. Loại bỏ các dòng NaN do quá trình shift tạo ra
    df_final = df.dropna().copy()
    
    print(f"--- THÀNH CÔNG ---")
    print(f"Cột hiện có: {df_final.columns.tolist()[:3]}...")
    print(f"Kích thước: {df_final.shape}")
    
    return df_final, lag_cols + ['Quarter_Num']

# Hàm xử lý ngoại lệ
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
    """
    Tách dữ liệu thành tập huấn luyện và tập kiểm chứng theo thời gian.
    
    Với mỗi trạm, N quý cuối cùng (theo thứ tự thời gian) được giữ lại
    làm tập validation. Đảm bảo không có rò rỉ dữ liệu từ tương lai.
    Nếu trạm có ít hơn n_val_quarters+1 quý, chỉ giữ lại tối đa
    len(group)-1 quý cho validation để đảm bảo luôn có ít nhất 1 dòng
    trong tập huấn luyện.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã chuẩn bị xong (có cột 'Station' và 'Date').
    n_val_quarters : int
        Số quý cuối cùng dùng làm tập validation cho mỗi trạm.
    
    Returns
    -------
    df_train, df_val : tuple of pd.DataFrame
    """
    def _val_mask_for_group(dates):
        n = min(n_val_quarters, len(dates) - 1)  # Luôn giữ ≥1 dòng cho train
        if n <= 0:
            return pd.Series(False, index=dates.index)
        return dates >= dates.nlargest(n).min()

    val_mask = df.groupby('Station')['Date'].transform(_val_mask_for_group)
    df_train = df[~val_mask].copy()
    df_val = df[val_mask].copy()
    
    print(f"📊 Train size: {len(df_train)} | Val size: {len(df_val)}")
    return df_train, df_val


# Hàm huấn luyện
def train_forecast_model(csv_path, features, model_out_path, meta_out_path=None):
    model_out_path = str(model_out_path)
    
    df_all, input_cols = prepare_time_series_data(csv_path, features, lags=[1, 4])
    
    if df_all is None:
        return
    
    df_all = handle_outliers(df_all, features)

    # ===== TEMPORAL TRAIN/VAL SPLIT =====
    df_train, df_val = temporal_train_val_split(df_all, n_val_quarters=2)
    
    X_train = df_train[input_cols]
    y_train = df_train[features]
    X_val = df_val[input_cols]
    y_val = df_val[features]

    # ===== HUẤN LUYỆN VỚI EARLY STOPPING =====
    # Sử dụng vòng lặp thủ công thay vì MultiOutputRegressor để hỗ trợ
    # early stopping cho từng biến mục tiêu
    estimators = []
    best_iterations = []
    
    print("\n⏳ Đang huấn luyện mô hình với Early Stopping...")
    print("-" * 50)
    
    for i, col_name in enumerate(features):
        est = xgb.XGBRegressor(
            n_estimators=2000,          # Tăng giới hạn trên, early stopping sẽ dừng sớm
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42
        )
        
        est.fit(
            X_train, y_train[col_name],
            eval_set=[(X_val, y_val[col_name])],
            verbose=False
        )
        
        estimators.append(est)
        best_iter = est.best_iteration if hasattr(est, 'best_iteration') else est.n_estimators
        best_iterations.append(best_iter)
    
    # ===== ĐÁNH GIÁ TRÊN TẬP HUẤN LUYỆN =====
    print("\n📊 KẾT QUẢ ĐÁNH GIÁ (TRAINING SCORE):")
    print("-" * 50)
    
    y_train_pred = np.column_stack([est.predict(X_train) for est in estimators])
    mse_train = mean_squared_error(y_train, y_train_pred, multioutput='raw_values')
    rmse_train = np.sqrt(mse_train)
    
    for i, col_name in enumerate(features):
        print(f"   🔹 {col_name:<15} RMSE(train): {rmse_train[i]:.4f}  "
              f"(best_iter: {best_iterations[i]})")
    
    # ===== ĐÁNH GIÁ TRÊN TẬP VALIDATION (OUT-OF-SAMPLE) =====
    print("\n📊 KẾT QUẢ ĐÁNH GIÁ (VALIDATION SCORE - OUT-OF-SAMPLE):")
    print("-" * 50)
    
    y_val_pred = np.column_stack([est.predict(X_val) for est in estimators])
    mse_val = mean_squared_error(y_val, y_val_pred, multioutput='raw_values')
    rmse_val = np.sqrt(mse_val)
    
    for i, col_name in enumerate(features):
        print(f"   🔹 {col_name:<15} RMSE(val): {rmse_val[i]:.4f}")
        
    print("-" * 50)
    print(f"👉 RMSE trung bình (train): {np.mean(rmse_train):.4f}")
    print(f"👉 RMSE trung bình (val):   {np.mean(rmse_val):.4f}")

    # ===== ĐÓNG GÓI THÀNH MultiOutputRegressor-COMPATIBLE OBJECT =====
    # Tạo lại wrapper để giữ tương thích với code inference hiện tại
    # (model.predict(X) trả về ma trận [n_samples, n_features])
    model = MultiOutputRegressor(xgb.XGBRegressor())
    model.estimators_ = estimators

    # Lưu model
    joblib.dump(model, model_out_path)
    print(f"\n🎉 Đã lưu model tại: {model_out_path}")

    # Lưu metadata
    if meta_out_path is None:
        meta_out_path = model_out_path.replace('.pkl', '_features.pkl')
    
    joblib.dump((input_cols, features), meta_out_path)
    print(f"ℹ️  Đã lưu danh sách features tại: {meta_out_path}")


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