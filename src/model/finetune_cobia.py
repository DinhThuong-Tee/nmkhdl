import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error

from basemodel import *

warnings.filterwarnings('ignore')

def finetune_model(base_model_path, new_data_path, output_path, features_list):
    """
    Hàm Fine-tune: Cập nhật mô hình cũ với dữ liệu mới.
    Sử dụng temporal split và early stopping để tránh overfitting.
    """
    base_model_path = str(base_model_path)
    output_path = str(output_path)
    
    print(f"\n🔧 BẮT ĐẦU FINE-TUNE MÔ HÌNH TỪ: {base_model_path}")
    
    # 1. LOAD MÔ HÌNH GỐC (BASE MODEL)
    if not os.path.exists(base_model_path):
        print(f"❌ Lỗi: Không tìm thấy file model gốc tại {base_model_path}")
        return
    
        model = joblib.load(base_model_path)
    print("✅ Đã load xong model gốc.")

    # 2. LOAD METADATA (Để biết ngày xưa train dùng cột nào)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    try:
        input_cols_old, features_old = joblib.load(meta_path)
        print("✅ Đã xác định được cấu trúc input/output cũ.")
    except:
        print("❌ Lỗi: Không tìm thấy file metadata (_features.pkl). Không thể fine-tune chuẩn.")
        return

    # 3. CHUẨN BỊ DỮ LIỆU MỚI (FINE-TUNE DATA)
    # Lưu ý: Phải dùng logic y hệt như lúc train base model
    print(f"🔄 Đang xử lý dữ liệu mới từ: {new_data_path}")
    df_ft, _ = prepare_time_series_data(new_data_path, features_list, lags=[1, 4])
    
    if df_ft is None or len(df_ft) == 0:
        print("⚠️ Dữ liệu fine-tune trống hoặc không đủ để tạo lag. Hủy bỏ.")
        return

    # Đảm bảo dữ liệu mới có đủ các cột như dữ liệu cũ

    # ===== TEMPORAL TRAIN/VAL SPLIT =====
    df_train, df_val = temporal_train_val_split(df_ft, n_val_quarters=2)
    
    X_train = df_train[input_cols_old]
    y_train = df_train[features_list]
    X_val = df_val[input_cols_old]
    y_val = df_val[features_list]

    print(f"📊 Kích thước dữ liệu Fine-tune: Train={len(X_train)}, Val={len(X_val)}")

        # 4. THỰC HIỆN FINE-TUNE (CẬP NHẬT TRỌNG SỐ) VỚI EARLY STOPPING
    # Vì model là MultiOutputRegressor (chứa nhiều model con), ta phải update từng cái
    
    print("⏳ Đang cập nhật kiến thức mới cho mô hình...")
    
    # Duyệt qua từng model con (tương ứng từng cột output: DO, pH, Temp...)
    for i, estimator in enumerate(model.estimators_):
        target_name = features_list[i]
        
        # A. Lấy "bộ não" (booster) của model cũ ra
        old_booster = estimator.get_booster()
        
        # B. Giảm tốc độ học (Learning Rate)
        # Khi fine-tune, ta nên học chậm lại để không "quên" kiến thức cũ quá nhanh
        estimator.set_params(learning_rate=0.005, n_estimators=500, early_stopping_rounds=30)
        
        # C. Train tiếp (Incremental Learning) với Early Stopping
        # Tham số quan trọng nhất: xgb_model=old_booster
        # Nghĩa là: "Đừng học từ đầu, hãy học tiếp từ cái cũ"
        estimator.fit(
            X_train, y_train.iloc[:, i],
            xgb_model=old_booster,
            eval_set=[(X_val, y_val.iloc[:, i])],
            verbose=False
        )
        
    # 5. ĐÁNH GIÁ TRÊN TẬP TRAIN
    print("\n📊 KẾT QUẢ SAU KHI FINE-TUNE (TRAINING):")
    print("-" * 50)
    y_train_pred = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred, multioutput='raw_values'))
    
    for i, col_name in enumerate(features_list):
        print(f"   🔹 {col_name:<15} RMSE(train): {rmse_train[i]:.4f}")
    
    # 6. ĐÁNH GIÁ TRÊN TẬP VALIDATION (OUT-OF-SAMPLE)
    print("\n📊 KẾT QUẢ SAU KHI FINE-TUNE (VALIDATION - OUT-OF-SAMPLE):")
    print("-" * 50)
    y_val_pred = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred, multioutput='raw_values'))
    
    for i, col_name in enumerate(features_list):
        print(f"   🔹 {col_name:<15} RMSE(val): {rmse_val[i]:.4f}")
    
    print("-" * 50)
    print(f"👉 RMSE trung bình (train): {np.mean(rmse_train):.4f}")
    print(f"👉 RMSE trung bình (val):   {np.mean(rmse_val):.4f}")
