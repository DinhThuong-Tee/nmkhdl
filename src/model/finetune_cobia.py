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