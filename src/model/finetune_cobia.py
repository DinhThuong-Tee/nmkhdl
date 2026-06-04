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