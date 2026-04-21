"""
=============================================================================
FINE-TUNING EXPERIMENT MODULE
=============================================================================
Mục đích: Thử nghiệm các cấu hình hyperparameter khác nhau cho quá trình
Fine-tune mô hình XGBoost từ dữ liệu Hồng Kông sang dữ liệu Quảng Ninh.

Thành viên: Nguyễn Phương Linh
Task: Tinh chỉnh mô hình (Fine-Tuning / Transfer Learning)

Thử nghiệm bao gồm:
1. Grid Search trên learning_rate cho fine-tuning
2. So sánh số lượng n_estimators khi fine-tune
3. Cross-validation trên dữ liệu Quảng Ninh
4. Phân tích ảnh hưởng của lag features
5. Early stopping strategy
6. Đánh giá catastrophic forgetting
=============================================================================
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
import os
import time
import json
from pathlib import Path
from copy import deepcopy
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, KFold
from itertools import product

from basemodel import (
    prepare_time_series_data,
    handle_outliers,
    clip_percentile,
    OYSTER_FEATURES,
    COBIA_FEATURES
)

warnings.filterwarnings('ignore')


# =============================================================================
# CẤU HÌNH THỬ NGHIỆM
# =============================================================================

EXPERIMENT_CONFIG = {
    "learning_rates": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
    "n_estimators_list": [50, 100, 200, 500, 1000],
    "max_depth_list": [3, 4, 5, 6, 7, 8],
    "subsample_list": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree_list": [0.6, 0.7, 0.8, 0.9, 1.0],
    "lag_configs": [[1], [1, 2], [1, 4], [1, 2, 4], [1, 4, 8]],
    "n_splits_cv": 3,
    "random_state": 42,
}



# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_metrics(y_true, y_pred, feature_names):
    """
    Tính toán đầy đủ các metrics đánh giá cho từng feature.
    
    Parameters:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        feature_names: Danh sách tên các features
        
    Returns:
        dict chứa RMSE, MAE, R2 cho từng feature
    """
    results = {}
    
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true.values
    else:
        y_true_arr = np.array(y_true)
        
    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred.values
    else:
        y_pred_arr = np.array(y_pred)
    
    for i, feat in enumerate(feature_names):
        true_col = y_true_arr[:, i]
        pred_col = y_pred_arr[:, i]
        
        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        mae = mean_absolute_error(true_col, pred_col)
        r2 = r2_score(true_col, pred_col)
        
        results[feat] = {
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 6),
        }
    
    avg_rmse = np.mean([v["rmse"] for v in results.values()])
    avg_mae = np.mean([v["mae"] for v in results.values()])
    avg_r2 = np.mean([v["r2"] for v in results.values()])
    
    results["__average__"] = {
        "rmse": round(avg_rmse, 6),
        "mae": round(avg_mae, 6),
        "r2": round(avg_r2, 6),
    }
    
    return results


def print_metrics_table(metrics, title=""):
    """
    In bảng metrics đẹp ra console.
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Feature':<15} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print(f"  {'-'*51}")
    
    for feat, vals in metrics.items():
        if feat == "__average__":
            continue
        print(f"  {feat:<15} {vals['rmse']:<12.4f} {vals['mae']:<12.4f} {vals['r2']:<12.4f}")
    
    avg = metrics["__average__"]
    print(f"  {'-'*51}")
    print(f"  {'TRUNG BÌNH':<15} {avg['rmse']:<12.4f} {avg['mae']:<12.4f} {avg['r2']:<12.4f}")
    print(f"{'='*60}\n")


def format_time(seconds):
    """Chuyển đổi giây sang định dạng dễ đọc."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def save_experiment_results(results, output_path):
    """
    Lưu kết quả thử nghiệm ra file JSON.
    """
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.floating):
            serializable[key] = float(value)
        elif isinstance(value, np.integer):
            serializable[key] = int(value)
        elif isinstance(value, dict):
            serializable[key] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in value.items()
            }
        else:
            serializable[key] = value
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Đã lưu kết quả tại: {output_path}")



# =============================================================================
# THỬ NGHIỆM 1: GRID SEARCH LEARNING RATE
# =============================================================================

def experiment_learning_rate(base_model_path, data_path, features_list, 
                             learning_rates=None):
    """
    Thử nghiệm các giá trị learning_rate khác nhau khi fine-tune.
    
    Mục đích: Tìm learning_rate tối ưu để cân bằng giữa:
    - Học đủ nhanh để thích nghi với dữ liệu Quảng Ninh
    - Học đủ chậm để không quên kiến thức từ Hồng Kông
    
    Parameters:
        base_model_path: Đường dẫn tới model gốc (đã train trên HK)
        data_path: Đường dẫn tới dữ liệu Quảng Ninh
        features_list: Danh sách features cần dự báo
        learning_rates: List các giá trị lr cần thử
        
    Returns:
        dict chứa kết quả cho từng learning_rate
    """
    if learning_rates is None:
        learning_rates = EXPERIMENT_CONFIG["learning_rates"]
    
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 1: GRID SEARCH LEARNING RATE CHO FINE-TUNING")
    print("="*70)
    
    base_model_path = str(base_model_path)
    
    if not os.path.exists(base_model_path):
        print(f"❌ Không tìm thấy model gốc: {base_model_path}")
        return None
    
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    df_ft, _ = prepare_time_series_data(data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("❌ Dữ liệu fine-tune trống.")
        return None
    
    X_new = df_ft[input_cols]
    y_new = df_ft[features_list]
    
    all_results = {}
    best_lr = None
    best_rmse = float('inf')
    
    for lr in learning_rates:
        print(f"\n  🔄 Đang thử learning_rate = {lr}...")
        start_time = time.time()
        
        model_copy = deepcopy(base_model)
        
        for i, estimator in enumerate(model_copy.estimators_):
            old_booster = estimator.get_booster()
            estimator.set_params(learning_rate=lr)
            estimator.fit(X_new, y_new.iloc[:, i], xgb_model=old_booster)
        
        y_pred = model_copy.predict(X_new)
        metrics = calculate_metrics(y_new, y_pred, features_list)
        
        elapsed = time.time() - start_time
        avg_rmse = metrics["__average__"]["rmse"]
        
        all_results[str(lr)] = {
            "metrics": metrics,
            "time_seconds": round(elapsed, 2),
            "avg_rmse": avg_rmse,
            "avg_r2": metrics["__average__"]["r2"],
        }
        
        print(f"     ✅ RMSE trung bình: {avg_rmse:.4f} | R²: {metrics['__average__']['r2']:.4f} | Thời gian: {format_time(elapsed)}")
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_lr = lr
    
    print(f"\n  {'='*50}")
    print(f"  🏆 LEARNING RATE TỐT NHẤT: {best_lr}")
    print(f"     RMSE: {best_rmse:.4f}")
    print(f"  {'='*50}")
    
    all_results["best_learning_rate"] = best_lr
    all_results["best_rmse"] = best_rmse
    
    return all_results



# =============================================================================
# THỬ NGHIỆM 2: SO SÁNH SỐ LƯỢNG N_ESTIMATORS
# =============================================================================

def experiment_n_estimators(base_model_path, data_path, features_list,
                            n_estimators_list=None, fixed_lr=0.005):
    """
    Thử nghiệm số lượng boosting rounds (n_estimators) khi fine-tune.
    
    Trong Transfer Learning với XGBoost, n_estimators quyết định:
    - Bao nhiêu cây mới được thêm vào trên nền tảng cũ
    - Quá ít: model chưa kịp học đặc trưng mới
    - Quá nhiều: model có thể overfit trên dữ liệu nhỏ của QN
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        data_path: Đường dẫn dữ liệu QN
        features_list: Danh sách features
        n_estimators_list: Các giá trị n_estimators cần thử
        fixed_lr: Learning rate cố định (dùng giá trị tốt nhất từ exp 1)
        
    Returns:
        dict kết quả
    """
    if n_estimators_list is None:
        n_estimators_list = EXPERIMENT_CONFIG["n_estimators_list"]
    
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 2: SO SÁNH SỐ LƯỢNG N_ESTIMATORS KHI FINE-TUNE")
    print(f"  (Learning rate cố định: {fixed_lr})")
    print("="*70)
    
    base_model_path = str(base_model_path)
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    df_ft, _ = prepare_time_series_data(data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("❌ Dữ liệu fine-tune trống.")
        return None
    
    X_new = df_ft[input_cols]
    y_new = df_ft[features_list]
    
    all_results = {}
    best_n = None
    best_rmse = float('inf')
    
    for n_est in n_estimators_list:
        print(f"\n  🔄 Đang thử n_estimators = {n_est}...")
        start_time = time.time()
        
        model_copy = deepcopy(base_model)
        
        for i, estimator in enumerate(model_copy.estimators_):
            old_booster = estimator.get_booster()
            estimator.set_params(
                learning_rate=fixed_lr,
                n_estimators=n_est
            )
            estimator.fit(X_new, y_new.iloc[:, i], xgb_model=old_booster)
        
        y_pred = model_copy.predict(X_new)
        metrics = calculate_metrics(y_new, y_pred, features_list)
        
        elapsed = time.time() - start_time
        avg_rmse = metrics["__average__"]["rmse"]
        
        all_results[str(n_est)] = {
            "metrics": metrics,
            "time_seconds": round(elapsed, 2),
            "avg_rmse": avg_rmse,
            "avg_r2": metrics["__average__"]["r2"],
        }
        
        print(f"     ✅ RMSE: {avg_rmse:.4f} | R²: {metrics['__average__']['r2']:.4f} | Thời gian: {format_time(elapsed)}")
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_n = n_est
    
    print(f"\n  {'='*50}")
    print(f"  🏆 N_ESTIMATORS TỐT NHẤT: {best_n}")
    print(f"     RMSE: {best_rmse:.4f}")
    print(f"  {'='*50}")
    
    all_results["best_n_estimators"] = best_n
    all_results["best_rmse"] = best_rmse
    
    return all_results



# =============================================================================
# THỬ NGHIỆM 3: CROSS-VALIDATION TRÊN DỮ LIỆU QUẢNG NINH
# =============================================================================

def experiment_cross_validation(base_model_path, data_path, features_list,
                                n_splits=None, fixed_lr=0.005):
    """
    Đánh giá fine-tuning bằng Time Series Cross-Validation.
    
    Vì dữ liệu là chuỗi thời gian, ta không thể dùng KFold thông thường
    (sẽ bị data leakage). Thay vào đó dùng TimeSeriesSplit:
    - Fold 1: Train [0:n], Test [n:n+k]
    - Fold 2: Train [0:n+k], Test [n+k:n+2k]
    - ...
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        data_path: Đường dẫn dữ liệu
        features_list: Danh sách features
        n_splits: Số fold
        fixed_lr: Learning rate
        
    Returns:
        dict kết quả cross-validation
    """
    if n_splits is None:
        n_splits = EXPERIMENT_CONFIG["n_splits_cv"]
    
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 3: TIME SERIES CROSS-VALIDATION")
    print(f"  (Số fold: {n_splits} | Learning rate: {fixed_lr})")
    print("="*70)
    
    base_model_path = str(base_model_path)
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    df_ft, _ = prepare_time_series_data(data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("❌ Dữ liệu trống.")
        return None
    
    X = df_ft[input_cols]
    y = df_ft[features_list]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_results = []
    all_rmse_per_fold = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n  📂 Fold {fold_idx + 1}/{n_splits}")
        print(f"     Train size: {len(train_idx)} | Test size: {len(test_idx)}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model_copy = deepcopy(base_model)
        
        start_time = time.time()
        
        for i, estimator in enumerate(model_copy.estimators_):
            old_booster = estimator.get_booster()
            estimator.set_params(learning_rate=fixed_lr)
            estimator.fit(X_train, y_train.iloc[:, i], xgb_model=old_booster)
        
        y_pred_test = model_copy.predict(X_test)
        metrics_test = calculate_metrics(y_test, y_pred_test, features_list)
        
        y_pred_train = model_copy.predict(X_train)
        metrics_train = calculate_metrics(y_train, y_pred_train, features_list)
        
        elapsed = time.time() - start_time
        
        fold_result = {
            "fold": fold_idx + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_metrics": metrics_train,
            "test_metrics": metrics_test,
            "time_seconds": round(elapsed, 2),
            "overfitting_gap": round(
                metrics_test["__average__"]["rmse"] - metrics_train["__average__"]["rmse"], 4
            ),
        }
        
        fold_results.append(fold_result)
        all_rmse_per_fold.append(metrics_test["__average__"]["rmse"])
        
        print(f"     Train RMSE: {metrics_train['__average__']['rmse']:.4f}")
        print(f"     Test  RMSE: {metrics_test['__average__']['rmse']:.4f}")
        print(f"     Overfitting gap: {fold_result['overfitting_gap']:.4f}")
        print(f"     Thời gian: {format_time(elapsed)}")
    
    avg_cv_rmse = np.mean(all_rmse_per_fold)
    std_cv_rmse = np.std(all_rmse_per_fold)
    
    print(f"\n  {'='*50}")
    print(f"  📊 KẾT QUẢ CROSS-VALIDATION TỔNG HỢP:")
    print(f"     RMSE trung bình: {avg_cv_rmse:.4f} ± {std_cv_rmse:.4f}")
    print(f"     Min RMSE: {min(all_rmse_per_fold):.4f}")
    print(f"     Max RMSE: {max(all_rmse_per_fold):.4f}")
    print(f"  {'='*50}")
    
    return {
        "folds": fold_results,
        "avg_rmse": round(avg_cv_rmse, 6),
        "std_rmse": round(std_cv_rmse, 6),
        "min_rmse": round(min(all_rmse_per_fold), 6),
        "max_rmse": round(max(all_rmse_per_fold), 6),
    }



# =============================================================================
# THỬ NGHIỆM 4: PHÂN TÍCH ẢNH HƯỞNG CỦA LAG FEATURES
# =============================================================================

def experiment_lag_features(base_model_path, data_path, features_list,
                            lag_configs=None, fixed_lr=0.005):
    """
    So sánh hiệu quả fine-tuning với các cấu hình lag khác nhau.
    
    Lag features là các giá trị quá khứ được dùng làm input:
    - lag1: Giá trị quý trước
    - lag4: Giá trị cùng quý năm trước (seasonality)
    - lag2: Giá trị 2 quý trước
    
    Câu hỏi: Cấu hình lag nào cho kết quả fine-tune tốt nhất?
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        data_path: Đường dẫn dữ liệu
        features_list: Danh sách features
        lag_configs: List các cấu hình lag cần thử
        fixed_lr: Learning rate cố định
        
    Returns:
        dict kết quả
    """
    if lag_configs is None:
        lag_configs = EXPERIMENT_CONFIG["lag_configs"]
    
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 4: PHÂN TÍCH ẢNH HƯỞNG CỦA LAG FEATURES")
    print("="*70)
    
    base_model_path = str(base_model_path)
    
    all_results = {}
    best_lag = None
    best_rmse = float('inf')
    
    for lag_config in lag_configs:
        lag_str = str(lag_config)
        print(f"\n  🔄 Đang thử lag config: {lag_str}...")
        
        start_time = time.time()
        
        df_ft, input_cols_new = prepare_time_series_data(
            data_path, features_list, lags=lag_config
        )
        
        if df_ft is None or len(df_ft) == 0:
            print(f"     ⚠️ Không đủ dữ liệu với lag={lag_str}. Bỏ qua.")
            all_results[lag_str] = {"error": "insufficient_data"}
            continue
        
        base_model = joblib.load(base_model_path)
        meta_path = base_model_path.replace('.pkl', '_features.pkl')
        input_cols_old, _ = joblib.load(meta_path)
        
        common_cols = [c for c in input_cols_old if c in df_ft.columns]
        
        if len(common_cols) < len(input_cols_old) * 0.5:
            print(f"     ⚠️ Quá ít cột chung ({len(common_cols)}/{len(input_cols_old)}). Bỏ qua.")
            all_results[lag_str] = {"error": "column_mismatch"}
            continue
        
        missing_cols = [c for c in input_cols_old if c not in df_ft.columns]
        for col in missing_cols:
            df_ft[col] = 0.0
        
        X_new = df_ft[input_cols_old]
        y_new = df_ft[features_list]
        
        model_copy = deepcopy(base_model)
        
        for i, estimator in enumerate(model_copy.estimators_):
            old_booster = estimator.get_booster()
            estimator.set_params(learning_rate=fixed_lr)
            estimator.fit(X_new, y_new.iloc[:, i], xgb_model=old_booster)
        
        y_pred = model_copy.predict(X_new)
        metrics = calculate_metrics(y_new, y_pred, features_list)
        
        elapsed = time.time() - start_time
        avg_rmse = metrics["__average__"]["rmse"]
        
        all_results[lag_str] = {
            "metrics": metrics,
            "n_samples": len(X_new),
            "n_features": len(input_cols_old),
            "n_common_cols": len(common_cols),
            "n_missing_cols": len(missing_cols),
            "time_seconds": round(elapsed, 2),
            "avg_rmse": avg_rmse,
        }
        
        print(f"     ✅ Samples: {len(X_new)} | RMSE: {avg_rmse:.4f} | Thời gian: {format_time(elapsed)}")
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_lag = lag_config
    
    print(f"\n  {'='*50}")
    print(f"  🏆 LAG CONFIG TỐT NHẤT: {best_lag}")
    print(f"     RMSE: {best_rmse:.4f}")
    print(f"  {'='*50}")
    
    all_results["best_lag_config"] = best_lag
    all_results["best_rmse"] = best_rmse
    
    return all_results



# =============================================================================
# THỬ NGHIỆM 5: EARLY STOPPING STRATEGY
# =============================================================================

def experiment_early_stopping(base_model_path, data_path, features_list,
                              fixed_lr=0.005, max_rounds=2000,
                              early_stopping_rounds_list=None):
    """
    Thử nghiệm Early Stopping khi fine-tune.
    
    Early stopping giúp tự động dừng training khi model không cải thiện nữa,
    tránh overfitting trên tập dữ liệu nhỏ của Quảng Ninh.
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        data_path: Đường dẫn dữ liệu
        features_list: Danh sách features
        fixed_lr: Learning rate
        max_rounds: Số rounds tối đa
        early_stopping_rounds_list: Các giá trị patience cần thử
        
    Returns:
        dict kết quả
    """
    if early_stopping_rounds_list is None:
        early_stopping_rounds_list = [5, 10, 20, 50, 100]
    
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 5: EARLY STOPPING STRATEGY")
    print(f"  (Max rounds: {max_rounds} | LR: {fixed_lr})")
    print("="*70)
    
    base_model_path = str(base_model_path)
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    df_ft, _ = prepare_time_series_data(data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("❌ Dữ liệu trống.")
        return None
    
    X = df_ft[input_cols]
    y = df_ft[features_list]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  Train size: {len(X_train)} | Validation size: {len(X_val)}")
    
    all_results = {}
    best_patience = None
    best_val_rmse = float('inf')
    
    for patience in early_stopping_rounds_list:
        print(f"\n  🔄 Early stopping patience = {patience}...")
        start_time = time.time()
        
        model_copy = deepcopy(base_model)
        stopped_rounds = []
        
        for i, estimator in enumerate(model_copy.estimators_):
            old_booster = estimator.get_booster()
            
            estimator.set_params(
                learning_rate=fixed_lr,
                n_estimators=max_rounds,
                early_stopping_rounds=patience,
            )
            
            estimator.fit(
                X_train, y_train.iloc[:, i],
                xgb_model=old_booster,
                eval_set=[(X_val, y_val.iloc[:, i])],
                verbose=False
            )
            
            best_iter = getattr(estimator, 'best_iteration', max_rounds)
            stopped_rounds.append(best_iter)
        
        y_pred_val = model_copy.predict(X_val)
        metrics_val = calculate_metrics(y_val, y_pred_val, features_list)
        
        y_pred_train = model_copy.predict(X_train)
        metrics_train = calculate_metrics(y_train, y_pred_train, features_list)
        
        elapsed = time.time() - start_time
        avg_val_rmse = metrics_val["__average__"]["rmse"]
        avg_stopped = np.mean(stopped_rounds)
        
        all_results[str(patience)] = {
            "val_metrics": metrics_val,
            "train_metrics": metrics_train,
            "avg_stopped_round": round(avg_stopped, 1),
            "min_stopped_round": min(stopped_rounds),
            "max_stopped_round": max(stopped_rounds),
            "time_seconds": round(elapsed, 2),
            "avg_val_rmse": avg_val_rmse,
            "overfitting_gap": round(
                avg_val_rmse - metrics_train["__average__"]["rmse"], 4
            ),
        }
        
        print(f"     ✅ Val RMSE: {avg_val_rmse:.4f} | Avg stopped: {avg_stopped:.0f} rounds")
        print(f"     Overfitting gap: {all_results[str(patience)]['overfitting_gap']:.4f}")
        
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            best_patience = patience
    
    print(f"\n  {'='*50}")
    print(f"  🏆 PATIENCE TỐT NHẤT: {best_patience}")
    print(f"     Val RMSE: {best_val_rmse:.4f}")
    print(f"  {'='*50}")
    
    all_results["best_patience"] = best_patience
    all_results["best_val_rmse"] = best_val_rmse
    
    return all_results



# =============================================================================
# THỬ NGHIỆM 6: ĐÁNH GIÁ CATASTROPHIC FORGETTING
# =============================================================================

def experiment_catastrophic_forgetting(base_model_path, hk_data_path, 
                                        qn_data_path, features_list,
                                        learning_rates=None):
    """
    Đánh giá mức độ "quên" kiến thức cũ (catastrophic forgetting).
    
    Sau khi fine-tune trên dữ liệu QN, model có thể "quên" cách dự báo
    trên dữ liệu HK. Thử nghiệm này đo lường:
    - Performance trên HK trước fine-tune
    - Performance trên HK sau fine-tune
    - Mức suy giảm (forgetting rate)
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        hk_data_path: Đường dẫn dữ liệu Hồng Kông (để test forgetting)
        qn_data_path: Đường dẫn dữ liệu Quảng Ninh (để fine-tune)
        features_list: Danh sách features
        learning_rates: Các LR cần thử
        
    Returns:
        dict kết quả phân tích forgetting
    """
    if learning_rates is None:
        learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 6: ĐÁNH GIÁ CATASTROPHIC FORGETTING")
    print("="*70)
    
    base_model_path = str(base_model_path)
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    print("\n  📊 Chuẩn bị dữ liệu Hồng Kông (để đo forgetting)...")
    df_hk, _ = prepare_time_series_data(hk_data_path, features_list, lags=[1, 4])
    if df_hk is None or len(df_hk) == 0:
        print("❌ Dữ liệu HK trống.")
        return None
    
    X_hk = df_hk[input_cols]
    y_hk = df_hk[features_list]
    
    y_pred_hk_before = base_model.predict(X_hk)
    metrics_hk_before = calculate_metrics(y_hk, y_pred_hk_before, features_list)
    
    print(f"  ✅ Performance trên HK (TRƯỚC fine-tune): RMSE = {metrics_hk_before['__average__']['rmse']:.4f}")
    
    print("\n  📊 Chuẩn bị dữ liệu Quảng Ninh (để fine-tune)...")
    df_qn, _ = prepare_time_series_data(qn_data_path, features_list, lags=[1, 4])
    if df_qn is None or len(df_qn) == 0:
        print("❌ Dữ liệu QN trống.")
        return None
    
    X_qn = df_qn[input_cols]
    y_qn = df_qn[features_list]
    
    all_results = {
        "before_finetune_hk_rmse": metrics_hk_before["__average__"]["rmse"],
        "before_finetune_hk_metrics": metrics_hk_before,
    }
    
    for lr in learning_rates:
        print(f"\n  🔄 Fine-tune với LR={lr}, sau đó test lại trên HK...")
        
        model_copy = deepcopy(base_model)
        
        for i, estimator in enumerate(model_copy.estimators_):
            old_booster = estimator.get_booster()
            estimator.set_params(learning_rate=lr)
            estimator.fit(X_qn, y_qn.iloc[:, i], xgb_model=old_booster)
        
        y_pred_hk_after = model_copy.predict(X_hk)
        metrics_hk_after = calculate_metrics(y_hk, y_pred_hk_after, features_list)
        
        y_pred_qn = model_copy.predict(X_qn)
        metrics_qn = calculate_metrics(y_qn, y_pred_qn, features_list)
        
        rmse_before = metrics_hk_before["__average__"]["rmse"]
        rmse_after = metrics_hk_after["__average__"]["rmse"]
        forgetting_rate = (rmse_after - rmse_before) / rmse_before * 100
        
        result_lr = {
            "hk_rmse_after": rmse_after,
            "qn_rmse": metrics_qn["__average__"]["rmse"],
            "forgetting_rate_percent": round(forgetting_rate, 2),
            "hk_metrics_after": metrics_hk_after,
            "qn_metrics": metrics_qn,
        }
        
        all_results[f"lr_{lr}"] = result_lr
        
        print(f"     HK RMSE sau fine-tune: {rmse_after:.4f}")
        print(f"     QN RMSE: {metrics_qn['__average__']['rmse']:.4f}")
        print(f"     Forgetting rate: {forgetting_rate:+.2f}%")
        
        if forgetting_rate > 20:
            print(f"     ⚠️ CẢNH BÁO: Mức forgetting cao ({forgetting_rate:.1f}%)")
        elif forgetting_rate > 10:
            print(f"     ⚡ Mức forgetting trung bình ({forgetting_rate:.1f}%)")
        else:
            print(f"     ✅ Mức forgetting chấp nhận được ({forgetting_rate:.1f}%)")
    
    print(f"\n  {'='*50}")
    print(f"  📊 TỔNG KẾT CATASTROPHIC FORGETTING:")
    print(f"  {'='*50}")
    
    for lr in learning_rates:
        key = f"lr_{lr}"
        if key in all_results:
            r = all_results[key]
            status = "✅" if r["forgetting_rate_percent"] < 10 else "⚠️"
            print(f"  {status} LR={lr:<6} | HK RMSE: {r['hk_rmse_after']:.4f} | QN RMSE: {r['qn_rmse']:.4f} | Forget: {r['forgetting_rate_percent']:+.1f}%")
    
    return all_results



# =============================================================================
# THỬ NGHIỆM 7: FULL HYPERPARAMETER GRID SEARCH
# =============================================================================

def experiment_full_grid_search(base_model_path, data_path, features_list,
                                 param_grid=None, top_k=5):
    """
    Grid search đầy đủ trên nhiều hyperparameter cùng lúc.
    
    Kết hợp: learning_rate x max_depth x subsample x colsample_bytree
    để tìm bộ tham số tối ưu cho fine-tuning.
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        data_path: Đường dẫn dữ liệu
        features_list: Danh sách features
        param_grid: Dict chứa các tham số cần search
        top_k: Số lượng kết quả tốt nhất cần hiển thị
        
    Returns:
        dict kết quả grid search
    """
    if param_grid is None:
        param_grid = {
            "learning_rate": [0.001, 0.005, 0.01, 0.05],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        }
    
    all_combinations = list(product(
        param_grid["learning_rate"],
        param_grid["max_depth"],
        param_grid["subsample"],
        param_grid["colsample_bytree"],
    ))
    
    total_combos = len(all_combinations)
    
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 7: FULL HYPERPARAMETER GRID SEARCH")
    print(f"  Tổng số tổ hợp: {total_combos}")
    print("="*70)
    
    base_model_path = str(base_model_path)
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    df_ft, _ = prepare_time_series_data(data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("❌ Dữ liệu trống.")
        return None
    
    X = df_ft[input_cols]
    y = df_ft[features_list]
    
    split_idx = int(len(X) * 0.75)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")
    
    results_list = []
    overall_start = time.time()
    
    for idx, (lr, depth, subsample, colsample) in enumerate(all_combinations):
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed_total = time.time() - overall_start
            eta = (elapsed_total / (idx + 1)) * (total_combos - idx - 1) if idx > 0 else 0
            print(f"  [{idx+1}/{total_combos}] ETA: {format_time(eta)}")
        
        model_copy = deepcopy(base_model)
        
        for i, estimator in enumerate(model_copy.estimators_):
            old_booster = estimator.get_booster()
            estimator.set_params(
                learning_rate=lr,
                max_depth=depth,
                subsample=subsample,
                colsample_bytree=colsample,
            )
            estimator.fit(X_train, y_train.iloc[:, i], xgb_model=old_booster)
        
        y_pred_val = model_copy.predict(X_val)
        metrics_val = calculate_metrics(y_val, y_pred_val, features_list)
        
        results_list.append({
            "params": {
                "learning_rate": lr,
                "max_depth": depth,
                "subsample": subsample,
                "colsample_bytree": colsample,
            },
            "val_rmse": metrics_val["__average__"]["rmse"],
            "val_r2": metrics_val["__average__"]["r2"],
        })
    
    total_time = time.time() - overall_start
    
    results_list.sort(key=lambda x: x["val_rmse"])
    
    print(f"\n  {'='*60}")
    print(f"  🏆 TOP {top_k} BỘ THAM SỐ TỐT NHẤT:")
    print(f"  {'='*60}")
    
    for rank, result in enumerate(results_list[:top_k], 1):
        p = result["params"]
        print(f"  #{rank}: RMSE={result['val_rmse']:.4f} | R²={result['val_r2']:.4f}")
        print(f"       lr={p['learning_rate']}, depth={p['max_depth']}, "
              f"subsample={p['subsample']}, colsample={p['colsample_bytree']}")
    
    print(f"\n  ⏱️ Tổng thời gian: {format_time(total_time)}")
    
    return {
        "total_combinations": total_combos,
        "total_time_seconds": round(total_time, 2),
        "top_results": results_list[:top_k],
        "best_params": results_list[0]["params"],
        "best_val_rmse": results_list[0]["val_rmse"],
    }



# =============================================================================
# THỬ NGHIỆM 8: SO SÁNH FINE-TUNE VS TRAIN TỪ ĐẦU
# =============================================================================

def experiment_finetune_vs_scratch(base_model_path, data_path, features_list,
                                    fixed_lr=0.005):
    """
    So sánh hiệu quả giữa:
    A) Fine-tune từ model HK (Transfer Learning)
    B) Train model mới hoàn toàn trên dữ liệu QN (From Scratch)
    
    Mục đích: Chứng minh Transfer Learning thực sự có lợi khi dữ liệu
    target domain (QN) ít.
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        data_path: Đường dẫn dữ liệu QN
        features_list: Danh sách features
        fixed_lr: Learning rate cho fine-tuning
        
    Returns:
        dict so sánh kết quả
    """
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 8: FINE-TUNE VS TRAIN TỪ ĐẦU")
    print("="*70)
    
    base_model_path = str(base_model_path)
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    df_ft, _ = prepare_time_series_data(data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("❌ Dữ liệu trống.")
        return None
    
    X = df_ft[input_cols]
    y = df_ft[features_list]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    
    # --- PHƯƠNG PHÁP A: FINE-TUNE ---
    print("\n  🅰️  PHƯƠNG PHÁP A: FINE-TUNE TỪ MODEL HK")
    start_a = time.time()
    
    model_ft = deepcopy(base_model)
    for i, estimator in enumerate(model_ft.estimators_):
        old_booster = estimator.get_booster()
        estimator.set_params(learning_rate=fixed_lr)
        estimator.fit(X_train, y_train.iloc[:, i], xgb_model=old_booster)
    
    y_pred_ft = model_ft.predict(X_test)
    metrics_ft = calculate_metrics(y_test, y_pred_ft, features_list)
    time_ft = time.time() - start_a
    
    print(f"     RMSE: {metrics_ft['__average__']['rmse']:.4f}")
    print(f"     R²:   {metrics_ft['__average__']['r2']:.4f}")
    print(f"     Thời gian: {format_time(time_ft)}")
    
    # --- PHƯƠNG PHÁP B: TRAIN TỪ ĐẦU ---
    print("\n  🅱️  PHƯƠNG PHÁP B: TRAIN TỪ ĐẦU (FROM SCRATCH)")
    start_b = time.time()
    
    model_scratch = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    ))
    
    model_scratch.fit(X_train, y_train)
    
    y_pred_scratch = model_scratch.predict(X_test)
    metrics_scratch = calculate_metrics(y_test, y_pred_scratch, features_list)
    time_scratch = time.time() - start_b
    
    print(f"     RMSE: {metrics_scratch['__average__']['rmse']:.4f}")
    print(f"     R²:   {metrics_scratch['__average__']['r2']:.4f}")
    print(f"     Thời gian: {format_time(time_scratch)}")
    
    # --- SO SÁNH ---
    rmse_ft = metrics_ft["__average__"]["rmse"]
    rmse_scratch = metrics_scratch["__average__"]["rmse"]
    improvement = (rmse_scratch - rmse_ft) / rmse_scratch * 100
    
    print(f"\n  {'='*50}")
    print(f"  📊 KẾT QUẢ SO SÁNH:")
    print(f"  {'='*50}")
    print(f"  Fine-tune RMSE:    {rmse_ft:.4f}")
    print(f"  From Scratch RMSE: {rmse_scratch:.4f}")
    print(f"  Cải thiện:         {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"  ✅ Transfer Learning TỐT HƠN {improvement:.1f}%")
    else:
        print(f"  ⚠️ Train từ đầu tốt hơn (có thể do domain gap quá lớn)")
    
    print(f"  {'='*50}")
    
    # So sánh từng feature
    print(f"\n  📋 CHI TIẾT TỪNG FEATURE:")
    print(f"  {'Feature':<15} {'Fine-tune':<12} {'Scratch':<12} {'Winner':<10}")
    print(f"  {'-'*49}")
    
    for feat in features_list:
        ft_rmse = metrics_ft[feat]["rmse"]
        sc_rmse = metrics_scratch[feat]["rmse"]
        winner = "FT ✅" if ft_rmse < sc_rmse else "Scratch"
        print(f"  {feat:<15} {ft_rmse:<12.4f} {sc_rmse:<12.4f} {winner}")
    
    return {
        "finetune": {
            "metrics": metrics_ft,
            "time_seconds": round(time_ft, 2),
        },
        "from_scratch": {
            "metrics": metrics_scratch,
            "time_seconds": round(time_scratch, 2),
        },
        "improvement_percent": round(improvement, 2),
        "winner": "finetune" if improvement > 0 else "from_scratch",
    }



# =============================================================================
# THỬ NGHIỆM 9: FEATURE IMPORTANCE ANALYSIS SAU FINE-TUNE
# =============================================================================

def experiment_feature_importance(base_model_path, data_path, features_list,
                                   fixed_lr=0.005):
    """
    Phân tích Feature Importance trước và sau fine-tune.
    
    So sánh xem features nào quan trọng nhất khi model chuyển từ
    domain HK sang domain QN. Sự thay đổi importance cho thấy
    model đã học được gì mới từ dữ liệu QN.
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        data_path: Đường dẫn dữ liệu QN
        features_list: Danh sách features
        fixed_lr: Learning rate
        
    Returns:
        dict chứa feature importance trước/sau
    """
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 9: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    base_model_path = str(base_model_path)
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    # Feature importance TRƯỚC fine-tune
    print("\n  📊 Feature Importance TRƯỚC fine-tune:")
    importance_before = {}
    for i, estimator in enumerate(base_model.estimators_):
        booster = estimator.get_booster()
        scores = booster.get_score(importance_type='gain')
        for feat, score in scores.items():
            if feat not in importance_before:
                importance_before[feat] = 0
            importance_before[feat] += score
    
    # Normalize
    total_before = sum(importance_before.values()) if importance_before else 1
    importance_before = {k: v/total_before for k, v in importance_before.items()}
    
    # Sort và hiển thị top 10
    sorted_before = sorted(importance_before.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 10 features (trước fine-tune):")
    for feat, score in sorted_before[:10]:
        bar = "█" * int(score * 50)
        print(f"    {feat:<25} {score:.4f} {bar}")
    
    # Fine-tune
    df_ft, _ = prepare_time_series_data(data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("❌ Dữ liệu trống.")
        return None
    
    X_new = df_ft[input_cols]
    y_new = df_ft[features_list]
    
    model_ft = deepcopy(base_model)
    for i, estimator in enumerate(model_ft.estimators_):
        old_booster = estimator.get_booster()
        estimator.set_params(learning_rate=fixed_lr)
        estimator.fit(X_new, y_new.iloc[:, i], xgb_model=old_booster)
    
    # Feature importance SAU fine-tune
    print(f"\n  📊 Feature Importance SAU fine-tune:")
    importance_after = {}
    for i, estimator in enumerate(model_ft.estimators_):
        booster = estimator.get_booster()
        scores = booster.get_score(importance_type='gain')
        for feat, score in scores.items():
            if feat not in importance_after:
                importance_after[feat] = 0
            importance_after[feat] += score
    
    total_after = sum(importance_after.values()) if importance_after else 1
    importance_after = {k: v/total_after for k, v in importance_after.items()}
    
    sorted_after = sorted(importance_after.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 10 features (sau fine-tune):")
    for feat, score in sorted_after[:10]:
        bar = "█" * int(score * 50)
        print(f"    {feat:<25} {score:.4f} {bar}")
    
    # So sánh thay đổi
    print(f"\n  📊 THAY ĐỔI IMPORTANCE LỚN NHẤT:")
    print(f"  {'Feature':<25} {'Trước':<10} {'Sau':<10} {'Thay đổi':<10}")
    print(f"  {'-'*55}")
    
    all_features_imp = set(list(importance_before.keys()) + list(importance_after.keys()))
    changes = []
    for feat in all_features_imp:
        before = importance_before.get(feat, 0)
        after = importance_after.get(feat, 0)
        change = after - before
        changes.append((feat, before, after, change))
    
    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    
    for feat, before, after, change in changes[:15]:
        direction = "↑" if change > 0 else "↓"
        print(f"  {feat:<25} {before:<10.4f} {after:<10.4f} {direction} {abs(change):.4f}")
    
    return {
        "importance_before": dict(sorted_before[:20]),
        "importance_after": dict(sorted_after[:20]),
        "top_changes": [(f, b, a, c) for f, b, a, c in changes[:15]],
    }



# =============================================================================
# THỬ NGHIỆM 10: PROGRESSIVE FINE-TUNING (LAYER-WISE)
# =============================================================================

def experiment_progressive_finetune(base_model_path, data_path, features_list,
                                     lr_schedule=None):
    """
    Thử nghiệm Progressive Fine-tuning: Giảm dần learning rate qua các epoch.
    
    Ý tưởng: Thay vì dùng 1 LR cố định, ta giảm dần LR qua nhiều vòng:
    - Vòng 1: LR cao hơn để nhanh chóng thích nghi
    - Vòng 2: LR trung bình để tinh chỉnh
    - Vòng 3: LR thấp để polish
    
    Tương tự learning rate scheduling trong Deep Learning.
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        data_path: Đường dẫn dữ liệu
        features_list: Danh sách features
        lr_schedule: List các (learning_rate, n_estimators) cho mỗi phase
        
    Returns:
        dict kết quả
    """
    if lr_schedule is None:
        lr_schedule = [
            (0.05, 100),    # Phase 1: Warm-up nhanh
            (0.01, 200),    # Phase 2: Tinh chỉnh
            (0.005, 300),   # Phase 3: Polish
            (0.001, 500),   # Phase 4: Final refinement
        ]
    
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 10: PROGRESSIVE FINE-TUNING")
    print("="*70)
    print(f"  Schedule: {lr_schedule}")
    
    base_model_path = str(base_model_path)
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    df_ft, _ = prepare_time_series_data(data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("❌ Dữ liệu trống.")
        return None
    
    X = df_ft[input_cols]
    y = df_ft[features_list]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model_progressive = deepcopy(base_model)
    phase_results = []
    
    for phase_idx, (lr, n_est) in enumerate(lr_schedule):
        print(f"\n  📍 Phase {phase_idx + 1}: LR={lr}, n_estimators={n_est}")
        start_time = time.time()
        
        for i, estimator in enumerate(model_progressive.estimators_):
            old_booster = estimator.get_booster()
            estimator.set_params(
                learning_rate=lr,
                n_estimators=n_est,
            )
            estimator.fit(X_train, y_train.iloc[:, i], xgb_model=old_booster)
        
        y_pred_val = model_progressive.predict(X_val)
        metrics_val = calculate_metrics(y_val, y_pred_val, features_list)
        
        y_pred_train = model_progressive.predict(X_train)
        metrics_train = calculate_metrics(y_train, y_pred_train, features_list)
        
        elapsed = time.time() - start_time
        
        phase_result = {
            "phase": phase_idx + 1,
            "learning_rate": lr,
            "n_estimators": n_est,
            "val_rmse": metrics_val["__average__"]["rmse"],
            "train_rmse": metrics_train["__average__"]["rmse"],
            "val_r2": metrics_val["__average__"]["r2"],
            "time_seconds": round(elapsed, 2),
        }
        phase_results.append(phase_result)
        
        print(f"     Train RMSE: {metrics_train['__average__']['rmse']:.4f}")
        print(f"     Val RMSE:   {metrics_val['__average__']['rmse']:.4f}")
        print(f"     Val R²:     {metrics_val['__average__']['r2']:.4f}")
        print(f"     Thời gian:  {format_time(elapsed)}")
    
    # So sánh với single-phase fine-tune
    print(f"\n  📊 SO SÁNH VỚI SINGLE-PHASE FINE-TUNE:")
    
    model_single = deepcopy(base_model)
    for i, estimator in enumerate(model_single.estimators_):
        old_booster = estimator.get_booster()
        estimator.set_params(learning_rate=0.005)
        estimator.fit(X_train, y_train.iloc[:, i], xgb_model=old_booster)
    
    y_pred_single = model_single.predict(X_val)
    metrics_single = calculate_metrics(y_val, y_pred_single, features_list)
    
    progressive_rmse = phase_results[-1]["val_rmse"]
    single_rmse = metrics_single["__average__"]["rmse"]
    
    print(f"  Progressive RMSE: {progressive_rmse:.4f}")
    print(f"  Single-phase RMSE: {single_rmse:.4f}")
    
    improvement = (single_rmse - progressive_rmse) / single_rmse * 100
    if improvement > 0:
        print(f"  ✅ Progressive tốt hơn {improvement:.2f}%")
    else:
        print(f"  ⚠️ Single-phase tốt hơn {abs(improvement):.2f}%")
    
    return {
        "phases": phase_results,
        "final_val_rmse": progressive_rmse,
        "single_phase_val_rmse": single_rmse,
        "improvement_percent": round(improvement, 2),
    }



# =============================================================================
# THỬ NGHIỆM 11: DATA AUGMENTATION CHO FINE-TUNING
# =============================================================================

def augment_time_series(X, y, noise_level=0.02, n_augmented=2):
    """
    Tăng cường dữ liệu bằng cách thêm nhiễu Gaussian.
    
    Khi dữ liệu QN ít, ta có thể tạo thêm mẫu "giả" bằng cách
    thêm nhiễu nhỏ vào dữ liệu gốc. Điều này giúp model:
    - Robust hơn với nhiễu
    - Tránh overfit trên tập nhỏ
    
    Parameters:
        X: Features DataFrame
        y: Target DataFrame
        noise_level: Mức nhiễu (tỷ lệ so với std)
        n_augmented: Số bản sao augmented
        
    Returns:
        X_aug, y_aug: Dữ liệu đã augment
    """
    X_list = [X.copy()]
    y_list = [y.copy()]
    
    for _ in range(n_augmented):
        X_noisy = X.copy()
        y_noisy = y.copy()
        
        for col in X.columns:
            std = X[col].std()
            noise = np.random.normal(0, noise_level * std, size=len(X))
            X_noisy[col] = X[col] + noise
        
        for col in y.columns:
            std = y[col].std()
            noise = np.random.normal(0, noise_level * std * 0.5, size=len(y))
            y_noisy[col] = y[col] + noise
        
        X_list.append(X_noisy)
        y_list.append(y_noisy)
    
    X_aug = pd.concat(X_list, ignore_index=True)
    y_aug = pd.concat(y_list, ignore_index=True)
    
    return X_aug, y_aug


def experiment_data_augmentation(base_model_path, data_path, features_list,
                                  noise_levels=None, n_augmented_list=None,
                                  fixed_lr=0.005):
    """
    Thử nghiệm Data Augmentation cho fine-tuning.
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        data_path: Đường dẫn dữ liệu
        features_list: Danh sách features
        noise_levels: Các mức nhiễu cần thử
        n_augmented_list: Số bản augmented cần thử
        fixed_lr: Learning rate
        
    Returns:
        dict kết quả
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.02, 0.05, 0.1]
    if n_augmented_list is None:
        n_augmented_list = [1, 2, 3, 5]
    
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 11: DATA AUGMENTATION CHO FINE-TUNING")
    print("="*70)
    
    base_model_path = str(base_model_path)
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    df_ft, _ = prepare_time_series_data(data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("❌ Dữ liệu trống.")
        return None
    
    X = df_ft[input_cols]
    y = df_ft[features_list]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Baseline (không augment)
    model_baseline = deepcopy(base_model)
    for i, estimator in enumerate(model_baseline.estimators_):
        old_booster = estimator.get_booster()
        estimator.set_params(learning_rate=fixed_lr)
        estimator.fit(X_train, y_train.iloc[:, i], xgb_model=old_booster)
    
    y_pred_baseline = model_baseline.predict(X_val)
    metrics_baseline = calculate_metrics(y_val, y_pred_baseline, features_list)
    baseline_rmse = metrics_baseline["__average__"]["rmse"]
    
    print(f"  📊 Baseline (không augment): RMSE = {baseline_rmse:.4f}")
    print(f"  Train size gốc: {len(X_train)}")
    
    all_results = {"baseline_rmse": baseline_rmse}
    best_config = None
    best_rmse = baseline_rmse
    
    for noise in noise_levels:
        for n_aug in n_augmented_list:
            config_key = f"noise_{noise}_naug_{n_aug}"
            print(f"\n  🔄 noise={noise}, n_augmented={n_aug}...")
            
            np.random.seed(EXPERIMENT_CONFIG["random_state"])
            X_aug, y_aug = augment_time_series(X_train, y_train, noise, n_aug)
            
            model_aug = deepcopy(base_model)
            for i, estimator in enumerate(model_aug.estimators_):
                old_booster = estimator.get_booster()
                estimator.set_params(learning_rate=fixed_lr)
                estimator.fit(X_aug, y_aug.iloc[:, i], xgb_model=old_booster)
            
            y_pred_aug = model_aug.predict(X_val)
            metrics_aug = calculate_metrics(y_val, y_pred_aug, features_list)
            aug_rmse = metrics_aug["__average__"]["rmse"]
            
            improvement = (baseline_rmse - aug_rmse) / baseline_rmse * 100
            
            all_results[config_key] = {
                "noise_level": noise,
                "n_augmented": n_aug,
                "augmented_train_size": len(X_aug),
                "val_rmse": aug_rmse,
                "improvement_percent": round(improvement, 2),
            }
            
            status = "✅" if improvement > 0 else "❌"
            print(f"     {status} Train size: {len(X_aug)} | RMSE: {aug_rmse:.4f} | Improvement: {improvement:+.2f}%")
            
            if aug_rmse < best_rmse:
                best_rmse = aug_rmse
                best_config = (noise, n_aug)
    
    print(f"\n  {'='*50}")
    if best_config:
        print(f"  🏆 BEST CONFIG: noise={best_config[0]}, n_augmented={best_config[1]}")
        print(f"     RMSE: {best_rmse:.4f} (baseline: {baseline_rmse:.4f})")
    else:
        print(f"  ⚠️ Augmentation không cải thiện kết quả")
    print(f"  {'='*50}")
    
    all_results["best_config"] = best_config
    all_results["best_rmse"] = best_rmse
    
    return all_results



# =============================================================================
# THỬ NGHIỆM 12: ENSEMBLE FINE-TUNED MODELS
# =============================================================================

def experiment_ensemble_finetune(base_model_path, data_path, features_list,
                                  n_models=5, lr_range=(0.001, 0.05)):
    """
    Thử nghiệm Ensemble: Fine-tune nhiều model với LR khác nhau,
    sau đó lấy trung bình dự đoán.
    
    Ý tưởng: Mỗi model fine-tune với LR khác nhau sẽ học được
    các khía cạnh khác nhau. Kết hợp chúng có thể cho kết quả
    robust hơn.
    
    Parameters:
        base_model_path: Đường dẫn model gốc
        data_path: Đường dẫn dữ liệu
        features_list: Danh sách features
        n_models: Số model trong ensemble
        lr_range: Khoảng learning rate
        
    Returns:
        dict kết quả
    """
    print("\n" + "="*70)
    print("  THỬ NGHIỆM 12: ENSEMBLE FINE-TUNED MODELS")
    print(f"  (Số model: {n_models} | LR range: {lr_range})")
    print("="*70)
    
    base_model_path = str(base_model_path)
    base_model = joblib.load(base_model_path)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols, _ = joblib.load(meta_path)
    
    df_ft, _ = prepare_time_series_data(data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("❌ Dữ liệu trống.")
        return None
    
    X = df_ft[input_cols]
    y = df_ft[features_list]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    learning_rates = np.linspace(lr_range[0], lr_range[1], n_models)
    
    ensemble_predictions = []
    individual_results = []
    
    for model_idx, lr in enumerate(learning_rates):
        print(f"\n  🔄 Model {model_idx + 1}/{n_models} (LR={lr:.4f})...")
        
        model_copy = deepcopy(base_model)
        
        for i, estimator in enumerate(model_copy.estimators_):
            old_booster = estimator.get_booster()
            estimator.set_params(learning_rate=lr)
            estimator.fit(X_train, y_train.iloc[:, i], xgb_model=old_booster)
        
        y_pred = model_copy.predict(X_val)
        ensemble_predictions.append(y_pred)
        
        metrics = calculate_metrics(y_val, y_pred, features_list)
        individual_results.append({
            "model_idx": model_idx,
            "learning_rate": round(lr, 5),
            "val_rmse": metrics["__average__"]["rmse"],
            "val_r2": metrics["__average__"]["r2"],
        })
        
        print(f"     RMSE: {metrics['__average__']['rmse']:.4f} | R²: {metrics['__average__']['r2']:.4f}")
    
    # Ensemble prediction (simple average)
    ensemble_pred = np.mean(ensemble_predictions, axis=0)
    metrics_ensemble = calculate_metrics(y_val, ensemble_pred, features_list)
    
    # Weighted ensemble (inverse RMSE weighting)
    weights = [1.0 / r["val_rmse"] for r in individual_results]
    weight_sum = sum(weights)
    weights_normalized = [w / weight_sum for w in weights]
    
    weighted_pred = np.zeros_like(ensemble_predictions[0])
    for pred, w in zip(ensemble_predictions, weights_normalized):
        weighted_pred += pred * w
    
    metrics_weighted = calculate_metrics(y_val, weighted_pred, features_list)
    
    # Best individual
    best_individual = min(individual_results, key=lambda x: x["val_rmse"])
    
    print(f"\n  {'='*60}")
    print(f"  📊 KẾT QUẢ ENSEMBLE:")
    print(f"  {'='*60}")
    print(f"  Best Individual RMSE:    {best_individual['val_rmse']:.4f} (LR={best_individual['learning_rate']})")
    print(f"  Simple Average RMSE:     {metrics_ensemble['__average__']['rmse']:.4f}")
    print(f"  Weighted Average RMSE:   {metrics_weighted['__average__']['rmse']:.4f}")
    
    ensemble_rmse = metrics_ensemble["__average__"]["rmse"]
    weighted_rmse = metrics_weighted["__average__"]["rmse"]
    best_ind_rmse = best_individual["val_rmse"]
    
    best_method = "weighted" if weighted_rmse < min(ensemble_rmse, best_ind_rmse) else \
                  "simple_avg" if ensemble_rmse < best_ind_rmse else "individual"
    
    print(f"  🏆 Phương pháp tốt nhất: {best_method}")
    print(f"  {'='*60}")
    
    return {
        "individual_results": individual_results,
        "simple_avg_rmse": metrics_ensemble["__average__"]["rmse"],
        "weighted_avg_rmse": metrics_weighted["__average__"]["rmse"],
        "best_individual_rmse": best_ind_rmse,
        "best_method": best_method,
        "weights": [round(w, 4) for w in weights_normalized],
    }



# =============================================================================
# MAIN: CHẠY TẤT CẢ THỬ NGHIỆM
# =============================================================================

def run_all_experiments(species="cobia"):
    """
    Chạy toàn bộ pipeline thử nghiệm cho một loài.
    
    Parameters:
        species: "cobia" hoặc "oyster"
    """
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    
    MODEL_DIR = PROJECT_DIR / "model" / "output"
    DATA_DIR = PROJECT_DIR / "data"
    
    if species == "cobia":
        base_model_path = MODEL_DIR / "hk_cobia_forecast_model.pkl"
        features_list = COBIA_FEATURES
        hk_data_path = DATA_DIR / "hk_water_quality" / "hk_cobia_quarterly_21vars.csv"
    elif species == "oyster":
        base_model_path = MODEL_DIR / "hk_oyster_forecast_model.pkl"
        features_list = OYSTER_FEATURES
        hk_data_path = DATA_DIR / "hk_water_quality" / "hk_oyster_quarterly_21vars.csv"
    else:
        print(f"❌ Species không hợp lệ: {species}")
        return
    
    qn_data_path = DATA_DIR / "data_quang_ninh" / "qn_env_clean_ready.csv"
    
    print(f"\n{'#'*70}")
    print(f"#  CHẠY TOÀN BỘ THỬ NGHIỆM FINE-TUNING CHO: {species.upper()}")
    print(f"#  Model gốc: {base_model_path.name}")
    print(f"#  Dữ liệu fine-tune: {qn_data_path.name}")
    print(f"{'#'*70}")
    
    all_experiment_results = {}
    total_start = time.time()
    
    # Experiment 1: Learning Rate
    print("\n" + "━"*70)
    result_1 = experiment_learning_rate(base_model_path, qn_data_path, features_list)
    all_experiment_results["exp1_learning_rate"] = result_1
    
    # Experiment 2: N_Estimators
    print("\n" + "━"*70)
    best_lr = result_1.get("best_learning_rate", 0.005) if result_1 else 0.005
    result_2 = experiment_n_estimators(base_model_path, qn_data_path, features_list,
                                       fixed_lr=best_lr)
    all_experiment_results["exp2_n_estimators"] = result_2
    
    # Experiment 3: Cross-Validation
    print("\n" + "━"*70)
    result_3 = experiment_cross_validation(base_model_path, qn_data_path, features_list,
                                           fixed_lr=best_lr)
    all_experiment_results["exp3_cross_validation"] = result_3
    
    # Experiment 4: Lag Features
    print("\n" + "━"*70)
    result_4 = experiment_lag_features(base_model_path, qn_data_path, features_list,
                                       fixed_lr=best_lr)
    all_experiment_results["exp4_lag_features"] = result_4
    
    # Experiment 5: Early Stopping
    print("\n" + "━"*70)
    result_5 = experiment_early_stopping(base_model_path, qn_data_path, features_list,
                                          fixed_lr=best_lr)
    all_experiment_results["exp5_early_stopping"] = result_5
    
    # Experiment 6: Catastrophic Forgetting
    print("\n" + "━"*70)
    result_6 = experiment_catastrophic_forgetting(
        base_model_path, hk_data_path, qn_data_path, features_list
    )
    all_experiment_results["exp6_catastrophic_forgetting"] = result_6
    
    # Experiment 7: Full Grid Search
    print("\n" + "━"*70)
    result_7 = experiment_full_grid_search(base_model_path, qn_data_path, features_list)
    all_experiment_results["exp7_full_grid_search"] = result_7
    
    # Experiment 8: Fine-tune vs Scratch
    print("\n" + "━"*70)
    result_8 = experiment_finetune_vs_scratch(base_model_path, qn_data_path, features_list,
                                              fixed_lr=best_lr)
    all_experiment_results["exp8_finetune_vs_scratch"] = result_8
    
    # Experiment 9: Feature Importance
    print("\n" + "━"*70)
    result_9 = experiment_feature_importance(base_model_path, qn_data_path, features_list,
                                             fixed_lr=best_lr)
    all_experiment_results["exp9_feature_importance"] = result_9
    
    # Experiment 10: Progressive Fine-tuning
    print("\n" + "━"*70)
    result_10 = experiment_progressive_finetune(base_model_path, qn_data_path, features_list)
    all_experiment_results["exp10_progressive_finetune"] = result_10
    
    # Experiment 11: Data Augmentation
    print("\n" + "━"*70)
    result_11 = experiment_data_augmentation(base_model_path, qn_data_path, features_list,
                                             fixed_lr=best_lr)
    all_experiment_results["exp11_data_augmentation"] = result_11
    
    # Experiment 12: Ensemble
    print("\n" + "━"*70)
    result_12 = experiment_ensemble_finetune(base_model_path, qn_data_path, features_list)
    all_experiment_results["exp12_ensemble"] = result_12
    
    # Tổng kết
    total_time = time.time() - total_start
    
    print(f"\n{'#'*70}")
    print(f"#  HOÀN THÀNH TẤT CẢ THỬ NGHIỆM")
    print(f"#  Tổng thời gian: {format_time(total_time)}")
    print(f"#  Species: {species.upper()}")
    print(f"{'#'*70}")
    
    print(f"\n  📋 TÓM TẮT KẾT QUẢ:")
    print(f"  {'─'*50}")
    
    if result_1:
        print(f"  1. Best LR: {result_1.get('best_learning_rate', 'N/A')}")
    if result_2:
        print(f"  2. Best n_estimators: {result_2.get('best_n_estimators', 'N/A')}")
    if result_3:
        print(f"  3. CV RMSE: {result_3.get('avg_rmse', 'N/A'):.4f} ± {result_3.get('std_rmse', 0):.4f}")
    if result_4:
        print(f"  4. Best lag config: {result_4.get('best_lag_config', 'N/A')}")
    if result_5:
        print(f"  5. Best patience: {result_5.get('best_patience', 'N/A')}")
    if result_7:
        print(f"  7. Best params: {result_7.get('best_params', 'N/A')}")
    if result_8:
        print(f"  8. FT vs Scratch: {result_8.get('winner', 'N/A')} ({result_8.get('improvement_percent', 0):+.1f}%)")
    if result_10:
        print(f"  10. Progressive improvement: {result_10.get('improvement_percent', 0):+.1f}%")
    if result_12:
        print(f"  12. Best ensemble method: {result_12.get('best_method', 'N/A')}")
    
    return all_experiment_results


if __name__ == "__main__":
    print("="*70)
    print("  FINE-TUNING EXPERIMENT SUITE")
    print("  Nguyễn Phương Linh - Transfer Learning Research")
    print("="*70)
    
    results_cobia = run_all_experiments(species="cobia")
    
    results_oyster = run_all_experiments(species="oyster")
    
    print("\n\n" + "="*70)
    print("  ✅ TẤT CẢ THỬ NGHIỆM ĐÃ HOÀN THÀNH")
    print("  Kết luận: Sử dụng kết quả tốt nhất để cấu hình cho")
    print("  file finetune_cobia.py và finetune_oyster.py")
    print("="*70)
