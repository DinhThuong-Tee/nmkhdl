import joblib
import pandas as pd
import numpy as np
import functools
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=8)
def _load_model(model_path):
    return joblib.load(str(model_path))


@functools.lru_cache(maxsize=4)
def _load_csv(csv_path):
    return pd.read_csv(str(csv_path))


def _get_csv_copy(csv_path):
    return _load_csv(csv_path).copy()


def predict_future_metal_field_for_station(start_year, start_quarter, n_quarters, x, y):
    """
    Rolling forecast nồng độ kim loại tại một trạm.

    Hàm này tải dữ liệu lịch sử đã được làm sạch từ CSV và mô hình chuỗi thời gian
    đã huấn luyện cho các biến kim loại (CN, As, Cd, Pb, Cu, Hg, Zn, Total_Cr).
    Hàm sử dụng 4 quý quan trắc gần nhất để tạo các đặc trưng độ trễ (lag)
    cần thiết cho mô hình (lag1 và lag4 cho từng kim loại), sau đó dự báo
    lặp dần từng quý trong tương lai (kết quả dự báo được đưa ngược lại
    làm dữ liệu lịch sử cho bước kế tiếp).

    Tham số
    ----------
    start_year : int
        Năm của quý dự báo đầu tiên.
    start_quarter : int
        Số quý (1..4) của bước dự báo đầu tiên.
    n_quarters : int
        Số lượng quý cần dự báo (theo kiểu rolling).
    x, y : numeric
        Tọa độ trạm, dùng để chọn trạm tương ứng trong file CSV (toạ độ VN2000)
        (các cột "X", "Y").

    Giá trị trả về
    -------
    pd.DataFrame
        DataFrame gồm các dòng tương ứng với từng quý dự báo, chứa các cột:
        "year", "quarter" và các cột kim loại dự báo (giá trị không âm).
    """

    DATA_PATH = _PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"
    model_path = _PROJECT_DIR / "model" / "output" / "metal_ts_model.pkl"

    # ===== PREDICT cho 1 trạm =====
    df = _get_csv_copy(DATA_PATH)
    df_station = df[(df["X"] == x) & (df["Y"] == y)]

    target_cols = ["CN", "As", "Cd", "Pb", "Cu", "Hg", "Zn", "Total_Cr"]

    model, feature_cols = _load_model(model_path)

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
            ignore_index=True,
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


def predict_future_non_metal_field_for_station(
    species, x, y, start_year, start_quarter, n_quarters=4
):
    """
    Rolling forecast các biến môi trường không phải kim loại
    tại một trạm.

    Hàm tải mô hình đã fine-tune theo loài (hàu hoặc cá giò) cùng với metadata
    đặc trưng của mô hình, sau đó trích xuất dữ liệu lịch sử của trạm từ file
    CSV đã làm sạch. Các đặc trưng độ trễ (lag1, lag4) và chỉ số quý được xây dựng
    để dự báo số quý tương lai yêu cầu. Kết quả dự báo của mỗi bước sẽ được
    bổ sung vào lịch sử để dùng cho bước dự báo tiếp theo (rolling forecast).

    Hành vi chính / các cơ chế bảo vệ:
    - Tự động chọn file mô hình dựa trên tham số `species`.
    - Tải metadata (input_cols, features) từ file "_features.pkl" đi kèm.
    - Yêu cầu tối thiểu 4 quý dữ liệu lịch sử của trạm (để tạo lag1 và lag4).
    - Ép kiểu dữ liệu lịch sử và các đặc trưng đầu vào về numeric
      (có thể phát sinh NaN nếu dữ liệu không hợp lệ).
    - Đảm bảo các giá trị dự báo được cắt về ≥ 0 (ràng buộc vật lý).

    Tham số
    ----------
    species : {"oyster", "cobia"}
        Loài sử dụng mô hình dự báo (hàu hoặc cá giò).
    x, y : numeric
        Tọa độ trạm, dùng để chọn trạm tương ứng trong file CSV
        (các cột "X", "Y").
    start_year : int
        Năm của quý dự báo đầu tiên.
    start_quarter : int
        Số quý (1..4) của bước dự báo đầu tiên.
    n_quarters : int, mặc định = 4
        Số lượng quý cần dự báo.

    Giá trị trả về
    -------
    pd.DataFrame
        DataFrame gồm các dòng cho từng quý dự báo, chứa các cột "year", "quarter"
        và các cột biến môi trường không phải kim loại (giá trị đã được cắt ≥ 0).
    """
    csv_data_path = _PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"
    if species == "cobia":
        model_path = _PROJECT_DIR / "model" / "output" / "hk_cobia_finetuned.pkl"
    elif species == "oyster":
        model_path = _PROJECT_DIR / "model" / "output" / "hk_oyster_finetuned.pkl"
    # ===== LOAD MODEL + METADATA =====
    loaded = _load_model(model_path)
    model = loaded
    meta_path = str(model_path).replace(".pkl", "_features.pkl")
    input_cols, features = _load_model(meta_path)

    # ===== LOAD DATA =====
    df = _get_csv_copy(csv_data_path)

    df["Date"] = pd.to_datetime(df["Quarter"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # ===== LỌC 1 TRẠM =====
    df_station = df[(df["X"] == x) & (df["Y"] == y)].copy()
    if len(df_station) == 0:
        raise ValueError(f"❌ Không tìm thấy trạm: {x}, {y}")

    df_station = df_station.sort_values("Date")

    # ===== ÉP NUMERIC (RẤT QUAN TRỌNG) =====
    for c in features:
        df_station[c] = pd.to_numeric(df_station[c], errors="coerce")

    # ===== LẤY LỊCH SỬ GẦN NHẤT (đủ cho lag 1 & 4) =====
    history = df_station[features].iloc[-4:].copy()
    if len(history) < 4:
        raise ValueError("❌ Không đủ dữ liệu lịch sử (cần ≥ 4 quý)")

    results = []
    year, quarter = start_year, start_quarter

    # ===== ROLLING FORECAST =====
    for _ in range(n_quarters):
        row = {}

        for c in features:
            row[f"{c}_lag1"] = float(history[c].iloc[-1])
            row[f"{c}_lag4"] = float(history[c].iloc[0])

        row["Quarter_Num"] = quarter

        X_pred = pd.DataFrame([row])[input_cols].astype(float)

        y_pred = model.predict(X_pred)[0]

        result = {"year": year, "quarter": quarter}
        result.update(dict(zip(features, y_pred)))
        results.append(result)

        # ---- cập nhật history ----
        history = pd.concat(
            [history.iloc[1:], pd.DataFrame([y_pred], columns=features)],
            ignore_index=True,
        )

        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1

    df_future = pd.DataFrame(results)

    # ===== CLIP ÂM (VẬT LÝ) =====
    for c in features:
        df_future[c] = df_future[c].clip(lower=0)

    return df_future


def predict_for_station(species, x, y, start_year, start_quarter, n_quarters=4):
    """
    Dự báo một quý cho trạm cụ thể.

    Hàm này sử dụng hàm `predict_future_non_metal_field_for_station` và `predict_future_metal_field_for_station`  để dự báo
    một quý duy nhất cho trạm xác định bởi tọa độ (x, y).

    Tham số
    ----------
    species : {"oyster", "cobia"}
        Loài sử dụng mô hình dự báo (hàu hoặc cá giò).
    """
    df1 = predict_future_non_metal_field_for_station(
        species=species,
        x=x,
        y=y,
        start_year=start_year,
        start_quarter=start_quarter,
        n_quarters=n_quarters,
    )
    df2 = predict_future_metal_field_for_station(
        start_year=start_year,
        start_quarter=start_quarter,
        n_quarters=n_quarters,
        x=x,
        y=y,
    )
    df_merged = pd.merge(df1, df2, on=["year", "quarter"], how="inner")
    return df_merged


if __name__ == "__main__":
    df = predict_for_station(
        species="cobia",
        x=2318587,
        y=428692,
        start_year=2026,
        start_quarter=1,
        n_quarters=4,
    )
    print(df)
    df.info()
