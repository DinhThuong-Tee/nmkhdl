from pathlib import Path


# ===== DIRECTORY PATHS =====

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
QN_DATA_DIR = DATA_DIR / "data_quang_ninh"
HK_DATA_DIR = DATA_DIR / "hk_water_quality"
MODEL_OUTPUT_DIR = PROJECT_DIR / "model" / "output"

# Key data files
QN_ENV_CSV = QN_DATA_DIR / "qn_env_clean_ready.csv"
QN_COORDS_CSV = QN_DATA_DIR / "toa_do_qn.csv"


# ===== FEATURE LISTS =====

# Hàu (Oyster) — 12 environmental features
OYSTER_FEATURES = [
    "DO",
    "Temperature",
    "pH",
    "Salinity",
    "NH3",
    "H2S",
    "BOD5",
    "COD",
    "TSS",
    "Coliform",
    "Alkalinity",
    "Transparency",
]

# Cá giò (Cobia) — 12 environmental features
COBIA_FEATURES = [
    "DO",
    "Temperature",
    "pH",
    "Salinity",
    "NH3",
    "PO4",
    "BOD5",
    "COD",
    "TSS",
    "Coliform",
    "Alkalinity",
    "Transparency",
]

# Kim loại nặng (Heavy metals) — 8 chemical features
METAL_FEATURES = ["CN", "As", "Cd", "Pb", "Cu", "Hg", "Zn", "Total_Cr"]

# Default lag periods for time-series feature construction
DEFAULT_LAGS = [1, 4]


# ===== MODEL HYPERPARAMETERS =====

# Base model (trained on HK data)
BASE_XGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)

# Metal model (trained on QN data)
METAL_XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)

# Fine-tuning learning rate (reduced to prevent catastrophic forgetting)
FINETUNE_LEARNING_RATE = 0.005


# ===== MODEL FILE PATHS =====

# Base models (trained on HK data)
HK_OYSTER_BASE_MODEL = MODEL_OUTPUT_DIR / "hk_oyster_forecast_model.pkl"
HK_COBIA_BASE_MODEL = MODEL_OUTPUT_DIR / "hk_cobia_forecast_model.pkl"

# Fine-tuned models (adapted to QN data)
HK_OYSTER_FINETUNED_MODEL = MODEL_OUTPUT_DIR / "hk_oyster_finetuned.pkl"
HK_COBIA_FINETUNED_MODEL = MODEL_OUTPUT_DIR / "hk_cobia_finetuned.pkl"

# Metal model
METAL_MODEL = MODEL_OUTPUT_DIR / "metal_ts_model.pkl"


# ===== HSI CONFIGURATION =====

HSI_THRESHOLDS = {
    "very_suitable": 0.85,
    "suitable": 0.75,
    "less_suitable": 0.50,
}
