import os

# --- PATHS ---
# Get project root assuming config.py is in the root directory
# If config.py is in src/, use os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "saved_models", "stgnn_multiyear_study_v1")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# --- ARTIFACT FILENAMES ---
SCALER_FILENAME = "feature_scaler_multiyear.joblib"
MODEL_FILENAME = "stgnn_multiyear_study_v1_model_final_trained_multiyear.pth"
BEST_PARAMS_FILENAME = "stgnn_multiyear_study_v1_best_params.json"
COLUMN_ORDER_FILENAME = "training_column_order.json"
COMMON_FEATURES_FILENAME = "common_features_config.json"

# --- MODEL & DATA PARAMS ---
INPUT_SEQ_LEN = 24
PREDICTION_HORIZON = 1
ZONE_IDENTIFIERS = ['DOM', 'PN', 'PEPCO', 'AECO', 'PE']
TARGET_COLUMN_TEMPLATE = '{ZONE}_Load'