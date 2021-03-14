# config.py

BINARY_FILE = "input/raw/zoom/S2A.bin"
LABEL_DIR = "input/raw/zoom/labels/"

TRAINING_FILE = "input/processed/zoom/s2a_train_folds_water.csv"

MODEL_OUTPUT = "models/zoom/"

GRID_SEARCH_PARAMS = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 4, 5, 7, 9],
    "criterion": ["gini", "entropy"],
    "max_features": ["auto", 0.1]
}

INFERENCE_MODEL = "models/zoom/water_rf_after_grid_1.bin"