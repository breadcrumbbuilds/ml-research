# config.py

BINARY_FILE = "input/raw/zoom/S2A.bin"
LABEL_DIR = "input/raw/zoom/labels/"

TRAINING_FILE = "input/processed/zoom/s2a_train_folds_water.csv"

MODEL_OUTPUT = "models/zoom/"

GRID_SEARCH_PARAMS = {
    "n_estimators": [500, 1000],
    "max_depth": [3, 12],
    "max_features": [0.1, 'auto']
}

INFERENCE_MODEL = "models/zoom/water_svm-opt-water_0.bin"