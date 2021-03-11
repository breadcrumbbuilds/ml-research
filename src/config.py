# config.py

BINARY_FILE = "input/raw/full/S2A.bin"
LABEL_DIR = "input/raw/full/labels/"

TRAINING_FILE = "input/processed/full/train_folds_conifer.csv"

MODEL_OUTPUT = "models/"

GRID_SEARCH_PARAMS = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 4, 5, 7, 9],
    "criterion": ["gini", "entropy"],
    "max_features": ["auto", 0.1]
}