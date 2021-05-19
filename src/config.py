# config.py

BINARY_FILE = "input/raw/20210420_data/stack.bin"
BINARY_DATA_FILE = "input/raw/20210420_data/stack.bin"
BINARY_MASK_FILE = "input/raw/20210420_data/mask.bin"
INFERENCE_FILE = "output/prediction.bin"
LABEL_DIR = "input/raw/zoom/labels/"
LABEL_FILE = "input/raw/20210420_data/mask.bin"
TRAINING_FILES = [
    "input/processed/20210420_data/data.csv"
]

DATA_FILE = "input/processed/20210420_data/data.csv"
TARGET_FILE = "input/processed/20210420_data/lake.csv"


MODEL_OUTPUT = "models/2021/"

GRID_SEARCH_PARAMS = {
    "n_estimators": [500, 1000, 2500],
    "max_depth": [1, 3, 8],
    "max_features": [0.1, 'auto'],
    "criterion": ["gini", "entropy"]
}

INFERENCE_MODEL = "models/2021/run__2021_05/data_rf-stump_grid_smote_2.bin"