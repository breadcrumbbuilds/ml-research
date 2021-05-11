# config.py

BINARY_FILE = "input/raw/20210420_data/mask.bin"
BINARY_DATA_FILE = "input/raw/20210420_data/stack.bin"
BINARY_MASK_FILE = "input/raw/20210420_data/mask.bin"
LABEL_DIR = "input/raw/zoom/labels/"
LABEL_FILE = "input/raw/20210420_data/mask.bin"
TRAINING_FILES = [
    "input/processed/20210420_data/lake.csv",
    "input/processed/20210420_data/fireweedgrass.csv",
    "input/processed/20210420_data/blowdownfireweed.csv",
    "input/processed/20210420_data/deciduous.csv",
    "input/processed/20210420_data/fireweedgrass.csv",
    "input/processed/20210420_data/lake.csv",
    "input/processed/20210420_data/pineburnedfireweed.csv",
    "input/processed/20210420_data/blowdownlichen.csv",
    "input/processed/20210420_data/exposed.csv",
    "input/processed/20210420_data/grass.csv",
    "input/processed/20210420_data/pineburned.csv",
    "input/processed/20210420_data/windthrowgreenherbs}.csv",
]

DATA_FILE = "input/processed/20210420_data/data.csv"
TARGET_FILE = "input/processed/20210420_data/lake.csv"


MODEL_OUTPUT = "models/2021/"

GRID_SEARCH_PARAMS = {
    "n_estimators": [500, 1000, 2500],
    "max_depth": [1, 3],
    "max_features": [0.1, 'auto'],
    "criterion": ["gini", "entropy"]
}

INFERENCE_MODEL = "models/2021/run__2021_05/lake_rf-stump_None_smote-pipeline.bin"