import os

DATA_DIR = "data"
FULL_DS_DIR = os.path.join(DATA_DIR, "assignment_dataset")
TRAIN_DS_DIR = os.path.join(DATA_DIR, "train_dataset")
VAL_DS_DIR = os.path.join(DATA_DIR, "validation_dataset")
INPUT_FILENAME_PATTERN = "*_input.txt"
OUTPUT_FILENAME_PATTERN = "*_output.json"

SPLIT_DS_SEED = 689
MODELS_RANDOM_SEED = 20
