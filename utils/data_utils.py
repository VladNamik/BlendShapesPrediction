from config import *
import errno
import os
import shutil
import random
import glob
import json
import numpy as np


def split_train_val(full_ds_dir=FULL_DS_DIR, train_ds_dir=TRAIN_DS_DIR, val_ds_dir=VAL_DS_DIR, split_val=0.8):
    if not os.path.exists(full_ds_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), full_ds_dir)

    input_filenames = sorted(glob.glob(os.path.join(full_ds_dir, INPUT_FILENAME_PATTERN)))
    output_filenames = sorted(glob.glob(os.path.join(full_ds_dir, OUTPUT_FILENAME_PATTERN)))
    ds_filenames = list(zip(input_filenames, output_filenames))

    if os.path.exists(train_ds_dir):
        shutil.rmtree(train_ds_dir)
    os.makedirs(train_ds_dir)
    if os.path.exists(val_ds_dir):
        shutil.rmtree(val_ds_dir)
    os.makedirs(val_ds_dir)

    random.seed(SPLIT_DS_SEED)
    random.shuffle(ds_filenames)

    # saving training dataset
    for input_filename, output_filename in ds_filenames[:int(len(ds_filenames) * split_val)]:
        shutil.copyfile(input_filename, os.path.join(train_ds_dir, os.path.basename(input_filename)))
        shutil.copyfile(output_filename, os.path.join(train_ds_dir, os.path.basename(output_filename)))

    # saving validation dataset
    for input_filename, output_filename in ds_filenames[int(len(ds_filenames) * split_val):]:
        shutil.copyfile(input_filename, os.path.join(val_ds_dir, os.path.basename(input_filename)))
        shutil.copyfile(output_filename, os.path.join(val_ds_dir, os.path.basename(output_filename)))


def read_ds(ds_path):
    input_filenames = sorted(glob.glob(os.path.join(ds_path, INPUT_FILENAME_PATTERN)))
    output_filenames = sorted(glob.glob(os.path.join(ds_path, OUTPUT_FILENAME_PATTERN)))
    ds_filenames = list(zip(input_filenames, output_filenames))

    input_data_list = []
    output_data_list = []
    for input_filename, output_filename in ds_filenames:
        with open(input_filename) as f:
            input_data = np.array([float(x.strip()) for x in f.readlines()], dtype=np.float64)
        with open(output_filename) as f:
            output_data = json.load(f)
            output_data = np.array([float(value) for _, value in output_data.items()], dtype=np.float64)

        input_data_list.append(input_data)
        output_data_list.append(output_data)

    return input_data_list, output_data_list
