"""Module for reading curves from ZIP files and folders, ready for ML/DL."""


import os
import io
import copy
import random
import shutil
import zipfile
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SUBSET_NAMES = ["holdout", "working"]
WORKING_SUBSET_NAMES = ["train", "valid", "test"]


def lines_reader(file):
    """Read lines from a file in a ZIP archive or folder."""
    contents = io.TextIOWrapper(file, encoding='utf-8').readlines()
    return contents


def csv_reader(file):
    """Read CSV file from a ZIP archive or folder."""
    return pd.read_csv(io.TextIOWrapper(file, encoding='utf-8'))


class CurvesLoader:
    """Class for loading curve data from ZIP files and folders."""
    def __init__(self, path, txtreader=csv_reader, binreader=None,
                 maxfiles=None):
        self.path = self.set_path(path)
        self.reader, self.binary_format = self.set_reader(
            txtreader=txtreader, binreader=binreader)
        self.data = None
        self.load_data(maxfiles=maxfiles)

    @staticmethod
    def set_path(path):
        """"Set the input path after performing checks."""
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        return path

    def set_reader(self, txtreader, binreader):
        """Set the file content reader."""
        if txtreader is None:
            if binreader is None:
                raise ValueError("At least one reader must be specified.")
            else:
                return binreader, True
        else:
            return txtreader, False

    def load_data_from_dir(self, path, maxfiles=None):
        """Load data from a directory."""
        raise NotImplementedError("`load_data_from_dir` not implemented yet.")

    def load_data_from_zip(self, path, maxfiles=None):
        """Load data from a ZIP archive."""
        with zipfile.ZipFile(path) as zip_file:
            self.data = {}
            for filename in zip_file.namelist():
                if not filename.endswith('/'):
                    with zip_file.open(filename, "r") as file:
                        data = self.reader(file)
                        if data is not None:
                            data.filename = filename
                            self.data[filename] = data
                            if maxfiles is not None:
                                if len(self.data) >= maxfiles:
                                    break
                        else:
                            warnings.warn(f"File {filename} is empty.")

    def load_data(self, maxfiles=None):
        """Load data from a file or directory."""
        if os.path.isdir(self.path):
            self.load_data_from_dir(self.path, maxfiles=maxfiles)
        elif os.path.isfile(self.path):
            self.load_data_from_zip(self.path, maxfiles=maxfiles)
        else:
            raise ValueError(f"Path `{self.path}` is not a file or directory.")

    def make_subsets(self,
                     output_dir,
                     features,
                     targets,
                     holdout=0.1,
                     working_validation=0.1,
                     working_test=0.1,
                     # seed=None,
                     overwrite=True):
        """Write the holdout/training/validation/test subsets."""
        if not 0 < holdout < 1:
            raise ValueError("`holdout_curves_fraction` must be in [0, 1).")
        if not 0 < working_test < 1:
            raise ValueError("`included_curves_test` must be in [0, 1).")
        if not 0 < working_validation < 1:
            raise ValueError("`included_curves_validation` must be in [0, 1).")

        if working_test + working_validation >= 1:
            raise ValueError("Nothing will be left for training...")

        filenames = np.array(list(self.data.keys()))
        working_filenames, holdout_filenames = split_list(filenames,
                                                          fraction=holdout)

        if os.path.exists(output_dir):
            if not overwrite:
                raise ValueError(
                    f"Directory `{output_dir}` already exists.")
            if not os.path.isdir(output_dir):
                raise ValueError(
                    f"`{output_dir}` is not a directory. Cannot overwrite.")
        clear_folder(output_dir)

        subsets = {subsetname: pd.DataFrame() for subsetname in SUBSET_NAMES}

        for subset_filenames, subset_name in zip(
                [holdout_filenames, working_filenames], SUBSET_NAMES):
            subset_path = os.path.join(output_dir, subset_name + ".txt")
            with open(subset_path, "w", encoding="utf-8") as file:
                for filename in subset_filenames:
                    file.write(filename + "\n")
                    subsets[subset_name] = pd.concat([subsets[subset_name],
                                                      self.data[filename]],
                                                     ignore_index=True)

        subsets["train"], subsets["validtest"] = train_test_split(
            subsets["working"], test_size=working_test + working_validation,
            shuffle=True)
        holdout = subsets["holdout"]
        del subsets["working"]

        subsets["valid"], subsets["test"] = train_test_split(
            subsets["validtest"],
            test_size=working_test / (working_test + working_validation),
            shuffle=True)
        del subsets["validtest"]

        subsets = split_to_features_and_targets(subsets, features, targets)
        for key, subset in subsets.items():
            subpath = os.path.join(output_dir, key + ".csv")
            subset.to_csv(subpath, index=False)


def split_list(original_list, fraction=0.3, shuffle=True):
    """Split a list into two parts given the fraction of the second one."""
    size = len(original_list)
    assert 0 < fraction < 1 and size >= 2
    cut_at = size - max(1, int(np.round(size * fraction)))
    list_to_split = copy.deepcopy(original_list)
    if shuffle:
        random.shuffle(list_to_split)
    return list_to_split[:cut_at], list_to_split[cut_at:]


def clear_folder(folder):
    """Remove and recreate a folder."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def split_to_features_and_targets(subsets, features, targets):
    """Split subsets to X and y dataframes for machine/deep-learning."""
    arrays = {}
    for subset_name in subsets.keys():
        arrays["X_" + subset_name] = subsets[subset_name][features]
        arrays["y_" + subset_name] = subsets[subset_name][targets]
    return arrays


def load_subsets(subsets_dir):
    """Load holdout/training/validatoin/test subsets from a directory."""
    data = {}
    subsets_to_search = WORKING_SUBSET_NAMES + [SUBSET_NAMES[0]]
    for subset_name in subsets_to_search:
        x_name = "X_" + subset_name
        y_name = "y_" + subset_name

        x_path = os.path.join(subsets_dir, x_name + ".csv")
        y_path = os.path.join(subsets_dir, y_name + ".csv")

        x_exists = os.path.exists(x_path)
        y_exists = os.path.exists(y_path)
        if x_exists and y_exists:
            data[x_name] = pd.read_csv(x_path)
            data[y_name] = pd.read_csv(y_path)
        elif x_exists or y_exists:
            raise ValueError(f"Both X and y must exist for `{subset_name}`.")

    if len(data) == 0:
        raise ValueError("No data were loaded.")
    if len(data) != 2 * len(subsets_to_search):
        warnings.warn("Some data were not loaded.")

    return data
