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


MAX_LOGT = 10.0
MAX_LOGRHO = 9.7

FEATURE_COLUMNS = ["logrho", "logT", "logB_mag", "B_ang"]
TARGET_COLUMNS = ["logTs"]

SUBSET_NAMES = ["holdout", "working"]
WORKING_SUBSET_NAMES = ["train", "valid", "test"]


def lines_reader(file):
    contents = io.TextIOWrapper(file, encoding='utf-8').readlines()
    return contents


def csv_reader(file):
    return pd.read_csv(io.TextIOWrapper(file, encoding='utf-8'))


def transform_dataframe(old_df, B_mag=None, B_ang=None,
                        max_logrho=MAX_LOGRHO, max_logT=MAX_LOGT):
    """Tranform data frame to a convenient form."""
    logrho = np.log10(old_df["rho"])
    logT = np.log10(np.exp(old_df["lnT"]))

    if max_logrho is not None:
        valid = logrho <= max_logrho
        logrho, logT = logrho[valid], logT[valid]

    if max_logT is None:
        valid = np.isfinite(logT)
    else:
        valid = logT <= max_logT

    if not np.all(valid):
        last_ok = np.where(valid)[0][-1]
        first_bad = np.where(~valid)[0][0]
        if last_ok >= first_bad:
            raise RuntimeError("The temperature profile is weird.")
        warnings.warn(f"Found logT > {max_logT} for "
                      f"logrho >= {logrho[first_bad]:.2f}... Removing curve!")
        return None

    new_df = pd.DataFrame()
    # new_df.logB_mag = np.log10(B_mag)
    # new_df.B_ang = B_ang

    # new_df.logTs = np.min(logT)
    new_df["logrho"] = logrho
    new_df["logT"] = logT
    new_df["logB_mag"] = np.log10(B_mag)
    new_df["B_ang"] = B_ang
    new_df["logTs"] = np.min(logT)
    return new_df


def dfile_reader(file, ignore_last_columns=2):
    lines = lines_reader(file)

    B_mag = None
    B_ang = None

    in_header = True
    csv_contents = ""
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            if not in_header:
                raise ValueError("Header should not be mixed with data.")
            line = line[1:]
            if "B [G]" in line:
                B_mag = float(line.split("B [G]")[-1])
            elif "ThetaB" in line:
                B_ang = float(line.split("ThetaB")[-1])
            elif "|" in line:
                header = line.split("|")
                header = [field.strip() for field in header]
                header = [field.replace(" ", "")
                          for field in header if field != '']
                in_header = False
            else:
                raise ValueError(
                    f"Unrecognized header line: {line}")
        else:
            if in_header:
                raise ValueError("Data should not be mixed with header.")
            csv_contents += line + "\n"

    csv_file_like = io.StringIO(csv_contents)

    try:
        data = pd.read_csv(csv_file_like, header=None, delim_whitespace=True)
    except pd.errors.EmptyDataError:
        return None

    if not (len(header) <= len(data.columns) <= len(header)
            + ignore_last_columns):
        raise RuntimeError("Number of columns do not match with header!")

    data.rename(columns={i: header[i] for i in range(len(header))},
                inplace=True)
    data = transform_dataframe(data, B_mag=B_mag, B_ang=B_ang)
    return data


class CurvesLoader:
    def __init__(self, path, txtreader=csv_reader, binreader=None,
                 maxfiles=None):
        self.path = self.set_path(path)
        self.reader, self.binary_format = self.set_reader(
            txtreader=txtreader, binreader=binreader)
        self.data = None
        self.load_data(maxfiles=maxfiles)

    @staticmethod
    def set_path(path):
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        return path

    def set_reader(self, txtreader, binreader):
        if txtreader is None:
            if binreader is None:
                raise ValueError("At least one reader must be specified.")
            else:
                return binreader, True
        else:
            return txtreader, False

    def load_data_from_dir(self, path, maxfiles=None):
        raise NotImplementedError("`load_data_from_dir` not implemented yet.")

    def load_data_from_zip(self, path, maxfiles=None):
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
        if os.path.isdir(self.path):
            self.load_data_from_dir(self.path, maxfiles=maxfiles)
        elif os.path.isfile(self.path):
            self.load_data_from_zip(self.path, maxfiles=maxfiles)
        else:
            raise ValueError(f"Path `{self.path}` is not a file or directory.")

    def make_subsets(self,
                     output_dir,
                     features=FEATURE_COLUMNS,
                     targets=TARGET_COLUMNS,
                     holdout=0.1,
                     working_validation=0.1,
                     working_test=0.1,
                     # seed=None,
                     overwrite=True):
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

        subsets = split_to_X_y(subsets, features, targets)
        for key, subset in subsets.items():
            subpath = os.path.join(output_dir, key + ".csv")
            subset.to_csv(subpath, index=False)


def split_list(original_list, fraction=0.3, shuffle=True):
    size = len(original_list)
    assert 0 < fraction < 1 and size >= 2
    cut_at = size - max(1, int(np.round(size * fraction)))
    list_to_split = copy.deepcopy(original_list)
    if shuffle:
        random.shuffle(list_to_split)
    return list_to_split[:cut_at], list_to_split[cut_at:]


def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def split_to_X_y(subsets, features, targets):
    arrays = {}
    for subset_name in subsets.keys():
        arrays["X_" + subset_name] = subsets[subset_name][features]
        arrays["y_" + subset_name] = subsets[subset_name][targets]
    return arrays


def load_subsets(subsets_dir):
    data = {}
    subsets_to_search = WORKING_SUBSET_NAMES + [SUBSET_NAMES[0]]
    for subset_name in subsets_to_search:
        X_name = "X_" + subset_name
        y_name = "y_" + subset_name

        X_path = os.path.join(subsets_dir, X_name + ".csv")
        y_path = os.path.join(subsets_dir, y_name + ".csv")

        X_exists = os.path.exists(X_path)
        y_exists = os.path.exists(y_path)
        if X_exists and y_exists:
            data[X_name] = pd.read_csv(X_path)
            data[y_name] = pd.read_csv(y_path)
        elif X_exists or y_exists:
            raise ValueError(f"Both X and y must exist for `{subset_name}`.")

    if len(data) == 0:
        raise ValueError("No data were loaded.")
    if len(data) != 2 * len(subsets_to_search):
        warnings.warn("Some data were not loaded.")

    return data
