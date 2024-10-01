import os
import io
import zipfile
import warnings

import numpy as np
import pandas as pd

MAX_LOGT = 10.0
MAX_LOGRHO = 9.7


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


class EnvelopeData:
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
