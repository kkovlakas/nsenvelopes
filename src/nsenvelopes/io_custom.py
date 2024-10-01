"""Loading NS envelope data in the de Grandis et al. in prep. format."""

import io
import warnings
import numpy as np
import pandas as pd
from nsenvelopes.io import lines_reader


MAX_LOGT = 10.0
MAX_LOGRHO = 9.7

FEATURE_COLUMNS = ["logrho", "logT", "logB_mag", "B_ang"]
TARGET_COLUMNS = ["logTs"]


def dfile_reader(file, ignore_last_columns=2):
    """Read NS envelope data from a file in a ZIP archive or folder."""
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
