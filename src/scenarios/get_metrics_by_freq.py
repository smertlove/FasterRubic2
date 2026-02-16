import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path

from src.metrics import lacc_cer_by_freq_classes
from src.data_utils import get_sample_from_row_original
from datetime import datetime


def run_script(df: pd.DataFrame, predict_function, save_cache=False):

    df["sample"] = df.apply(lambda row: get_sample_from_row_original(row)[0], axis=1)

    df_holdout = df[df["split"] == "holdout"]
    df_unknown = df[df["split"] == "unknown"]

    print(df.head())
    print(f"Total {df.shape[0]}, holdout {df_holdout.shape[0]}, unknown {df_unknown.shape[0]}")
    
    all_ = lacc_cer_by_freq_classes(predict_function, df).round(3)
    holdout_ = lacc_cer_by_freq_classes(predict_function, df_holdout).round(3)
    unknown_ = lacc_cer_by_freq_classes(predict_function, df_unknown).round(3)

    return all_, holdout_, unknown_
