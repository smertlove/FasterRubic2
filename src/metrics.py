from .data_utils import prep_inpts_targets, filter_irrelevant
import time
from jiwer import cer
import numpy as np

import pandas as pd

import pymorphy3 as pm
from functools import lru_cache


class _Lemmatizer:
    morph = pm.MorphAnalyzer()

    @classmethod
    @lru_cache(123123123)
    def lemmatize(cls, word: str):
        return cls.morph.parse(word)[0].normal_form


def __preproc(word: str, normalize=False):
    result = word.lower().replace("ё", "е")

    if normalize:
        result = _Lemmatizer.lemmatize(result)

    return result

def _preproc(word: str | list[str], normalize=False):

    if isinstance(word, str):
        return __preproc(word, normalize=normalize)
    else:
        return [__preproc(w, normalize=normalize) for w in word]


def _lemmatization_accuracy(targets: list[str], preds: list[str], normalize=False) -> list[int]:

    assert len(targets) == len(preds)

    accs = []

    for target, pred in zip(targets, preds):

        cur_acc = _preproc(target, normalize=normalize) == _preproc(
            pred, normalize=normalize
        )

        accs.append(cur_acc)

    return accs


def _lemmatization_cer(targets: list[str], preds: list[str]) -> list[float]:

    assert len(targets) == len(preds)

    cers = [
        cer(
            _preproc(target),
            _preproc(pred),
        )
        for target, pred in zip(targets, preds)
    ]

    return cers


def preds_time_tps_lacc(predict_function, df):
    inputs, targets = prep_inpts_targets(df)

    start = time.perf_counter()

    preds = predict_function(inputs)

    end = time.perf_counter()

    elapsed_time = end - start

    tps = len(inputs) / elapsed_time

    lacc = np.mean(_lemmatization_accuracy(targets, preds))

    return preds, elapsed_time, tps, lacc


def _get_freq_class(rank: str):
    if rank == "other":
        return "10001-n"

    rank_int = int(rank)

    if rank_int < 101:
        return "1-100"

    if rank_int < 1001:
        return "101-1000"

    if rank_int < 10001:
        return "1001-10000"

    return "10001-n"


def _add_freq_class(df):
    df_copy = df.copy()

    df_copy["freq_class"] = df_copy["freq_rank"].map(_get_freq_class)

    return df_copy

    # freq_dfs = {
    #     "1-100": df_copy[df_copy["freq_class"] == "1-100"],
    #     "101-1000": df_copy[df_copy["freq_class"] == "101-1000"],
    #     "1001-10000": df_copy[df_copy["freq_class"] == "1001-10000"],
    #     "10001-n": df_copy[df_copy["freq_class"] == "10001-n"],
    #     "all": df_copy,
    # }

    # return freq_dfs


def _get_freq_df(
    predict_function,
    df,
    filter_irrelevant_=True,
):

    df_copy = df.copy()
    if filter_irrelevant_:
        df_copy = filter_irrelevant(df_copy)

    freq_df = _add_freq_class(df_copy)

    inputs, targets = prep_inpts_targets(freq_df)
    preds = predict_function(inputs)

    freq_df["target"] = targets
    freq_df["pred"] = preds

    freq_df["lAcc"] = _lemmatization_accuracy(targets, preds)
    freq_df["lAcc (norm)"] = _lemmatization_accuracy(targets, preds, normalize=True)
    freq_df["CER"] = _lemmatization_cer(targets, preds)

    return freq_df


def lacc_cer_by_freq_classes(
    predict_function,
    df,
    filter_irrelevant_=True,
):
    freq_df = _get_freq_df(
        predict_function,
        df,
        filter_irrelevant_,
    )

    freq_groups = {
        "1-100": freq_df[freq_df["freq_class"] == "1-100"],
        "101-1000": freq_df[freq_df["freq_class"] == "101-1000"],
        "1001-10000": freq_df[freq_df["freq_class"] == "1001-10000"],
        "10001-n": freq_df[freq_df["freq_class"] == "10001-n"],
        "all": freq_df,
    }

    metrics = []

    for name, group in freq_groups.items():
        metrics.append({
            "class": name,
            "lAcc": group["lAcc"].mean(),
            "lAcc (norm)": group["lAcc (norm)"].mean(),
            "CER (total)": group["CER"].mean(),
            "CER (errors)": group[group["pred"] != group["target"]]["CER"].mean(),
        })

    return pd.DataFrame(metrics)