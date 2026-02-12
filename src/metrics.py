from .data_utils import prep_inpts_targets, filter_irrelevant
import time
from jiwer import cer
import numpy as np

import pymorphy3 as pm
from functools import lru_cache


class Lemmatizer:
    morph = pm.MorphAnalyzer()

    @classmethod
    @lru_cache(123123123)
    def lemmatize(cls, word: str):
        return cls.morph.parse(word)[0].normal_form


def __preproc(word: str, normalize=False):

    result = word.lower().replace("ё", "е")

    if normalize:
        result = Lemmatizer.lemmatize(result)

    return result


def lemmatization_accuracy(
    targets: list[str],
    preds: list[str],
    normalize=False
):

    assert len(targets) == len(preds)

    matches = 0
    tot = len(targets)

    for target, pred in zip(targets, preds):
        
        matches += __preproc(target, normalize=normalize) == __preproc(pred, normalize=normalize)

    return matches / tot


def lemmatization_cer(targets: list[str], preds: list[str]):

    assert len(targets) == len(preds)

    cers = []

    for target, pred in zip(targets, preds):
        
        preproc_target = __preproc(target)
        preproc_pred   = __preproc(pred)

        if preproc_target != preproc_pred:

            cers.append(
                cer(preproc_target, preproc_pred)
            )

    return np.mean(cers)


def preds_time_tps_lacc(predict_function, df):
    inputs, targets = prep_inpts_targets(df)

    start = time.perf_counter()

    preds = predict_function(inputs)

    end = time.perf_counter()

    elapsed_time = end - start

    tps = len(inputs) / elapsed_time

    lacc = lemmatization_accuracy(targets, preds)

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


def _get_freq_dfs(df, filter_irrelevant_=True):
    df_copy = df.copy()

    if filter_irrelevant_:
        df_copy = filter_irrelevant(df_copy)

    df_copy["freq_class"] = df_copy["freq_rank"].map(_get_freq_class)

    freq_dfs = {
        "1-100": df_copy[df_copy["freq_class"] == "1-100"],
        "101-1000": df_copy[df_copy["freq_class"] == "101-1000"],
        "1001-10000": df_copy[df_copy["freq_class"] == "1001-10000"],
        "10001-n": df_copy[df_copy["freq_class"] == "10001-n"],
        "all": df_copy,
    }

    return freq_dfs


def lacc_cer_by_freq_classes(
    predict_function,
    df,
    filter_irrelevant_=True,
    return_preds=False
):

    freq_dfs = _get_freq_dfs(df, filter_irrelevant_=filter_irrelevant_)

    result = dict()

    if return_preds:
        result_preds = dict()
        result_targets = dict()

    for freq_name, freq_df in freq_dfs.items():
        inputs, targets = prep_inpts_targets(freq_df)

        preds = predict_function(inputs)

        lacc = lemmatization_accuracy(targets, preds)
        cer_ = lemmatization_cer(targets, preds)
        lacc_norm = lemmatization_accuracy(targets, preds, normalize=True)

        result[freq_name] = (lacc, cer_, lacc_norm)

        if return_preds:
            result_preds[freq_name] = preds
            result_targets[freq_name] = targets

    if return_preds:
        return result, result_preds, result_targets

    return result


