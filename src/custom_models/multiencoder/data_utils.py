import pandas as pd


def get_inpt_from_row(row: dict[str, str]):

    pos = row["pos"]
    if pd.isna(pos):
        pos = "POS:[UNDEF]"
    else:
        pos = f"POS:{pos}"

    feats = row["feats"]
    if pd.isna(feats):
        feats = ""

    result = [pos]
    for feat in feats.split():
        result.append(feat)

    return result
