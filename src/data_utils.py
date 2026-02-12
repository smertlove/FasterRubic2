import pandas as pd


def filter_irrelevant(df):
    df_filtered = df[
        ~df["feats"].str.contains("Typo", na=False) &
        ~df["feats"].str.contains("Abbr", na=False) &
        ~df["feats"].str.contains("Anom", na=False) &
        ~df["feats"].str.contains("SYM" , na=False)
    ]
    return df_filtered


def prune_frequent_samples(df, min_=50):

    groups = df.groupby("sample")
    result = []

    for _, group in groups:
        freq = len(group)

        if freq <= min_:
            result.append(group)

        else:

            result.append(
                group.sample(
                    n=min_,
                    random_state=42
                )
            )

    df_fixed = pd.concat(result, ignore_index=True)

    return df_fixed.reset_index()


def get_sample_from_row_original(row):
    """
        Сампл для оригинальной модели
    """
    form  = row["form"]
    pos   = row["pos"]
    feats = row["feats"]
    lemma = row["lemma"]

    sample = " ".join(
        filter(
            lambda elem: pd.notna(elem),
            [form, pos, feats]
        )
    )

    return sample, lemma


def prep_inpts_targets(df: pd.DataFrame):
    inputs = []
    targets = []

    for _, row in df.iterrows():
        sample, lemma = get_sample_from_row_original(row)
        inputs.append(sample)
        targets.append(lemma)

    return inputs, targets
