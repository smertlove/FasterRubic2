import json
from pathlib import Path
import pandas as pd


class PrecomputeBasedModel:
    def __init__(self, json_file: Path):
        with open(json_file / "data.json", "r") as file:
            self.precompute_dict: dict[str, str] = json.load(file)
        self.unknown_inpt = "[UNKNOWN INPT]"

    def predict(self, inpts: list[str]):
        result = []

        for inpt in inpts:
            pred = self.precompute_dict.get(inpt, self.unknown_inpt)
            result.append(pred)

        return result


def precompute_and_store(predict_function, df: pd.DataFrame, path_json: Path):

    path_json.parent.mkdir(parents=True)

    inpts = df["sample"].unique().tolist()
    preds = predict_function(inpts)

    inputs2preds = {inpt: pred for inpt, pred in zip(inpts, preds)}

    with open(path_json, "w") as file:
        json.dump(inputs2preds, file, ensure_ascii=False, indent=2)
