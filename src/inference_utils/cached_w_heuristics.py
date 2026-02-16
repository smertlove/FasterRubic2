from tqdm import tqdm
import pymorphy3 as pm


class GenerativeModelWithHeuristicsAndCaching:

    def __init__(self, gen_model):
        self.model = gen_model
        self.morph = pm.MorphAnalyzer()
        self._cache: dict[str, str] = dict()

    def _exec_model_predict(self, batch: list[str]):
        uniq_inpts = list(set(batch))
        preds = self.model.predict(uniq_inpts)

        for inpt, pred in zip(uniq_inpts, preds):
            self._cache[inpt] = pred

        result = [self._cache[inpt] for inpt in batch]
        return result

    def _apply_heuristics(self, inpt: str):
        if "PUNCT" in inpt or "Foreign:Yes" in inpt or "SYM" in inpt:
            result = inpt.split(maxsplit=1)[0]
        elif "NOUN" in inpt and "Case:Nom" in inpt and " Number:Sing" in inpt:
            result = inpt.split(maxsplit=1)[0]
        elif "VERB" in inpt and "VerbForm:Inf" in inpt:
            result = inpt.split(maxsplit=1)[0]
        elif "NUM" in inpt or "ANUM" in inpt:
            form = inpt.split(maxsplit=1)[0]
            if not form.isalpha():
                result = form
            else:
                result = None
        else:
            result = None

        if result is not None:
            self._cache[inpt] = result

        return result


    def _predict_batch(self, batch: list[str]):

        result = []

        for inpt in batch:
            x = self._cache.get(inpt)
            if x is not None:
                result.append(x)
                continue

            x = self._apply_heuristics(inpt)
            result.append(x)

        pruned_batch = [
            inpt
            for inpt, res
            in zip(batch, result)
            if res is None
        ]

        if pruned_batch:
            preds = self._exec_model_predict(pruned_batch)

            preds_ptr = 0
            for i in range(len(result)):
                if result[i] is None:
                    result[i] = preds[preds_ptr]
                    preds_ptr += 1

        return result

    def predict(self, inpts: list[str], bs=64):
        result = []

        ln = len(inpts)
        tot = int(ln / bs)

        for i in tqdm(range(0, ln, bs), total=tot):
            batch = inpts[i:i+bs]
            preds = self._predict_batch(batch)
            result.extend(preds)

        return result
