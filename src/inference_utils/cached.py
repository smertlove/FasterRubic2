from tqdm import tqdm


class GenerativeModelWithCaching:
    # TODO: Переделать все эти модели через наследование?

    def __init__(self, gen_model):
        self.model = gen_model

        self._cache: dict[str, str] = dict()

    def _exec_model_predict(self, batch: list[str]):
        uniq_inpts = list(set(batch))
        preds = self.model.predict(uniq_inpts)

        for inpt, pred in zip(uniq_inpts, preds):
            self._cache[inpt] = pred

        result = [self._cache[inpt] for inpt in batch]
        return result

    def _predict_batch(self, batch: list[str]):

        result = [self._cache.get(inpt) for inpt in batch]
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
