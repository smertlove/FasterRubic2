from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from tqdm import tqdm
import math
import torch

from typing import Iterable


class GenerativeModel:

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device="cuda",
    ):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    def preproc(self, texts: list[str]):
        """
        Предобрабатывает входные тексты
        """
        return texts

    def postproc(self, texts: list[str]):
        """
        Постобрабатывает гипотезы
        """
        return texts

    def predict(
        self,
        texts: list[str],
        max_length=32,
        batch_size=32,
        verbose=True,
    ) -> list[str]:

        texts = self.preproc(texts)

        preds = self._predict(
            texts, max_length=max_length, batch_size=batch_size, verbose=verbose
        )

        preds = self.postproc(preds)

        return preds

    def _predict(
        self,
        texts: list[str],
        max_length: int = 32,
        batch_size: int = 32,
        verbose=True,
    ) -> list[str]:

        preds = []

        if verbose:
            pbar: Iterable[int] = tqdm(
                range(0, len(texts), batch_size),
                total=int(len(texts) / batch_size),
            )
        else:
            pbar = range(0, len(texts), batch_size)

        for i in pbar:
            batch = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch,
                max_length=70,
                truncation=True,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=1,
                    temperature=1,
                    early_stopping=True,
                )

            cur_preds = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            preds.extend(cur_preds)

        return preds


class GenerativeModelWithCachingAndHeuristics(GenerativeModel):

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)

        self.bos = self.tokenizer.bos_token or ""
        self.eos = self.tokenizer.eos_token or ""

        self._cache: dict[str, str] = dict()

    def preproc(self, texts):
        return [self.bos + text + self.eos for text in texts]

    def postproc(self, texts):
        return ["".join(text.split()) for text in texts]

    def populate_global_cache(self, key: str, val: str):
        self._cache[key] = val

    def get_from_global_cache(self, key: str):
        return self._cache.get(key)

    def _exec_model_predict(
        self,
        batch: list[str],
        max_length=32,
        batch_size=32,
    ):
        uniq_inpts = list(set(batch))
        preds = self.predict(
            uniq_inpts, max_length=max_length, batch_size=batch_size, verbose=False
        )

        #  Populate local cache
        loc_cache = {inpt: pred for inpt, pred in zip(uniq_inpts, preds)}

        result = []
        for inpt in batch:
            pred = loc_cache[inpt]
            result.append(pred)

            # global cache might behave differently depending on implementation
            # so we populate it separately
            self.populate_global_cache(inpt, pred)

        return result

    def _predict_batch(
        self,
        batch: list[str],
        max_length=32,
        batch_size=32,
    ):

        result = [self.get_from_global_cache(inpt) for inpt in batch]
        pruned_batch = [inpt for inpt, res in zip(batch, result) if res is None]

        if pruned_batch:
            preds = self._exec_model_predict(
                pruned_batch,
                max_length=max_length,
                batch_size=batch_size,
            )

            preds_ptr = 0
            for i in range(len(result)):
                if result[i] is None:
                    result[i] = preds[preds_ptr]
                    preds_ptr += 1

        return result

    def predict_w_caching(
        self,
        inpts: list[str],
        max_length=32,
        batch_size=32,
        verbose=True,
    ) -> list[str]:
        result = []

        ln = len(inpts)
        bs = batch_size * 3

        if verbose:
            tot = math.ceil(ln / bs)
            pbar: tqdm[int] | range = tqdm(range(0, ln, bs), total=tot)
        else:
            pbar = range(0, ln, bs)

        for i in pbar:
            batch = inpts[i : i + bs]
            preds = self._predict_batch(
                batch, max_length=max_length, batch_size=batch_size
            )
            result.extend(preds)

        return result
