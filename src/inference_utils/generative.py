from tqdm import tqdm
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizer,
    AutoTokenizer,
)
import torch
from pathlib import Path


class GenerativeModel:

    def __init__(
        self,
        model: BartForConditionalGeneration | str,
        tokenizer: PreTrainedTokenizer | str,
        device="cuda",
        verbose=True,
    ):
        self.device = device

        if isinstance(model, str) or isinstance(model, Path):
            self.model = BartForConditionalGeneration.from_pretrained(model).to(
                self.device
            )
        else:
            self.model = model.to(self.device)

        self.model.eval()

        if isinstance(tokenizer, str) or isinstance(model, Path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.verbose = verbose

    def predict(
        self,
        texts: list[str],
        max_length: int = 32,
        batch_size: int = 32,
    ) -> list[str]:

        return predict(
            texts=texts,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=max_length,
            batch_size=batch_size,
            verbose=self.verbose,
        )


def predict(
    texts: list[str],
    model,
    tokenizer,
    device,
    max_length: int = 70,
    batch_size: int = 32,
    verbose=True,
) -> list[str]:

    preds = []

    if verbose:
        pbar = tqdm(
            range(0, len(texts), batch_size),
            total=int(len(texts) / batch_size),
        )
    else:
        pbar = range(0, len(texts), batch_size)

    for i in pbar:
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch, max_length=70, truncation=True, padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,
                temperature=1,
                early_stopping=True,
            )

        cur_preds = tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        preds.extend(cur_preds)

    return preds
