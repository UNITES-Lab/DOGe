from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


@dataclass
class CustomDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    extra_keys_to_ignore: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "label" in features:
            features["labels"] = features["label"]
            del features["label"]
        if "label_ids" in features:
            features["labels"] = features["label_ids"]
            del features["label_ids"]

        features_to_ignore = {
            k: [item[k] for item in features] for k in self.extra_keys_to_ignore
        } if self.extra_keys_to_ignore else {}
        features = [
            {k: v for k, v in feature.items() if k not in self.extra_keys_to_ignore} for feature in features
        ] if self.extra_keys_to_ignore else features

        # take labels out of features
        if "labels" in features[0]:
            labels_batch = [{"input_ids": feature["labels"]} for feature in features] # Fake name for padding
            features = [{k: v for k, v in feature.items() if k != "labels"} for feature in features]
        else:
            labels_batch = None
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if labels_batch is not None:
            labels_batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                labels_batch,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if self.tokenizer.pad_token_id is not None and labels_batch is not None:
            labels_batch["input_ids"][labels_batch["input_ids"] == self.tokenizer.pad_token_id] = -100
            labels_batch["labels"] = labels_batch["input_ids"]
            del labels_batch["input_ids"]
        else:
            labels_batch = {}
        batch = {**batch, **features_to_ignore, **labels_batch}
        return batch
