import numpy as np
import torchaudio
import torch

from transformers import WhisperFeatureExtractor, WhisperTokenizer
from typing import Tuple


class PrepareLabels:
    """Class to prepare labels for audio data."""

    def __init__(self, tokenizer: WhisperTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, prefix: str, caption: str) -> Tuple[list[int], list[int]]:
        forced_ac_decoder_ids = self.tokenizer(
            "", text_target=prefix, add_special_tokens=False
        ).labels
        fluff_tokens, eos = (
            self.tokenizer("", text_target="", add_special_tokens=True).labels[:-1],
            self.tokenizer("", text_target="", add_special_tokens=True).labels[-1],
        )
        labels = self.tokenizer(
            "", text_target=caption, add_special_tokens=False
        ).labels
        labels = fluff_tokens + forced_ac_decoder_ids + labels + [eos]
        return labels, forced_ac_decoder_ids


class PreprocessAudio:
    """Class to preprocess audio data."""

    def __init__(self, feature_extractor: WhisperFeatureExtractor) -> None:
        self.feature_extractor = feature_extractor
        self.num_features = feature_extractor.feature_size

    def __call__(self, audio_array: np.ndarray, sampling_rate: int) -> torch.Tensor:
        features = self.feature_extractor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features
        return features.reshape(self.num_features, -1)
