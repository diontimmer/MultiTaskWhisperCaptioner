from __future__ import annotations

import os
import sys
import pathlib
import functools
import dataclasses
import collections
import multiprocessing
from typing import Literal, Callable, Union, Tuple

import librosa
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torchdata.datapipes as dp
import transformers
import audiocap
import torchaudio
from col_ops import set_cols, del_cols, explode_col, rename_col
from preprocessing import PrepareLabels, PreprocessAudio


def create_prefix(task: str) -> str:
    return task + ": "


def load_and_process_audio(
    path: pathlib.Path, sr: int | None, audio_seconds: int = 25, mono: bool = True
) -> tuple[torch.Tensor, int]:
    try:
        audio, a_sr = torchaudio.load(path, num_frames=44100 * audio_seconds)
        if sr != a_sr:
            audio = torchaudio.transforms.Resample(a_sr, sr)(audio)
        if mono and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio.squeeze(), sr
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr, flush=True)
        return None, sr


@dataclasses.dataclass
class AudioFolder:
    shuffle: bool
    caption_columns: list[str]
    tokenizer: transformers.WhisperTokenizer
    feature_extractor: transformers.WhisperFeatureExtractor
    meta: pd.DataFrame = dataclasses.field(init=True)
    handle_multiple_captions: Literal["explode", "keep_first"] | None = None
    prepare_caption: Callable | None = None
    augment_config: audiocap.augment.AugmentConfig | None = None
    shuffle_buffer_size: int = 20
    prefetch: int = 10
    drop_audio_array: bool = True
    sample_n: int | None = None
    seed: int | None = None
    load_as_iterable: bool = True
    task: str = "tags"

    pipe: dp.iter.IterDataPipe | dp.map.MapDataPipe = dataclasses.field(init=False)
    augmenter: audiocap.augment.Augmenter | None = dataclasses.field(init=False)

    def __post_init__(self):
        if len(self.caption_columns) > 1 and self.handle_multiple_captions is None:
            raise ValueError(
                "Multiple caption columns found. "
                "Please specify how to handle them using `handle_multiple_captions`."
            )

        if self.meta.empty:
            raise ValueError("Metadata not found.")

        if self.sample_n is not None:
            self.meta = self.meta.sample(n=self.sample_n, random_state=self.seed)

        if self.shuffle:
            self.meta = self.meta.sample(frac=1, random_state=self.seed)

        if self.augment_config is not None:
            self.augmenter = audiocap.augment.Augmenter(self.augment_config)
        else:
            self.augmenter = None

        self.init_pipe()

    def init_pipe(self):
        prepare_labels = PrepareLabels(self.tokenizer)
        extract_features = PreprocessAudio(self.feature_extractor)
        sr = self.feature_extractor.sampling_rate

        pipe: dp.iter.IterDataPipe
        pipe = dp.iter.IterableWrapper(self.meta.to_dict("records"), deepcopy=False)

        pipe = pipe.sharding_filter().map(
            set_cols("path", lambda row: row["file_name"])
        )

        if self.augmenter is None:
            pipe = pipe.map(
                set_cols(
                    ("audio_array", "sampling_rate"),
                    lambda row: load_and_process_audio(row["path"], sr),
                )
            ).filter(lambda row: row["audio_array"] is not None)
        else:
            pipe = (
                pipe.map(
                    set_cols(
                        ("audio_array", "sampling_rate"),
                        lambda row: load_and_process_audio(row["path"], sr),
                    )
                )
                .filter(lambda row: row["audio_array"] is not None)
                .map(
                    set_cols(
                        "audio_array",
                        lambda row: self.augmenter(
                            row["audio_array"], row["sampling_rate"]
                        ),
                    )
                )
                .map(
                    set_cols(
                        "audio_array",
                        lambda row: torchaudio.transforms.Resample(
                            orig_freq=row["sampling_rate"], new_freq=sr
                        )(row["audio_array"]),
                    )
                )
                .map(set_cols("sampling_rate", lambda _: sr))
            )

        pipe = pipe.map(
            extract_features, ["audio_array", "sampling_rate"], "input_features"
        )
        pipe = pipe.map(del_cols("path"))

        prefix = create_prefix(self.task)

        if self.drop_audio_array:
            pipe = pipe.map(del_cols("audio_array"))

        if self.handle_multiple_captions == "explode":
            pipe = pipe.flatmap(
                explode_col(self.caption_columns, "caption", "caption_colname")
            )
        else:
            first_col, *rest_cols = self.caption_columns
            pipe = (
                pipe.map(rename_col({first_col: "caption"}))
                .map(set_cols("caption_colname", lambda _: first_col))
                .map(del_cols(rest_cols))
            )

        if self.prepare_caption is not None:
            pipe = pipe.map(self.prepare_caption, input_col="caption")

        if self.shuffle:
            pipe = pipe.shuffle(buffer_size=self.shuffle_buffer_size)

        pipe = pipe.map(set_cols("prefix", lambda row: prefix)).map(
            set_cols(
                ("labels", "forced_ac_decoder_ids"),
                lambda row: prepare_labels(prefix, row["caption"]),
            )
        )

        if self.load_as_iterable:
            self.pipe = pipe.prefetch(self.prefetch)
        else:
            self.pipe = pipe.enumerate().to_map_datapipe()

    def __len__(self):
        if len(self.caption_columns) == 1:
            return len(self.meta)
        if self.handle_multiple_captions == "keep_first":
            return len(self.meta)
        if self.handle_multiple_captions == "explode":
            return len(self.meta) * len(self.caption_columns)
        raise ValueError("Invalid value for `handle_multiple_captions`.")

    @functools.cached_property
    def alternative_captions(self) -> dict[str, list[str]]:
        if self.handle_multiple_captions == "explode":
            raise NotImplementedError(
                "Cannot return alternative captions when `handle_multiple_captions` is set to `flatten`."
            )
        caps = self.meta[self.caption_columns]
        if self.prepare_caption is not None:
            caps = caps.applymap(self.prepare_caption)
        caps = caps.set_index(self.caption_columns[0], drop=False)
        # this will drop values in case there are duplicates
        return {caption: alternatives for caption, *alternatives in caps.itertuples()}


@dataclasses.dataclass
class EncodedFolder:
    shuffle: bool
    encoded_base_path: str | None
    caption_columns: list[str]
    tokenizer: transformers.WhisperTokenizer
    meta: pd.DataFrame = dataclasses.field(init=True)
    handle_multiple_captions: Literal["explode", "keep_first"] | None = None
    prepare_caption: Callable | None = None
    shuffle_buffer_size: int = 20
    prefetch: int = 10
    drop_audio_array: bool = True
    sample_n: int | None = None
    seed: int | None = None
    load_as_iterable: bool = True
    create_metadata: bool = True
    task: str = "tags"
    pipe: dp.iter.IterDataPipe | dp.map.MapDataPipe = dataclasses.field(init=False)

    def __post_init__(self):
        if len(self.caption_columns) > 1 and self.handle_multiple_captions is None:
            raise ValueError(
                "Multiple caption columns found. "
                "Please specify how to handle them using `handle_multiple_captions`."
            )

        if self.meta.empty:
            raise ValueError("Metadata not found.")

        if self.sample_n is not None:
            self.meta = self.meta.sample(n=self.sample_n, random_state=self.seed)

        if self.shuffle:
            self.meta = self.meta.sample(frac=1, random_state=self.seed)

        self.init_pipe()

    def init_pipe(self):
        prepare_labels = PrepareLabels(self.tokenizer)

        def load_encoded_features(row):
            try:
                filename = os.path.splitext(os.path.basename(row["file_name"]))[0]
                encoded_filename = os.path.join(
                    self.encoded_base_path, f"{filename}_whisper.npy"
                )
                encoded = np.load(encoded_filename)
                encoded = encoded.squeeze()
                return encoded
            except Exception as e:
                print(
                    f"Error loading {os.path.splitext(os.path.basename(row['file_name']))[0]}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                return None

        pipe: dp.iter.IterDataPipe
        pipe = dp.iter.IterableWrapper(self.meta.to_dict("records"), deepcopy=False)

        pipe = pipe.sharding_filter().map(
            set_cols("path", lambda row: row["file_name"])
        )

        pipe = pipe.map(
            set_cols("input_features", lambda row: load_encoded_features(row))
        ).filter(lambda row: row["input_features"] is not None)

        pipe = pipe.map(del_cols("path"))

        prefix = create_prefix(self.task)

        if self.drop_audio_array:
            pipe = pipe.map(del_cols("audio_array"))

        if self.handle_multiple_captions == "explode":
            pipe = pipe.flatmap(
                explode_col(self.caption_columns, "caption", "caption_colname")
            )
        else:
            first_col, *rest_cols = self.caption_columns
            pipe = (
                pipe.map(rename_col({first_col: "caption"}))
                .map(set_cols("caption_colname", lambda _: first_col))
                .map(del_cols(rest_cols))
            )

        if self.prepare_caption is not None:
            pipe = pipe.map(self.prepare_caption, input_col="caption")

        if self.shuffle:
            pipe = pipe.shuffle(buffer_size=self.shuffle_buffer_size)

        pipe = pipe.map(set_cols("prefix", lambda row: prefix)).map(
            set_cols(
                ("labels", "forced_ac_decoder_ids"),
                lambda row: prepare_labels(prefix, row["caption"]),
            )
        )

        if self.load_as_iterable:
            self.pipe = pipe.prefetch(self.prefetch)
        else:
            self.pipe = pipe.enumerate().to_map_datapipe()

    def __len__(self):
        if len(self.caption_columns) == 1:
            return len(self.meta)
        if self.handle_multiple_captions == "keep_first":
            return len(self.meta)
        if self.handle_multiple_captions == "explode":
            return len(self.meta) * len(self.caption_columns)
        raise ValueError("Invalid value for `handle_multiple_captions`.")

    @functools.cached_property
    def alternative_captions(self) -> dict[str, list[str]]:
        if self.handle_multiple_captions == "explode":
            raise NotImplementedError(
                "Cannot return alternative captions when `handle_multiple_captions` is set to `flatten`."
            )
        caps = self.meta[self.caption_columns]
        if self.prepare_caption is not None:
            caps = caps.applymap(self.prepare_caption)
        caps = caps.set_index(self.caption_columns[0], drop=False)
        # this will drop values in case there are duplicates
        return {caption: alternatives for caption, *alternatives in caps.itertuples()}


def load_audios_for_prediction(
    src: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
    task: str,
    recursive: bool,
    suffixes: tuple[str, ...] = ("mp3", "wav"),
    take_n: int | None = None,
    prefetch: int = 10,
    encoded: bool = False,
    encoded_base_path: str | None = None,
) -> tuple[dp.iter.IterDataPipe, int]:

    src = pathlib.Path(src)

    if src.is_file():
        paths = [src]
    elif recursive:
        paths = [
            path
            for path in src.glob("**/*")
            if path.is_file() and path.suffix.strip(".") in suffixes
        ]
    else:
        paths = [
            path
            for path in src.iterdir()
            if path.is_file() and path.suffix.strip(".") in suffixes
        ]

    paths.sort()
    if take_n is not None:
        paths = paths[:take_n]

    num_files = len(paths)

    prepare_labels = PrepareLabels(tokenizer)
    extract_features = PreprocessAudio(feature_extractor)
    sr = feature_extractor.sampling_rate
    prefix = create_prefix(task)
    _, forced_ac_decoder_ids = prepare_labels(prefix, "")

    pipe: dp.iter.IterDataPipe
    pipe = dp.iter.IterableWrapper([{"path": path} for path in paths], deepcopy=False)

    def load_encoded_features(path):
        try:
            filename = os.path.splitext(os.path.basename(path))[0]
            encoded_filename = os.path.join(
                encoded_base_path, f"{filename}_whisper.npy"
            )
            encoded = np.load(encoded_filename)
            return encoded
        except Exception as e:
            print(f"Error loading {encoded_filename}: {e}", file=sys.stderr, flush=True)
            return None

    if encoded:
        pipe = (
            pipe.sharding_filter()
            .map(set_cols("file_name", lambda x: pathlib.Path(x["path"]).name))
            .map(
                set_cols(
                    "input_features", lambda row: load_encoded_features(row["path"])
                )
            )
            .filter(lambda row: row["input_features"] is not None)
            .map(del_cols("path"))
            .map(set_cols("forced_ac_decoder_ids", lambda _: forced_ac_decoder_ids))
            .prefetch(prefetch)
        )
    else:
        pipe = (
            pipe.sharding_filter()
            .map(set_cols("file_name", lambda x: pathlib.Path(x["path"]).name))
            .map(
                set_cols(
                    ("audio_array", "sampling_rate"),
                    lambda row: load_and_process_audio(row["path"], sr=sr),
                )
            )
            .map(del_cols("path"))
            .filter(lambda row: row["audio_array"] is not None)
            .map(extract_features, ["audio_array", "sampling_rate"], "input_features")
            .map(set_cols("forced_ac_decoder_ids", lambda _: forced_ac_decoder_ids))
            .prefetch(prefetch)
        )

    return pipe, num_files


def make_audiofolder(
    metadata: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
    augment_config: audiocap.augment.AugmentConfig,
    train_mini_size: int,
    val_mini_size: int,
    seed: int,
    encoded: bool,
    encoded_base_path: str,
    task: str,
) -> dict[str, AudioFolder]:

    ds = {}

    meta = pd.read_json(metadata, lines=True)

    # retrieve 500 samples for validation and 100 for test; then remove them from the training set
    val_metadata = meta.sample(n=500, random_state=seed)
    meta = meta.drop(val_metadata.index)
    test_metadata = meta.sample(n=100, random_state=seed)
    meta = meta.drop(test_metadata.index)
    # train_metadata is the rest
    train_metadata = meta

    if not encoded:

        common_args = dict(
            caption_columns=[
                "caption",
            ],
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )

        ds["train"] = AudioFolder(
            meta=train_metadata,
            handle_multiple_captions="explode",
            shuffle=True,
            augment_config=augment_config,
            task=task,
            **common_args,
        )

        ds["train_mini"] = AudioFolder(
            meta=train_metadata,
            handle_multiple_captions="keep_first",
            shuffle=False,
            augment_config=augment_config,
            sample_n=train_mini_size,
            drop_audio_array=False,
            load_as_iterable=False,
            seed=seed,
            task=task,
            **common_args,
        )

        ds["val"] = AudioFolder(
            meta=val_metadata,
            handle_multiple_captions="keep_first",
            shuffle=False,
            augment_config=None,
            sample_n=None,
            seed=seed,
            task=task,
            **common_args,
        )

        ds["val_mini"] = AudioFolder(
            meta=val_metadata,
            handle_multiple_captions="keep_first",
            shuffle=False,
            augment_config=None,
            sample_n=val_mini_size,
            drop_audio_array=False,
            load_as_iterable=False,
            seed=seed,
            task=task,
            **common_args,
        )

        ds["test"] = AudioFolder(
            meta=test_metadata,
            handle_multiple_captions="keep_first",
            shuffle=False,
            augment_config=None,
            task=task,
            **common_args,
        )

    else:
        common_args = dict(
            caption_columns=[
                "caption",
            ],
            tokenizer=tokenizer,
            encoded_base_path=encoded_base_path,
        )

        ds["train"] = EncodedFolder(
            meta=train_metadata,
            handle_multiple_captions="explode",
            shuffle=True,
            task=task,
            **common_args,
        )

        ds["train_mini"] = EncodedFolder(
            meta=train_metadata,
            handle_multiple_captions="keep_first",
            shuffle=False,
            sample_n=train_mini_size,
            load_as_iterable=False,
            seed=seed,
            task=task,
            **common_args,
        )

        ds["val"] = EncodedFolder(
            meta=val_metadata,
            handle_multiple_captions="keep_first",
            shuffle=False,
            sample_n=None,
            seed=seed,
            task=task,
            **common_args,
        )

        ds["val_mini"] = EncodedFolder(
            meta=val_metadata,
            handle_multiple_captions="keep_first",
            shuffle=False,
            sample_n=val_mini_size,
            load_as_iterable=False,
            seed=seed,
            task=task,
            **common_args,
        )

        ds["test"] = EncodedFolder(
            meta=test_metadata,
            handle_multiple_captions="keep_first",
            shuffle=False,
            task=task,
            **common_args,
        )
    return ds


def load_dataset_mixture(
    metadata: pathlib.Path | str,
    encoded: bool | False,
    encoded_base_path: str | None,
    log_preds_num_train: int,
    log_preds_num_valid: int,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
    augment_config: audiocap.augment.AugmentConfig,
    task: str,
):
    audiofolders: list[dict[str, audiocap.data.AudioFolder]] = []

    audiofolders.append(
        audiocap.data.make_audiofolder(
            metadata=metadata,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            augment_config=augment_config,
            train_mini_size=log_preds_num_train,
            val_mini_size=log_preds_num_valid,
            seed=0,
            encoded=encoded,
            encoded_base_path=encoded_base_path,
            task=task,
        )
    )

    if len(audiofolders) == 0:
        raise ValueError("No dataset specified")

    dataset = {}

    dataset["train"] = dp.iter.SampleMultiplexer(
        {af["train"].pipe.cycle(): 1.0 for af in audiofolders}
    )

    for split in ["val", "test"]:
        dataset[split] = dp.iter.Concater(*[af[split].pipe for af in audiofolders])
    for split in ["train_mini", "val_mini"]:
        dataset[split] = dp.map.Concater(*[af[split].pipe for af in audiofolders])

    ds_val_alternatives = {
        af["val"].task: af["val"].alternative_captions for af in audiofolders
    }

    return dataset, audiofolders, ds_val_alternatives


class DataCollatorAudioSeq2SeqWithPadding:

    def __init__(
        self,
        tokenizer: transformers.WhisperTokenizer,
        feature_extractor: transformers.WhisperFeatureExtractor,
        keep_cols: tuple[str, ...] = tuple(),
    ) -> None:
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.keep_cols = keep_cols

    def __call__(
        self,
        orig_batch: list[dict],
    ) -> collections.UserDict:

        batch_features = [{"input_features": x["input_features"]} for x in orig_batch]
        batch_forced_ac_decoder_ids = [x["forced_ac_decoder_ids"] for x in orig_batch]

        batch = self.feature_extractor.pad(batch_features, return_tensors="pt")
        batch["forced_ac_decoder_ids"] = torch.tensor(batch_forced_ac_decoder_ids)

        if "labels" in orig_batch[0]:
            batch_labels = [{"input_ids": x["labels"]} for x in orig_batch]
            batch_labels = self.tokenizer.pad(batch_labels, return_tensors="pt")
            # replace padding with -100 to ignore loss correctly
            labels = batch_labels["input_ids"].masked_fill(
                batch_labels.attention_mask != 1, -100
            )
            if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels

        for col in self.keep_cols:
            batch[col] = torch.utils.data.default_collate([x[col] for x in orig_batch])

        return batch


def find_corrupted_audios(
    folder: pathlib.Path | str, extension: str, num_workers: int
) -> list[pathlib.Path]:
    folder = pathlib.Path(folder)
    corrupted = []
    with multiprocessing.Pool(num_workers) as pool:
        files = list(folder.glob(f"**/*.{extension}"))
        print("found total files:", len(files))
        for path in files:
            if path.is_file():
                try:
                    pool.apply_async(librosa.load, args=(path,), kwds={"sr": None})
                except:
                    corrupted.append(path)
    print("found corrupted files:", len(corrupted))
    return corrupted
