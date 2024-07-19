from __future__ import annotations

import pathlib
import shutil
from typing import Optional, Any, List

import pandas as pd
import transformers
import wandb
import torch
import typer
import yaml
import peft

import audiocap.data
import audiocap.callbacks
import audiocap.models
import audiocap.augment

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    checkpoint_dir_root: pathlib.Path = typer.Option(
        ...,
        dir_okay=True,
        file_okay=False,
        readable=True,
        help="Path to the directory where checkpoints will be saved",
    ),
    # train file can be a list of jsonl files or a single jsonl file
    train_file: List[pathlib.Path] = typer.Option(
        ...,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="jsonl file with the training data",
    ),
    training_config: pathlib.Path = typer.Option(
        ...,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="yaml file with the training config",
    ),
    load_checkpoint: Optional[pathlib.Path] = typer.Option(
        None,
        dir_okay=True,
        file_okay=True,
        readable=True,
        help="Path to checkpoint to initialize the model with",
    ),
    encoded: bool = typer.Option(
        False,
        help="Whether the dataset is already encoded",
    ),
    wandb_group: Optional[str] = typer.Option(None, help="Wandb group"),
) -> None:

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    with open(training_config, "r") as f:
        training_config_dict: dict = yaml.safe_load(f)

    training_args_dict = training_config_dict["hf_training_args"]

    architecture_config = training_config_dict["architecture"]
    architecture_name = architecture_config["name"]
    use_pretrained_encoder = architecture_config["use_pretrained_whisper_encoder"]
    use_pretrained_decoder = architecture_config["use_pretrained_whisper_decoder"]

    early_stopping_config = training_config_dict["early_stopping"]
    should_early_stop = early_stopping_config["should_early_stop"]
    early_stopping_patience = early_stopping_config["early_stopping_patience"]
    early_stopping_threshold = early_stopping_config["early_stopping_threshold"]

    logging_config = training_config_dict["logging"]
    log_preds_every_n_steps = logging_config["log_preds_every_n_steps"]
    log_preds_num_train = logging_config["log_preds_num_train"]
    log_preds_num_valid = logging_config["log_preds_num_valid"]

    train_fc1_only = training_config_dict.get("train_fc1_only", False)
    train_using_peft = training_config_dict.get("peft_config", None) is not None
    peft_config_dict = training_config_dict.get("peft_config", {})
    clever_freeze = training_config_dict.get("clever_freeze", False)

    if train_fc1_only and clever_freeze:
        raise ValueError("Cannot train fc1 only and use clever freeze at the same time")

    if "augment" in training_config_dict:
        augment_config = audiocap.augment.AugmentConfig(
            **training_config_dict["augment"]
        )
    else:
        augment_config = None

    config = transformers.WhisperConfig.from_pretrained(architecture_name)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(
        architecture_name, language="en", task="transcribe"
    )
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        architecture_name
    )
    assert isinstance(config, transformers.WhisperConfig)
    model = get_whisper_model(
        architecture_name,
        config,
        load_checkpoint,
        use_pretrained_encoder,
        use_pretrained_decoder,
    )

    if train_using_peft:
        peft_config = peft.get_peft_config(peft_config_dict)
        model = peft.get_peft_model(model, peft_config)

    if train_fc1_only:
        for name, param in model.named_parameters():
            if "fc1" not in name:
                param.requires_grad = False

    if clever_freeze:
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True

        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv1d):
                for param in layer.parameters():
                    param.requires_grad = True

        for name, param in model.get_encoder().named_parameters():  # type: ignore
            if "fc1" in name:
                param.requires_grad = True

        for name, param in model.get_decoder().named_parameters():  # type: ignore
            if "encoder_attn" in name or "self_attn" in name or "fc1" in name:
                param.requires_grad = True

    tuned_params = sum(p.shape.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.shape.numel() for p in model.parameters())
    print(
        f"Number of trained parameters: {tuned_params}/{total_params} = {tuned_params/total_params*100:.2f}%"
    )

    metas = []

    for i, file in enumerate(train_file):
        metas.append(pd.read_json(file, lines=True))

    print("Task mapping:")
    for task_id, task in model.task_mapping.items():
        print(f"  {task}: {task_id}")

    dataset, audiofolders = audiocap.data.load_dataset_mixture(
        metas=metas,
        task_mapping=model.task_mapping,
        log_preds_num_train=log_preds_num_train,
        log_preds_num_valid=log_preds_num_valid,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        augment_config=augment_config,
        encoded=encoded,
    )

    for ds in audiofolders:
        for split_name, split in ds.items():
            print(f"{split_name}: {len(split)} audio-caption pairs")

    collator = audiocap.data.DataCollatorAudioSeq2SeqWithPadding(
        tokenizer, feature_extractor
    )

    log_config_dict = {
        key: val
        for key, val in training_config_dict.items()
        if key != "hf_training_args"
    }
    log_tags = [
        "supervised",
        architecture_name,
        f"trained_params_{tuned_params/total_params*100:.2f}%",
    ]

    if train_fc1_only:
        log_tags.append("fc1_only")
        log_config_dict["trained_params_percent"] = tuned_params / total_params
    if train_using_peft:
        log_tags.append("peft")
    if clever_freeze:
        log_tags.append("clever_freeze")

    wandb.init(
        project="audio-captioner",
        tags=log_tags,
        save_code=True,
        config=log_config_dict,
        group=wandb_group,
    )

    assert wandb.run is not None

    if train_using_peft and load_checkpoint is not None:
        # copy orig checkpoint
        shutil.copytree(
            load_checkpoint, checkpoint_dir_root / wandb.run.name / f"checkpoint-orig"
        )

    training_args_dict_preset: dict[str, Any] = {
        "output_dir": checkpoint_dir_root / wandb.run.name
    }
    training_args_dict = {**training_args_dict_preset, **training_args_dict}
    training_args = transformers.Seq2SeqTrainingArguments(**training_args_dict)

    callback_log_val_preds = audiocap.callbacks.PredictionLogger(
        log_prefix="val",
        dataset=dataset["val_mini"],
        collator=collator,
        log_every_n_steps=log_preds_every_n_steps,
        skip_special_tokens=False,
        log_to_wandb=True,
        log_to_stdout=True,
        log_to_file=f"logs/preds_during_training/{wandb.run.name}/predictions_val.jsonl",
        generate_kwargs={"max_length": training_args_dict["generation_max_length"]},
        encoded=encoded,
    )

    callback_log_train_preds = audiocap.callbacks.PredictionLogger(
        log_prefix="train",
        dataset=dataset["train_mini"],
        collator=collator,
        log_every_n_steps=log_preds_every_n_steps,
        skip_special_tokens=False,
        log_to_stdout=True,
        log_to_wandb=True,
        log_to_file=f"logs/preds_during_training/{wandb.run.name}/predictions_train.jsonl",
        generate_kwargs={"max_length": training_args_dict["generation_max_length"]},
        encoded=encoded,
    )

    callback_peft_checkpoint = audiocap.callbacks.SavePeftModelCallback()

    callbacks: list[transformers.TrainerCallback]
    callbacks = [
        callback_log_val_preds,
        callback_log_train_preds,
        callback_peft_checkpoint,
    ]

    if should_early_stop:
        if early_stopping_patience is None:
            raise ValueError(
                "early_stopping_patience must be specified if should_early_stop is True"
            )
        early_stopping_kwargs = dict(early_stopping_patience=early_stopping_patience)
        if early_stopping_threshold is not None:
            early_stopping_kwargs["early_stopping_threshold"] = early_stopping_threshold  # type: ignore
        early_stopping = transformers.EarlyStoppingCallback(**early_stopping_kwargs)
        callbacks.append(early_stopping)

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset["train"],
        args=training_args,
        callbacks=callbacks,
    )

    print("TRAINING")
    trainer.train()
    trainer.save_model(str(pathlib.Path(trainer.args.output_dir) / "final"))


def get_whisper_model(
    config_name: str,
    config: transformers.WhisperConfig,
    load_checkpoint: pathlib.Path | None,
    use_pretrained_whisper_encoder: bool,
    use_pretrained_whisper_decoder: bool,
) -> audiocap.WhisperForAudioCaptioning:

    if load_checkpoint is not None:
        load_checkpoint = pathlib.Path(load_checkpoint).resolve()
        model = audiocap.WhisperForAudioCaptioning.from_pretrained(load_checkpoint)
        assert isinstance(model, audiocap.WhisperForAudioCaptioning)
        return model

    if use_pretrained_whisper_encoder and use_pretrained_whisper_decoder:
        model = audiocap.WhisperForAudioCaptioning.from_pretrained(config_name)
        assert isinstance(model, audiocap.WhisperForAudioCaptioning)
        return model

    model_pretrained = audiocap.WhisperForAudioCaptioning.from_pretrained(config_name)
    assert isinstance(model_pretrained, audiocap.WhisperForAudioCaptioning)
    model = audiocap.WhisperForAudioCaptioning(config)

    if not use_pretrained_whisper_encoder:
        model_pretrained.model.encoder = model.get_encoder()

    if not use_pretrained_whisper_decoder:
        model_pretrained.model.decoder = model.get_decoder()

    del model
    return model_pretrained


if __name__ == "__main__":
    app()
