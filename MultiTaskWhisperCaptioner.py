import json
from audiocap.models import WhisperForAudioCaptioning
from audiocap.preprocessing import PrepareLabels
from transformers import WhisperTokenizer, WhisperFeatureExtractor
import torch
import os
import torchaudio
import typer
from typing import List


class MultiTaskWhisperCaptioner:
    def __init__(self, checkpoint, device="cuda", half_precision=True):
        self.device = device
        self.model, self.tokenizer, self.feature_extractor = self.load_model(checkpoint)
        self.label_preparer = PrepareLabels(self.tokenizer)
        self.model.to(self.device)
        self.task_mapping = self.model.named_task_mapping
        print(self.task_mapping)
        if half_precision:
            self.model.half()

    def load_model(self, checkpoint):
        print(f"Loading model from {checkpoint}")
        model = WhisperForAudioCaptioning.from_pretrained(checkpoint)
        tokenizer = WhisperTokenizer.from_pretrained(
            checkpoint, language="en", task="transcribe"
        )
        feature_extractor = WhisperFeatureExtractor(
            device=self.device, sampling_rate=16000
        )
        return model, tokenizer, feature_extractor

    def load_and_preprocess_audio(self, file_path):
        audio, sr = torchaudio.load(file_path, num_frames=44100 * 25)
        if audio.shape[0] == 2:
            audio = torch.mean(audio, dim=0, keepdim=True)
        resampler = torchaudio.transforms.Resample(
            sr, self.feature_extractor.sampling_rate
        )
        audio = resampler(audio)
        audio = audio.numpy()
        audio = audio.squeeze()

        return self.feature_extractor(
            audio,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
        ).input_features

    def prepare_caption_style(self, style_prefix):
        style_prefix_tokens = self.tokenizer(
            "", text_target=style_prefix, return_tensors="pt", add_special_tokens=False
        ).labels
        return style_prefix_tokens

    def process_audio_files(self, audio_files, tasks=["tags"], batch_size=1):

        if not isinstance(audio_files, list):
            audio_files = [audio_files]

        # batch size cant be greater than number of audio files, if so reduce it
        batch_size = min(batch_size, len(audio_files))

        predictions = {}
        self.model.to(self.device)

        def batch_iterable(iterable, n):
            length = len(iterable)
            for idx in range(0, length, n):
                yield iterable[idx : min(idx + n, length)]

        for task in tasks:
            prefix = f"{self.task_mapping[task]}: "
            _, forced_ac_decoder_ids = self.label_preparer(prefix, "")
            forced_ac_decoder_ids = torch.tensor(forced_ac_decoder_ids).to(self.device)
            if forced_ac_decoder_ids.dim() == 1:
                forced_ac_decoder_ids = forced_ac_decoder_ids.unsqueeze(0)

            for batch_files in batch_iterable(audio_files, batch_size):
                total_features = []

                for file_path in batch_files:
                    features = self.load_and_preprocess_audio(file_path)
                    input_features = features.to(torch.float16).to(self.device)
                    total_features.append(input_features)

                total_features = torch.cat(total_features, dim=0)
                total_forced_ac_decoder_ids = forced_ac_decoder_ids.expand(
                    (total_features.shape[0], -1)
                )

                preds_tokens = self.model.generate(
                    input_features=total_features,
                    forced_ac_decoder_ids=total_forced_ac_decoder_ids,
                )

                pred_texts = self.tokenizer.batch_decode(
                    preds_tokens, skip_special_tokens=True
                )
                for pred_text, file_path in zip(pred_texts, batch_files):
                    if file_path not in predictions:
                        predictions[file_path] = {}
                    predictions[file_path][task] = pred_text.replace(prefix, "")

        return predictions


def main(
    checkpoint: str = typer.Option(..., help="Path to model checkpoint"),
    audio: str = typer.Option(..., help="Path to audio file(s)"),
    output: str = typer.Option(
        ..., help="Output type ('print' or 'file') for processed audio"
    ),
    device: str = typer.Option("cuda", help="Device to run the model on"),
    task: List[str] = typer.Option(
        ["tags"], help="List of tasks to generate captions for"
    ),
    batch_size: int = typer.Option(1, help="Batch size for processing audio files"),
):
    generator = MultiTaskWhisperCaptioner(checkpoint=checkpoint, device=device)

    if os.path.isdir(audio):
        audiofiles = [os.path.join(audio, f) for f in os.listdir(audio)]
    else:
        audiofiles = [audio]

    captions = generator.process_audio_files(
        audiofiles, tasks=task, batch_size=batch_size
    )
    for audiofile, caption in captions.items():
        if output == "file":
            with open(audiofile + "_captions.json", "w") as file:
                json.dump(caption, file)
        elif output == "print":
            print(f"[{audiofile}] {json.dumps(caption, indent=2)}")


if __name__ == "__main__":
    typer.run(main)
