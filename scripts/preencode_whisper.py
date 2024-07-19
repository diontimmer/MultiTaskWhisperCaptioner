import os
from typing import List
import jsonlines
import numpy as np
import torchaudio
from transformers import WhisperFeatureExtractor
from tqdm import tqdm
import concurrent.futures
import json
import typer

INDUSTRY_STANDARD_SAMPLE_RATE = 44100  # worldstaaaarrrr


def process_and_save_audio(
    json_line, prog, extract_seconds, target_sample_rate, feature_extractor, output_dir
):
    try:
        if os.path.exists(
            os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(json_line['file_name']))[0]}_whisper.npy",
            )
        ):
            prog.update(1)
            return None
        # Load audio
        audio, sr = torchaudio.load(
            json_line["file_name"],
            num_frames=INDUSTRY_STANDARD_SAMPLE_RATE * extract_seconds,
        )
        audio = audio.mean(dim=0, keepdim=True).squeeze()

        if sr != target_sample_rate:
            audio = torchaudio.transforms.Resample(sr, target_sample_rate)(audio)
            sr = target_sample_rate

        features = feature_extractor(audio, sampling_rate=sr).input_features
        filename = os.path.splitext(os.path.basename(json_line["file_name"]))[0]
        encoded_path = os.path.join(output_dir, f"{filename}_whisper.npy")
        np.save(encoded_path, features)
        data = {}
        data["real"] = json_line["file_name"]
        json_filename = encoded_path.replace("_whisper.npy", "_whisper.json")
        with open(json_filename, "w") as f:
            json.dump(data, f)
        prog.update(1)
        return encoded_path
    except Exception as e:
        print(f"Error processing {json_line['file_path']}: {e}")
        prog.update(1)
        return None


def process_metadata_file(metadata_path):
    with jsonlines.open(metadata_path) as reader:
        lines = list(reader)
        print(f"Processing {len(lines)} files")

    prog = tqdm(total=len(lines))

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for line in lines:
            executor.submit(process_and_save_audio, line, prog)


def update_metadata_file(metadata_path, encoded_path):
    new_filename = metadata_path.replace(".jsonl", "_encoded.jsonl")
    encoded_jsons = [f for f in os.listdir(encoded_path) if f.endswith(".json")]
    with jsonlines.open(metadata_path) as reader:
        lines = list(reader)
        print(f"Processing {len(lines)} files")
    for encoded_json in encoded_jsons:
        with open(encoded_json) as f:
            data = json.load(f)
        for line in lines:
            if line["file_name"] == data["real"]:
                line["encoded_filename"] = encoded_json.replace(
                    "_whisper.json", "_whisper.npy"
                )
                break
    with jsonlines.open(new_filename, "w") as writer:
        for line in lines:
            writer.write(line)


# make typer app
app = typer.Typer(pretty_exceptions_enable=True)


def main(
    metadata_path: List[str] = typer.Option(
        ...,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to the metadata file",
    ),
    output_dir: str = typer.Option(
        ...,
        dir_okay=True,
        file_okay=False,
        exists=True,
        writable=True,
        help="Path to the folder where the encoded files will be saved",
    ),
    extract_seconds: int = typer.Option(
        25, help="Number of seconds to extract from the audio"
    ),
    target_sample_rate: int = typer.Option(
        16000, help="Target sample rate for the audio"
    ),
    device: str = typer.Option("cpu", help="Device to use for feature extraction"),
):
    # Initialize feature extractor
    feature_extractor = WhisperFeatureExtractor(device=device)

    paths = metadata_path

    for path in paths:
        print(f"Processing {path}")
        process_metadata_file(
            path, extract_seconds, target_sample_rate, feature_extractor, output_dir
        )
        update_metadata_file(path, output_dir)
