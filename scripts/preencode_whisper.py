import os
import jsonlines
import numpy as np
import torchaudio
from transformers import WhisperFeatureExtractor
from tqdm import tqdm
import concurrent.futures

# Initialize feature extractor
feature_extractor = WhisperFeatureExtractor(device="cpu")

# Load metadata
output_dir = "/data/datasets/pixabay_encoded"
paths = [
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_mood_val.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_mood_test.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_mood_train.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_movement_val.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_movement_test.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_movement_train.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_theme_val.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_theme_test.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_theme_train.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_tags_val.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_tags_test.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_tags_train.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_genre_val.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_genre_test.jsonl",
    "/chungus/disk1/datasets/pixabay/captioning_jsons/metadata_genre_train.jsonl",
]
extract_seconds = 25
target_sample_rate = 16000


def process_and_save_audio(json_line, prog):
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
            json_line["file_name"], num_frames=44100 * extract_seconds
        )
        audio = audio.mean(dim=0, keepdim=True).squeeze()

        if sr != target_sample_rate:
            audio = torchaudio.transforms.Resample(sr, target_sample_rate)(audio)
            sr = target_sample_rate

        features = feature_extractor(audio, sampling_rate=sr).input_features
        filename = os.path.splitext(os.path.basename(json_line["file_name"]))[0]
        encoded_path = os.path.join(output_dir, f"{filename}_whisper.npy")
        np.save(encoded_path, features)
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


for path in paths:
    print(f"Processing {path}")
    process_metadata_file(path)
