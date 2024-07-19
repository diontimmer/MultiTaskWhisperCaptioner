# Whisper Multitask Audio Captioner

This repository contains the code for the Whisper Multitask Audio Captioner, a transformer encoder-decoder model for automatic audio captioning. As opposed to speech-to-text, captioning describes the content and features of audio clips.
## Overview

The model is based on the Whisper architecture and is capable of generating captions for various audio-related tasks such as general descriptions, musical genres, emotional feelings, audio pace, and usage themes.

## Features

    Model Architecture: Whisper encoder-decoder transformer.
    Based on: https://github.com/prompteus/audio-captioning
    Supported Tasks:
        General descriptions
        Musical genres
        Emotional feelings
        Audio pace and expression
        Audio usage themes

## Usage

The model expects an audio clip (up to 30s) to the encoder as an input and information about caption style as forced prefix to the decoder.
The forced prefix is an integer which is mapped to various tasks. This mapping is defined in the model config and can be retrieved with a function.

The tag mapping of the current model is:

| Task     | ID | Description                                            |
| -------- | -- | ------------------------------------------------------ |
| tags     | 0  | General descriptions, can include genres and features. |
| genre    | 1  | Estimated musical genres.                              |
| mood     | 2  | Estimated emotional feeling.                           |
| movement | 3  | Estimated audio pace and expression.                   |
| theme    | 4  | Estimated audio usage (not very accurate)              |

## Python API Usage:

```python
device = "cuda"
checkpoint = "DionTimmer/whisper-small-multitask-analyzer"
audiofiles = [...] # List of paths.
tasks = [...] # List of tasks as strings (tags, mood etc.) to generate.
batch_size = 1 # Process multiple files at once.
return_as_tags = True # Return as list; if False returns as single, comma-separated, string.

generator = MultiTaskWhisperCaptioner(checkpoint=checkpoint, device=device)

captions = generator.process_audio_files(
  audiofiles, tasks=tasks, batch_size=batch_size, return_as_tags=return_as_tags
)
```

## Command Line Usage

You can also run the inference directly from the command line using typer.

```bash
python MultiTaskWhisperCaptioner.py \
--checkpoint "DionTimmer/whisper-small-multitask-analyzer" \
--audio "path/to/audio/file" \
--output "print" \
--device "cuda" \
--task tags \
--task mood
```

This command will load the model, preprocess the audio file, and print the generated captions to the console.

See ```python MultiTaskWhisperCaptioner.py -h``` for more options.
