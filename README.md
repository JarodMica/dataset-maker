# Dataset Maker
Multi-purpose dataset maker for various TTS models.
- Tortoise TTS/XTTS
- StyleTTS 2 ~ [Webui](https://github.com/JarodMica/StyleTTS-WebUI)
- Higgs Audio ~ [Base](https://github.com/JimmyMa99/train-higgs-audio) - [My fork](https://github.com/JarodMica/higgs-audio/tree/training)
- VibeVoice ~ [Base](https://github.com/voicepowered-ai/VibeVoice-finetuning) - [My fork](https://github.com/JarodMica/VibeVoice-finetuning)
- IndexTTS 2 ~ [My Trainer](https://github.com/JarodMica/index-tts/tree/training_v2)

## What does it output?
**Tortoise, StyleTTS2, XTTS** - Models like these take in a simple text file where audio:text pairs are sorted something like:
 - `path/to/audio/file | transcription`
 
 **Folder Sturcutre**
 ```bash
Dataset_name
- train.txt
-- seg1.wav
-- seg2.wav
 ```

**Higgs Audio** has a main metadata.json that includes all of the information and instructions for how to train on audio files, broken down by .txt files and .wav.

**Folder Structure**
```bash
Dataset_name
- metadata.json
- some_audio_1.txt
- some_audio_1.wav
- some_audio_2.txt
- some_audio_2.wav
```

**Vibe Voice** has a main `.jsonl` file that contains individual json entries with text and audio keys. It always prepends "Speaker 0: " before each transcription in accordance with what the trainer is expecting.
 - `{"text": "Speaker 0: some transcription", "audio": "path/to/audio"}`

**Folder Structure**
```bash
Dataset_name
- <project_name>_train.jsonl
- vibevoice_000000.wav
- vibevoice_000001.wav
```

## Installation (Windows)
1. Make sure you have astral uv installed on your PC
2. Run the following:
    ```bash
    git clone https://github.com/JarodMica/dataset-maker.git
    cd dataset-maker
    uv sync
    ```
3. uv should handle the installation of all packages and versioning. Once it finishes running, launch the gradio with:
    ```
    uv run .\gradio_interface.py
    ```
