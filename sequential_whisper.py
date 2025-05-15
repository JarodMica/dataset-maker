import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import whisperx
import gc
import time

# Import the Slicer class from slicer2.py
from slicer2 import Slicer

def load_whisperx_model(model_name="large-v3"):
    """Load and return the WhisperX model on CUDA (float16)."""
    asr_options = {"initial_prompt": '"Now, umm.. watch uhh.. this video," she said with hesitation. "No! Never, I do not want to!" he replied back.'}
    return whisperx.load_model(model_name, device="cuda", compute_type="float16", asr_options=asr_options)

def transcribe_segment(segment_path, model, language="en", chunk_size=10):
    """
    Transcribe a single segment using the WhisperX model.
    Loads the audio from the segment file, transcribes it, and returns the transcript text.
    """
    # Load the audio (using whisperx's load_audio utility)
    audio = whisperx.load_audio(str(segment_path))
    # Run transcription (without alignment or SRT generation)
    result = model.transcribe(audio=audio, language=language, chunk_size=chunk_size)
    # Return the full transcript text from the result dictionary
    return result['segments'][0].get("text", "").strip()

def process_audio_file(audio_file, model, output_base, train_txt_path, slicer_params=None):
    """
    Process one audio file by:
      - Creating a subfolder under output_base.
      - Loading the audio and using Slicer to split it into segments.
      - Saving each segment as seg1.wav, seg2.wav, etc.
      - Transcribing each segment individually using WhisperX.
      - Appending a line to train.txt for each segment.
    """
    print(f"Processing {audio_file}...")
    subfolder = output_base / audio_file.stem
    subfolder.mkdir(parents=True, exist_ok=True)
    
    # Load audio using librosa.
    y, sr = librosa.load(str(audio_file), sr=None, mono=False)
    
    # Set slicer parameters if not provided.
    if slicer_params is None:
        slicer_params = {
            'sr': sr,
            'threshold': -40.0,
            'min_length': 7000,    # milliseconds
            'min_interval': 1000,  # milliseconds
            'hop_size': 20,        # milliseconds
            'max_sil_kept': 500    # milliseconds
        }
    slicer = Slicer(**slicer_params)
    
    segments_np = slicer.slice(y)
    if not segments_np:
        print(f"No segments found for {audio_file}. Skipping.")
        return

    # Open train.txt for appending transcript entries.
    with train_txt_path.open("a", encoding="utf-8") as f:
        for i, seg in enumerate(segments_np):
            seg_filename = subfolder / f"seg{i+1}.wav"
            # If multi-channel, transpose the data for writing.
            seg_to_write = seg.T if seg.ndim > 1 else seg
            sf.write(str(seg_filename), seg_to_write, sr)
            print(f"Saved segment {i+1} to {seg_filename}")
            
            # Transcribe the segment
            transcript = transcribe_segment(seg_filename, model, language="en", chunk_size=10)
            # Append the filename and transcript to train.txt
            f.write(f"seg{i+1}.wav | {transcript}\n")
            print(f"Added dataset entry for seg{i+1}.wav")

def main():
    # Let the user select the folder containing audio files.
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Folder with Audio Files")
    if not folder_selected:
        print("No folder selected. Exiting.")
        return
    audio_dir = Path(folder_selected)
    
    # Create an output folder with a user-specified suffix.
    suffix = input("Enter output suffix (for output_{suffix} folder): ").strip() or "processed"
    output_base = Path.cwd() / f"output_{suffix}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Define the path to the dataset file (train.txt)
    train_txt_path = output_base / "train.txt"
    
    print("Loading WhisperX model (large-v3)...")
    model = load_whisperx_model("large-v2")
    
    # Start the timer
    start_time = time.time()
    
    audio_extensions = (".wav", ".mp3", ".m4a", ".opus", ".webm", ".mp4")
    for audio_file in audio_dir.iterdir():
        if audio_file.suffix.lower() in audio_extensions:
            try:
                process_audio_file(audio_file, model, output_base, train_txt_path)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
            gc.collect()
    
    # End the timer and print the elapsed time.
    total_time = time.time() - start_time
    print(f"Dataset creation complete. See {train_txt_path}")
    print(f"Total processing time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
