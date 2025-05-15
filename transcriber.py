# transcriber.py

import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import pysrt
import whisperx
import gc
import time

# Import the Slicer class from slicer2.py
from slicer2 import Slicer

def load_whisperx_model(model_name="large-v2"):
    """Load and return the WhisperX model on CUDA (float16)."""
    asr_options = {"initial_prompt": '"No! We must not be pushed back," he replied back. "Keep on fighting to your very deaths!"'}
    return whisperx.load_model(model_name, device="cuda", compute_type="float16", asr_options=asr_options)

def run_whisperx_transcription(audio_path, output_dir, language="en", chunk_size=20, no_align=False, model=None, batch_size=16):
    print(f"DEBUG: Running WhisperX transcription on {audio_path}...")
    audio = whisperx.load_audio(str(audio_path))
    if language == "None":
        result = model.transcribe(audio=audio, chunk_size=chunk_size, batch_size=batch_size)
    else:
        result = model.transcribe(audio=audio, language=language, chunk_size=chunk_size, batch_size=batch_size)
    if not no_align:
        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
        result = whisperx.align(result["segments"], align_model, metadata, audio, device="cuda", return_char_alignments=False)
    if "language" not in result:
        result["language"] = language
    srt_writer = whisperx.utils.get_writer("srt", str(output_dir))
    srt_writer(result, str(output_dir), {"max_line_width": None, "max_line_count": None, "highlight_words": False})
    srt_files = list(output_dir.glob("*.srt"))
    if not srt_files:
        raise FileNotFoundError("No SRT file generated.")
    print(f"DEBUG: WhisperX produced SRT file {srt_files[0]}")
    return srt_files[0]

def stitch_segments(segment_files, sr, silence_duration_sec=10):
    """
    Stitch segments from the list of WAV files by concatenating their NumPy arrays with a silent gap in between.
    Returns the stitched NumPy array.
    """
    # Sort files numerically (e.g. seg1.wav, seg2.wav, ...)
    segment_files = sorted(segment_files, key=lambda x: int(x.stem.replace("seg", "")))
    
    # Load the first segment to determine shape
    first_data, _ = sf.read(str(segment_files[0]), dtype='float32')
    if first_data.ndim == 1:
        silence = np.zeros(int(sr * silence_duration_sec), dtype=np.float32)
    else:
        num_channels = first_data.shape[1]
        silence = np.zeros((int(sr * silence_duration_sec), num_channels), dtype=np.float32)
    
    stitched = []
    for seg_file in segment_files:
        data, _ = sf.read(str(seg_file), dtype='float32')
        stitched.append(data)
        stitched.append(silence)
        print(f"DEBUG: Added segment {seg_file.name} and {silence_duration_sec} sec silence.")
    if stitched:
        stitched = stitched[:-1]  # Remove the final silence gap
    return np.concatenate(stitched)

def map_srt_to_segments(srt_file, seg_boundaries):
    """
    Given an SRT file (for the stitched audio) and a list of (start, end) times (in seconds)
    for each segment, assign each subtitle to the segment based on the segment's start time.
    
    Instead of using the segment midpoint, this version finds for each subtitle the
    segment whose start time is the closest preceding (or equal) time.
    
    Returns a list of transcript strings (one per segment).
    """
    print("DEBUG: Mapping SRT subtitles to segment boundaries.")
    subs = pysrt.open(str(srt_file))
    transcripts = ["" for _ in seg_boundaries]
    
    # Extract the start times of each segment.
    seg_starts = [start for start, _ in seg_boundaries]
    for idx, (start, end) in enumerate(seg_boundaries):
        print(f"DEBUG: Segment {idx+1} boundaries: start={start:.3f}, end={end:.3f}")
    
    for sub in subs:
        # Convert subtitle start time to seconds.
        t = (sub.start.hours * 3600 +
             sub.start.minutes * 60 +
             sub.start.seconds +
             sub.start.milliseconds / 1000.0)
        
        # Find the last segment whose start time is <= t.
        candidate_idx = 0
        for i, seg_start in enumerate(seg_starts):
            if t >= seg_start:
                candidate_idx = i
            else:
                break
        
        print(f"DEBUG: Subtitle '{sub.text}' starting at {t:.3f} sec assigned to segment {candidate_idx+1} (segment start: {seg_starts[candidate_idx]:.3f}).")
        transcripts[candidate_idx] += " " + sub.text
    # Clean up extra whitespace.
    transcripts = [t.strip() for t in transcripts]
    for idx, transcript in enumerate(transcripts):
        print(f"DEBUG: Final transcript for segment {idx+1}: {transcript}")
    return transcripts

def process_audio_file(audio_file, model, output_base, train_txt_path, silence_duration_sec=3,
                       slicer_params=None, purge_long_segments=False, max_segment_length=12,
                       starting_index=1, language="en"):
    """
    Process one audio file:
      - Creates a subfolder under output_base.
      - Loads the audio and uses slicer2.py's Slicer to split it.
      - Saves each segment (naming them with a continuous counter) and records its duration.
      - Stitches kept segments together (with a silence gap) and runs WhisperX on the stitched audio.
      - Computes segment boundaries and maps SRT subtitles.
      - Appends one train.txt entry per kept segment using unique segment names.
      
    Returns the next available segment index after processing this audio file.
    """
    print(f"DEBUG: Processing {audio_file}...")
    subfolder = output_base / audio_file.stem
    subfolder.mkdir(parents=True, exist_ok=True)

    # Load audio using librosa.
    print("DEBUG: Loading audio with librosa...")
    y, sr = librosa.load(str(audio_file), sr=None, mono=False)

    if slicer_params is None:
        slicer_params = {
            'sr': sr,
            'threshold': -40.0,
            'min_length': 7000,
            'min_interval': 1000,
            'hop_size': 20,
            'max_sil_kept': 500
        }
    slicer = Slicer(**slicer_params)

    print("DEBUG: Slicing audio...")
    segments_np = slicer.slice(y)
    if not segments_np:
        print(f"DEBUG: No segments found for {audio_file}. Skipping.")
        return starting_index

    seg_durations = []
    segment_files = []
    current_index = starting_index
    for i, seg in enumerate(segments_np):
        if seg.ndim > 1:
            duration = seg.shape[1] / sr
            seg_to_write = seg.T
        else:
            duration = len(seg) / sr
            seg_to_write = seg

        if purge_long_segments and (duration > max_segment_length):
            print(f"DEBUG: Skipping segment {i+1} because duration {duration:.3f} sec exceeds max allowed {max_segment_length} sec.")
            continue
        if duration < 1:
            print(f"DEBUG: Skipping segment {i+1} because duration {duration:.3f} sec is less than 1 sec.")
            continue

        seg_filename = subfolder / f"seg{current_index}.wav"
        current_index += 1
        seg_durations.append(duration)
        sf.write(str(seg_filename), seg_to_write, sr)
        print(f"DEBUG: Saved segment {current_index-1} to {seg_filename} with duration {duration:.3f} sec")
        segment_files.append(seg_filename)

    if not segment_files:
        print(f"DEBUG: No segments remaining after purging for {audio_file}. Skipping transcription.")
        return current_index

    print("DEBUG: Stitching segments with silence gap...")
    stitched_array = stitch_segments(segment_files, sr, silence_duration_sec)
    stitched_path = subfolder / "stitched.wav"
    sf.write(str(stitched_path), stitched_array, sr)
    print(f"DEBUG: Saved stitched audio: {stitched_path}")

    whisperx_out_dir = subfolder / "whisperx_output"
    whisperx_out_dir.mkdir(exist_ok=True)

    srt_file = run_whisperx_transcription(stitched_path, whisperx_out_dir, language=language,
                                            chunk_size=8, no_align=True, model=model)

    seg_boundaries = []
    current_time = 0.0
    for duration in seg_durations:
        seg_boundaries.append((current_time, current_time + duration))
        print(f"DEBUG: Calculated segment boundary: start={current_time:.3f}, end={current_time + duration:.3f}")
        current_time = current_time + duration + silence_duration_sec

    segment_transcripts = map_srt_to_segments(srt_file, seg_boundaries)

    with train_txt_path.open("a", encoding="utf-8") as f:
        for idx, transcript in enumerate(segment_transcripts):
            seg_number = starting_index + idx
            seg_filename = f"seg{seg_number}.wav"
            if len(transcript.strip()) < 2:
                print(f"DEBUG: Skipping segment {seg_number} because transcript is too short: {transcript}")
                continue
            f.write(f"{seg_filename} | {transcript}\n")
            print(f"DEBUG: Added dataset entry for {seg_filename} with transcript: {transcript}")

    return current_index


def main():
    # Let the user select the folder containing long audio files.
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Folder with Audio Files")
    if not folder_selected:
        print("DEBUG: No folder selected. Exiting.")
        return
    
    audio_dir = Path(folder_selected)
    
    # Instead of writing to the chosen folder, output to a folder named output_{suffix} in the current directory.
    suffix = input("Enter output suffix (for output_{suffix} folder): ").strip() or "processed"
    start_time = time.time()
    output_base = Path.cwd() / f"output_{suffix}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Define the path to the dataset file (train.txt)
    train_txt_path = output_base / "train.txt"
    
    print("DEBUG: Loading WhisperX model (large-v3)...")
    model = load_whisperx_model("large-v3")
    
    audio_extensions = (".wav", ".mp3", ".m4a", ".opus", ".webm", ".mp4")
    segment_counter = 1  # Global counter for segments across all files.
    for audio_file in audio_dir.iterdir():
        if audio_file.suffix.lower() in audio_extensions:
            try:
                segment_counter = process_audio_file(audio_file, model, output_base, train_txt_path,
                                                       silence_duration_sec=3, starting_index=segment_counter)
            except Exception as e:
                print(f"DEBUG: Error processing {audio_file}: {e}")
            gc.collect()

    print(f"DEBUG: Dataset creation complete. See {train_txt_path}")
    end_time = time.time()
    print(f"DEBUG: Total Time: {end_time - start_time}")

if __name__ == "__main__":
    main()
