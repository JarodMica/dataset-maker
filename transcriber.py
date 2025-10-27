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

SILENCE_SLICE_METHOD = "silence"
WHISPERX_SLICE_METHOD = "whisperx"
VALID_SLICE_METHODS = {SILENCE_SLICE_METHOD, WHISPERX_SLICE_METHOD}

# WhisperX VAD refinement defaults
WHISPERX_VAD_WINDOW_SEC = 0.03  # 30ms
WHISPERX_VAD_HOP_SEC = 0.01     # 10ms
WHISPERX_VAD_THRESHOLD_DB = -45.0
WHISPERX_VAD_MAX_EXPAND_SEC = 0.4
WHISPERX_VAD_SILENCE_FRAMES = 2
WHISPERX_VAD_MIN_GAP_SEC = 0.02
WHISPERX_VAD_TRAIL_BUFFER_SEC = 0.2

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


def _compute_rms_envelope(samples, sr):
    frame_length = max(int(round(WHISPERX_VAD_WINDOW_SEC * sr)), 1)
    hop_length = max(int(round(WHISPERX_VAD_HOP_SEC * sr)), 1)
    if frame_length <= hop_length:
        frame_length = hop_length * 2
    rms = librosa.feature.rms(y=samples, frame_length=frame_length, hop_length=hop_length)[0]
    times = (np.arange(len(rms)) * hop_length + frame_length / 2) / sr
    rms = np.maximum(rms, 1e-8)
    rms_db = librosa.amplitude_to_db(rms)
    return times, rms_db


def _refine_segment_boundaries(start, end, prev_end, next_start, rms_times, rms_db, audio_duration):
    if rms_times.size == 0:
        return max(start, prev_end), min(end, audio_duration)

    threshold = WHISPERX_VAD_THRESHOLD_DB
    max_expand = WHISPERX_VAD_MAX_EXPAND_SEC
    silence_frames = max(int(WHISPERX_VAD_SILENCE_FRAMES), 1)

    min_start_time = max(prev_end, start - max_expand, 0.0)
    max_end_time = min(audio_duration, end + max_expand)
    if next_start is not None:
        max_end_time = min(max_end_time, next_start)

    # Expand backwards
    start_idx = np.searchsorted(rms_times, start, side="left")
    candidate_start = start
    found_voice = False
    consecutive_below = 0
    idx = start_idx
    while idx > 0:
        idx -= 1
        t = rms_times[idx]
        if t < min_start_time:
            break
        if rms_db[idx] > threshold:
            candidate_start = t
            found_voice = True
            consecutive_below = 0
        else:
            if found_voice:
                consecutive_below += 1
                if consecutive_below >= silence_frames:
                    break
    refined_start = max(candidate_start, min_start_time, prev_end)

    # Expand forwards
    end_idx = np.searchsorted(rms_times, end, side="right")
    candidate_end = end
    found_voice = False
    consecutive_below = 0
    idx = end_idx
    while idx < rms_times.size:
        t = rms_times[idx]
        if t > max_end_time:
            break
        if rms_db[idx] > threshold:
            candidate_end = t
            found_voice = True
            consecutive_below = 0
        else:
            if found_voice:
                consecutive_below += 1
                if consecutive_below >= silence_frames:
                    break
        idx += 1
    refined_end = min(max(candidate_end, refined_start), max_end_time, audio_duration)

    if next_start is not None:
        max_allowed = max(next_start - WHISPERX_VAD_MIN_GAP_SEC, refined_start)
        refined_end = min(refined_end, max_allowed)

    if WHISPERX_VAD_TRAIL_BUFFER_SEC > 0:
        buffered_end = min(refined_end + WHISPERX_VAD_TRAIL_BUFFER_SEC, audio_duration, max_end_time)
        if next_start is not None:
            buffered_end = min(buffered_end, max(next_start - WHISPERX_VAD_MIN_GAP_SEC, refined_start))
        if buffered_end > refined_end:
            refined_end = buffered_end

    if refined_end - refined_start < 0.01:
        refined_start = max(start, prev_end)
        refined_end = min(end, audio_duration)
        if next_start is not None:
            refined_end = min(refined_end, max(next_start - WHISPERX_VAD_MIN_GAP_SEC, refined_start))

    return refined_start, refined_end

def _slice_audio_with_silence(audio_file, model, subfolder, y, sr, silence_duration_sec,
                              slicer_params, purge_long_segments, max_segment_length,
                              verbose_mode, starting_index, language, chunk_size, batch_size):
    print("DEBUG: Using silence-based slicing.")
    slicer = Slicer(**slicer_params)

    print("DEBUG: Slicing audio...")
    segments_np = slicer.slice(y)
    if not segments_np:
        print(f"DEBUG: No segments found for {audio_file}. Skipping.")
        return [], starting_index

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
            if not verbose_mode:
                print(f"DEBUG: Skipping segment {i+1} because duration {duration:.3f} sec is less than 1 sec.")
                continue
            print(f"DEBUG: Keeping short segment {i+1} (duration {duration:.3f} sec) due to verbose mode.")

        seg_filename = subfolder / f"seg{current_index}.wav"
        current_index += 1
        seg_durations.append(duration)
        sf.write(str(seg_filename), seg_to_write, sr)
        print(f"DEBUG: Saved segment {current_index-1} to {seg_filename} with duration {duration:.3f} sec")
        segment_files.append(seg_filename)

    if not segment_files:
        print(f"DEBUG: No segments remaining after purging for {audio_file}. Skipping transcription.")
        return [], current_index

    print("DEBUG: Stitching segments with silence gap...")
    stitched_array = stitch_segments(segment_files, sr, silence_duration_sec)
    stitched_path = subfolder / "stitched.wav"
    sf.write(str(stitched_path), stitched_array, sr)
    print(f"DEBUG: Saved stitched audio: {stitched_path}")

    whisperx_out_dir = subfolder / "whisperx_output"
    whisperx_out_dir.mkdir(exist_ok=True)

    srt_file = run_whisperx_transcription(
        stitched_path,
        whisperx_out_dir,
        language=language,
        chunk_size=chunk_size,
        no_align=True,
        model=model,
        batch_size=batch_size,
    )

    seg_boundaries = []
    current_time = 0.0
    for duration in seg_durations:
        seg_boundaries.append((current_time, current_time + duration))
        print(f"DEBUG: Calculated segment boundary: start={current_time:.3f}, end={current_time + duration:.3f}")
        current_time = current_time + duration + silence_duration_sec

    segment_transcripts = map_srt_to_segments(srt_file, seg_boundaries)
    segment_records = list(zip(segment_files, segment_transcripts))
    return segment_records, current_index

def _slice_audio_with_whisperx(audio_file, subfolder, y, sr, model, language,
                               purge_long_segments, max_segment_length, verbose_mode,
                               starting_index, chunk_size, batch_size):
    print("DEBUG: Using WhisperX timestamps for slicing.")
    total_samples = y.shape[1] if y.ndim > 1 else y.shape[0]
    audio_duration = total_samples / sr
    samples_for_vad = y.mean(axis=0) if y.ndim > 1 else y
    samples_for_vad = np.ascontiguousarray(samples_for_vad, dtype=np.float32)
    rms_times, rms_db = _compute_rms_envelope(samples_for_vad, sr)

    whisper_audio = whisperx.load_audio(str(audio_file))
    if language == "None":
        result = model.transcribe(audio=whisper_audio, chunk_size=chunk_size, batch_size=batch_size)
    else:
        result = model.transcribe(audio=whisper_audio, language=language, chunk_size=chunk_size, batch_size=batch_size)

    segments = result.get("segments", [])
    if not segments:
        print(f"DEBUG: WhisperX returned no segments for {audio_file}.")
        return [], starting_index

    segment_records = []
    current_index = starting_index
    prev_refined_end = 0.0

    for idx, segment in enumerate(segments):
        start = segment.get("start", 0.0) or 0.0
        end = segment.get("end", start) or start
        start = max(float(start), 0.0)
        end = max(float(end), start)
        end = min(end, audio_duration)

        next_start_raw = None
        if idx + 1 < len(segments):
            next_start_raw = segments[idx + 1].get("start")
        if next_start_raw is not None:
            try:
                next_start_raw = float(next_start_raw)
                next_start_raw = max(next_start_raw, 0.0)
            except (TypeError, ValueError):
                next_start_raw = None

        refined_start, refined_end = _refine_segment_boundaries(
            start=start,
            end=end,
            prev_end=prev_refined_end,
            next_start=next_start_raw,
            rms_times=rms_times,
            rms_db=rms_db,
            audio_duration=audio_duration,
        )

        if abs(refined_start - start) > 0.005 or abs(refined_end - end) > 0.005:
            print(f"DEBUG: Refined segment boundaries from ({start:.3f}, {end:.3f}) to ({refined_start:.3f}, {refined_end:.3f}).")

        duration = refined_end - refined_start

        if purge_long_segments and (duration > max_segment_length):
            print(f"DEBUG: Skipping WhisperX segment at {refined_start:.3f}-{refined_end:.3f} sec because duration {duration:.3f} sec exceeds max allowed {max_segment_length} sec.")
            continue

        if duration < 1:
            if not verbose_mode:
                print(f"DEBUG: Skipping WhisperX segment at {refined_start:.3f}-{refined_end:.3f} sec because duration {duration:.3f} sec is less than 1 sec.")
                continue
            print(f"DEBUG: Keeping short WhisperX segment at {refined_start:.3f}-{refined_end:.3f} sec (duration {duration:.3f} sec) due to verbose mode.")

        start_sample = max(int(round(refined_start * sr)), 0)
        end_sample = min(int(round(refined_end * sr)), total_samples)
        if end_sample <= start_sample:
            print(f"DEBUG: WhisperX segment at {refined_start:.3f}-{refined_end:.3f} sec had invalid sample range ({start_sample}, {end_sample}). Skipping.")
            continue

        if y.ndim > 1:
            segment_audio = y[:, start_sample:end_sample]
            seg_to_write = segment_audio.T
        else:
            seg_to_write = y[start_sample:end_sample]

        seg_filename = subfolder / f"seg{current_index}.wav"
        sf.write(str(seg_filename), seg_to_write, sr)

        transcript = segment.get("text", "")
        segment_records.append((seg_filename, transcript))
        prev_refined_end = refined_end
        print(f"DEBUG: Saved WhisperX segment {seg_filename.name} ({duration:.3f} sec) with transcript: {transcript}")

        current_index += 1

    if not segment_records:
        print(f"DEBUG: No WhisperX segments retained for {audio_file}.")

    return segment_records, current_index

def process_audio_file(audio_file, model, output_base, train_txt_path, silence_duration_sec=3,
                       slicer_params=None, purge_long_segments=False, max_segment_length=12,
                       verbose_mode=False,
                       starting_index=1, language="en",
                       slice_method=SILENCE_SLICE_METHOD, chunk_size=8, batch_size=16):
    """
    Process one audio file using the requested slicing method.
    
    Returns the next available segment index after processing this audio file.
    """
    print(f"DEBUG: Processing {audio_file}...")
    subfolder = output_base / audio_file.stem
    subfolder.mkdir(parents=True, exist_ok=True)

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

    method = (slice_method or SILENCE_SLICE_METHOD).lower()
    if method not in VALID_SLICE_METHODS:
        print(f"DEBUG: Unknown slice method '{slice_method}'. Falling back to silence slicer.")
        method = SILENCE_SLICE_METHOD

    if method == SILENCE_SLICE_METHOD:
        segment_records, next_index = _slice_audio_with_silence(
            audio_file=audio_file,
            model=model,
            subfolder=subfolder,
            y=y,
            sr=sr,
            silence_duration_sec=silence_duration_sec,
            slicer_params=slicer_params,
            purge_long_segments=purge_long_segments,
            max_segment_length=max_segment_length,
            verbose_mode=verbose_mode,
            starting_index=starting_index,
            language=language,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )
    else:
        segment_records, next_index = _slice_audio_with_whisperx(
            audio_file=audio_file,
            subfolder=subfolder,
            y=y,
            sr=sr,
            model=model,
            language=language,
            purge_long_segments=purge_long_segments,
            max_segment_length=max_segment_length,
            verbose_mode=verbose_mode,
            starting_index=starting_index,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )

    if not segment_records:
        print(f"DEBUG: No segments generated for {audio_file}.")
        return next_index

    with train_txt_path.open("a", encoding="utf-8") as f:
        for seg_path, transcript in segment_records:
            cleaned_transcript = transcript.strip()
            try:
                seg_number = int(seg_path.stem.replace("seg", ""))
            except ValueError:
                seg_number = seg_path.stem

            if len(cleaned_transcript) < 2:
                if not verbose_mode:
                    print(f"DEBUG: Skipping segment {seg_number} because transcript is too short: {transcript}")
                    continue
                print(f"DEBUG: Keeping blank transcript for segment {seg_number} due to verbose mode.")

            f.write(f"{seg_path.name} | {transcript}\n")
            print(f"DEBUG: Added dataset entry for {seg_path.name} with transcript: {transcript}")

    return next_index


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
