import gradio as gr
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================
def get_project_names():
    DATASETS_FOLDER.mkdir(exist_ok=True)
    full_paths = gu.get_available_items(str(DATASETS_FOLDER), valid_extensions=[], directory_only=True)
    return [os.path.basename(item) for item in full_paths]

# =============================================================================
# PROJECT SETUP FUNCTIONS
# =============================================================================
def create_project(project_name: str):
    project_name = project_name.strip()
    if not project_name:
        return "Project name cannot be empty.", {"choices": get_project_names(), "value": None}
    project_base = DATASETS_FOLDER / project_name
    try:
        (project_base / "wavs").mkdir(parents=True, exist_ok=True)
        (project_base / "transcribe").mkdir(parents=True, exist_ok=True)
        (project_base / "train_text_files").mkdir(parents=True, exist_ok=True)
        (project_base / "logs").mkdir(parents=True, exist_ok=True)
        status = f"Project '{project_name}' created successfully."
    except Exception as e:
        status = f"Error creating project: {str(e)}"
    return status, {"choices": get_project_names(), "value": get_project_names()[0] if get_project_names() else None}

def list_projects():
    names = get_project_names()
    default_value = names[0] if names else None
    return {"choices": names, "value": default_value}

def upload_audio_files(project: str, audio_files):
    if not project:
        return "No project selected.", list_audio_files(project)
    project_base = DATASETS_FOLDER / project
    wavs_folder = project_base / "wavs"
    wavs_folder.mkdir(parents=True, exist_ok=True)
    
    if not audio_files:
        return "No files uploaded.", list_audio_files(project)
    
    if not isinstance(audio_files, list):
        audio_files = [audio_files]
    
    messages = []
    for file in audio_files:
        try:
            file_name = getattr(file, "name", os.path.basename(file))
            file_name = os.path.basename(file_name)
            dest_path = wavs_folder / file_name
            if hasattr(file, "read"):
                with open(dest_path, "wb") as f:
                    f.write(file.read())
            elif isinstance(file, str):
                shutil.copy(file, dest_path)
            else:
                messages.append(f"Unknown file type for {file}")
                continue

            messages.append(f"{dest_path.name} uploaded.")
        except Exception as e:
            messages.append(f"Error uploading {getattr(file, 'name', file)}: {str(e)}")
    return "\n".join(messages), list_audio_files(project)

def list_audio_files(project: str):
    if not project:
        return []
    project_base = DATASETS_FOLDER / project
    wavs_folder = project_base / "wavs"
    wavs_folder.mkdir(parents=True, exist_ok=True)
    audio_files = [f.name for f in wavs_folder.iterdir() if f.suffix.lower() in VALID_AUDIO_EXTENSIONS]
    return audio_files

def load_train_txt(project: str):
    if not project:
        return "No project selected."
    project_base = DATASETS_FOLDER / project
    train_text_folder = project_base / "train_text_files"
    train_text_folder.mkdir(parents=True, exist_ok=True)
    train_txt_path = train_text_folder / "train.txt"
    if train_txt_path.exists():
        with open(train_txt_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "train.txt not found."

def load_train_with_prefix(project: str):
    content = load_train_txt(project)
    prefix_found = ""
    if content and content not in ["train.txt not found.", "No project selected."]:
        for line in content.splitlines():
            if "|" in line:
                file_id, _ = line.split("|", 1)
                file_id = file_id.strip()
                if "/" in file_id:
                    prefix_found = file_id.rsplit("/", 1)[0]
                    break
    return content, prefix_found

# =============================================================================
# NEW FUNCTION: Combine Transcribe Folders into One Dataset Folder
# =============================================================================
def export_dataset(project: str, export_format: str = "Base", speaker_id: str = "", gender: str = "unknown",
                   vibevoice_jsonl_name: str = ""):
    """
    Combine all folders (and any files directly inside) in the project's 'transcribe'
    folder into export-format-specific dataset structures.
    """
    if not project:
        return "No project selected."
    
    project = os.path.basename(project)
    export_format = (export_format or "Base").strip().lower()
    
    project_base = DATASETS_FOLDER / project
    transcribe_folder = project_base / "transcribe"
    train_text_path = project_base / "train_text_files" / "train.txt"
    if not transcribe_folder.exists():
        return "Transcribe folder not found in project."
    
    if export_format == "higgs":
        if not speaker_id.strip():
            return "Speaker ID is required for Higgs format export."
        target_folder = project_base / f"{project}_dataset"
        try:
            target_folder.mkdir(parents=True, exist_ok=False)
        except:
            raise gr.Error(f"Please remove existing exported dataset folder inside of {project} and try again.")
        return export_higgs_format(project, target_folder, transcribe_folder, train_text_path, speaker_id.strip(), gender)
    
    if export_format == "vibevoice":
        target_folder = project_base / f"{project}_vibevoice_dataset"
        try:
            target_folder.mkdir(parents=True, exist_ok=False)
        except:
            raise gr.Error(f"Please remove existing exported vibevoice dataset folder inside of {project} and try again.")
        return export_vibevoice_format(
            project=project,
            target_folder=target_folder,
            transcribe_folder=transcribe_folder,
            train_text_path=train_text_path,
            jsonl_name=vibevoice_jsonl_name.strip(),
        )

    # Base export
    target_folder = project_base / f"{project}_dataset"
    wav_folder = target_folder / "wavs"
    try:
        target_folder.mkdir(parents=True, exist_ok=False)
        wav_folder.mkdir(parents=True, exist_ok=False)
    except:
        raise gr.Error(f"Please remove existing exported dataset folder inside of {project} and try again.")
    
    file_count = 0
    for item in transcribe_folder.iterdir():
        if item.is_dir():
            for f in item.iterdir():
                if f.is_file() and "stitched" not in f.name:
                    target_file = wav_folder / f.name
                    if target_file.exists():
                        target_file = wav_folder / f"{item.name}_{f.name}"
                    shutil.copy(str(f), str(target_file))
                    file_count += 1
        elif item.is_file() and "stitched" not in item.name:
            target_file = wav_folder / item.name
            if target_file.exists():
                target_file = wav_folder / f"transcribe_{item.name}"
            shutil.copy(str(item), str(target_file))
            file_count += 1
            
    if train_text_path.exists():
        shutil.copy(str(train_text_path), str(target_folder / "train.txt"))

    return f"Combined {file_count} audio files into folder '{target_folder.name}'."

def export_higgs_format(project: str, target_folder, transcribe_folder, train_text_path, speaker_id: str, gender: str = "unknown"):
    """
    Export dataset in Higgs Audio format with:
    - Speaker-named files (speaker_id_000000.wav, speaker_id_000000.txt)
    - metadata.json with sample information and durations
    """
    import json
    import subprocess
    from pathlib import Path
    
    if not train_text_path.exists():
        return "train.txt file not found. Please generate transcription first."
    
    # Read train.txt to get transcript mappings
    transcript_map = {}
    with open(train_text_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                filename, transcript = line.strip().split('|', 1)
                transcript_map[filename.strip()] = transcript.strip()
    
    # Collect all audio files and create speaker-named files
    audio_files = []
    samples = []
    file_count = 0
    total_duration = 0.0
    
    # Process files from transcribe folder
    for item in transcribe_folder.iterdir():
        if item.is_dir():
            for f in item.iterdir():
                if f.is_file() and f.suffix.lower() in ['.wav', '.mp3', '.ogg', '.m4a'] and "stitched" not in f.name:
                    audio_files.append(f)
        elif item.is_file() and item.suffix.lower() in ['.wav', '.mp3', '.ogg', '.m4a'] and "stitched" not in item.name:
            audio_files.append(item)
    
    # Sort files to ensure consistent numbering
    audio_files.sort(key=lambda x: x.name)
    
    for i, audio_file in enumerate(audio_files):
        # Generate speaker-formatted names
        sample_id = f"{speaker_id}_{i:06d}"
        wav_filename = f"{sample_id}.wav"
        txt_filename = f"{sample_id}.txt"
        
        # Copy and rename audio file
        target_audio_path = target_folder / wav_filename
        shutil.copy(str(audio_file), str(target_audio_path))
        
        # Get transcript for this file
        original_filename = audio_file.name
        transcript = transcript_map.get(original_filename, "")
        if not transcript:
            # Try without extension
            name_without_ext = audio_file.stem
            transcript = transcript_map.get(name_without_ext, "")
        
        # Create individual transcript file
        txt_path = target_folder / txt_filename
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Get duration using ffprobe
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', str(target_audio_path)
            ], capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            # Fallback: estimate duration or set to 0
            duration = 0.0
        
        total_duration += duration
        
        # Create sample metadata
        sample_data = {
            "id": sample_id,
            "audio_file": wav_filename,
            "transcript_file": txt_filename,
            "duration": round(duration, 2),
            "speaker_id": speaker_id,
            "speaker_name": speaker_id.replace('_speaker', '').capitalize(),
            "scene": "",
            "emotion": "",
            "language": "en",  # Default to English, could be made configurable
            "gender": gender,
            "quality_score": 1.0,
            "original_audio_path": str(audio_file),
            "user_instruction": "<audio> /translate",
            "task_type": "audio_generation"
        }
        samples.append(sample_data)
        file_count += 1
    
    # Create metadata.json
    metadata = {
        "dataset_info": {
            "total_samples": file_count,
            "speakers": [speaker_id],
            "languages": ["en"],
            "total_duration": round(total_duration, 2),
            "avg_duration": round(total_duration / file_count, 2) if file_count > 0 else 0,
            "created_from": [str(train_text_path)]
        },
        "samples": samples
    }
    
    # Write metadata.json
    metadata_path = target_folder / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return f"Exported {file_count} files in Higgs format to '{target_folder.name}' with metadata.json (Total duration: {total_duration:.1f}s)"

def export_vibevoice_format(project: str, target_folder, transcribe_folder, train_text_path, jsonl_name: str):
    """
    Export dataset in Vibevoice format:
    - Flat folder of sequentially named wav files (<prefix>_000000.wav, ...)
    - JSONL file containing {"text": "Speaker 0: ...", "audio": "<dataset_folder>/<filename>"}
    """
    import json
    import posixpath

    if not train_text_path.exists():
        return "train.txt file not found. Please generate transcription first."

    transcript_map = {}
    with open(train_text_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                filename, transcript = line.strip().split('|', 1)
                transcript_map[filename.strip()] = transcript.strip()

    audio_files = []
    for item in transcribe_folder.iterdir():
        if item.is_dir():
            for f in item.iterdir():
                if f.is_file() and f.suffix.lower() in ['.wav', '.mp3', '.ogg', '.m4a'] and "stitched" not in f.name:
                    audio_files.append(f)
        elif item.is_file() and item.suffix.lower() in ['.wav', '.mp3', '.ogg', '.m4a'] and "stitched" not in item.name:
            audio_files.append(item)

    if not audio_files:
        return "No audio files found for export."

    audio_files.sort(key=lambda x: x.name)

    prefix = "vibevoice"
    dataset_folder_name = target_folder.name
    jsonl_basename = jsonl_name if jsonl_name else f"{project}_train"
    jsonl_basename = jsonl_basename.replace(" ", "_")
    jsonl_path = target_folder / f"{jsonl_basename}.jsonl"

    entries = []
    for idx, audio_file in enumerate(audio_files):
        new_name = f"{prefix}_{idx:06d}{audio_file.suffix.lower()}"
        target_audio_path = target_folder / new_name
        shutil.copy(str(audio_file), str(target_audio_path))

        original_filename = audio_file.name
        transcript = transcript_map.get(original_filename, "")
        if not transcript:
            name_without_ext = audio_file.stem
            transcript = transcript_map.get(name_without_ext, "")

        transcript = transcript.strip()
        text = f"Speaker 0: {transcript}" if transcript else "Speaker 0: "
        audio_rel_path = posixpath.join(dataset_folder_name, new_name)
        entries.append({"text": text, "audio": audio_rel_path})

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    return f"Exported {len(entries)} files in Vibevoice format to '{target_folder.name}' with {jsonl_path.name}"

# =============================================================================
# NEW FUNCTION: Combine All Audio Samples (with batching and multiprocessing)
# =============================================================================
def process_batch_func(args):
    from pydub import AudioSegment
    batch_files, batch_index, total_batches, wavs_folder, silence = args
    combined = AudioSegment.empty()
    for i, file in enumerate(batch_files):
        file_format = file.suffix[1:].lower()
        try:
            audio = AudioSegment.from_file(str(file), format=file_format)
        except Exception as e:
            return f"Error processing {file.name} in batch {batch_index}: {str(e)}"
        if i > 0:
            combined += silence
        combined += audio
    if total_batches == 1:
        out_name = "combined.wav"
    else:
        out_name = f"combined_{batch_index}.wav"
    output_path = wavs_folder / out_name
    combined.export(str(output_path), format="wav")
    return f"Batch {batch_index}: saved as '{out_name}' ({len(combined) // 1000} seconds)."

# Helper for multiprocessing duration retrieval
def _get_duration(file):
    from pydub.utils import mediainfo
    try:
        info = mediainfo(str(file))
        return float(info["duration"]) * 1000
    except Exception as e:
        raise RuntimeError(f"Error retrieving duration for {file.name}: {str(e)}")

# Helper for multiprocessing move and unlink of original files
def _move_to_uncombined(args):
    from pathlib import Path
    import shutil
    file_path, dest_folder = args
    file = Path(file_path)
    dest = Path(dest_folder)
    shutil.copy(str(file), str(dest / file.name))
    file.unlink()



# =============================================================================
# TAB FUNCTIONS
# =============================================================================

def combine_all_samples(project: str, progress_callback=gr.Progress()):
        """
        Combine all audio files (supported: .wav, .ogg, .mp3, .m4a) in the project's 'wavs' folder
        into one or more large audio files, each with a maximum duration of 2 hours.
        Inserts 10 seconds of silence between files.
        The output files are saved in the same folder with names 'combined.wav' (if one batch)
        or 'combined_1.wav', 'combined_2.wav', etc.
        """
        if not project:
            yield "No project selected."
            return
        project_base = DATASETS_FOLDER / project
        wavs_folder = project_base / "wavs"
        uncombined_folder = project_base / "uncombined_wavs"
        uncombined_folder.mkdir(parents=True, exist_ok=True)
        if not wavs_folder.exists():
            yield "Wavs folder not found in project."
            return
        audio_files = [
            f
            for ext in VALID_AUDIO_EXTENSIONS
            for f in wavs_folder.glob(f"*{ext}")
        ]
        if not audio_files:
            yield "No audio files found in the project's wavs folder."
            return

        try:
            from pydub import AudioSegment
            from pydub.utils import mediainfo
        except ImportError:
            yield "pydub module is not installed. Please install it via pip install pydub"
            return
        max_duration_ms = 2 * 60 * 60 * 1000
        silence = AudioSegment.silent(duration=10000)
        batches = []
        current_batch = []
        current_duration = 0

        # Retrieve durations in parallel using multiprocessing
        progress_callback(0, "Calculating durations...")
        try:
            with multiprocessing.Pool() as pool:
                durations = pool.map(_get_duration, audio_files)
        except Exception as e:
            yield str(e)
            return

        # Build batches using retrieved durations
        progress_callback(0.25, "Building batches...")
        for audio_file, file_duration in zip(audio_files, durations):
            additional_duration = file_duration + (10000 if current_batch else 0)
            if current_duration + additional_duration > max_duration_ms and current_batch:
                batches.append(current_batch)
                current_batch = [audio_file]
                current_duration = file_duration
            else:
                if current_batch:
                    current_duration += 10000
                current_batch.append(audio_file)
                current_duration += file_duration
        if current_batch:
            batches.append(current_batch)
        total_batches = len(batches)
        # Combine batches sequentially using numpy and soundfile instead of multiprocessing
        progress_callback(0.5, 'Processing batches...')
        import soundfile as sf
        import numpy as np
        messages = []
        for idx, batch in enumerate(batches, start=1):
            # Read first file to get sample rate and build silence buffer
            first_data, sr = sf.read(str(batch[0]), dtype='float32')
            if first_data.ndim == 1:
                silence = np.zeros(int(sr * 10), dtype=np.float32)
            else:
                silence = np.zeros((int(sr * 10), first_data.shape[1]), dtype=np.float32)
            parts = []
            for file in batch:
                data, _ = sf.read(str(file), dtype='float32')
                parts.append(data)
                parts.append(silence)
            if parts:
                parts = parts[:-1]
            combined = np.concatenate(parts)
            # Determine output filename
            out_name = 'combined.wav' if len(batches) == 1 else f'combined_{idx}.wav'
            output_path = wavs_folder / out_name
            # Write combined audio
            sf.write(str(output_path), combined, sr)
            # Report progress
            duration_sec = combined.shape[0] // sr
            messages.append(f'Batch {idx}: saved as {out_name} ({duration_sec} seconds).')
            yield '\n'.join(messages)
        # Move original wav files to uncombined folder
        progress_callback(0.75, 'Finishing up and moving original wav files.')
        import shutil
        for f in audio_files:
            shutil.copy(str(f), str(uncombined_folder / f.name))
            f.unlink()

def get_resume_status(project: str):
    """
    Check if transcription can be resumed and return the status info.
    Returns: (can_resume: bool, processed_files: list, starting_index: int, message: str)
    """
    if not project:
        return False, [], 1, "No project selected."
    
    project_base = DATASETS_FOLDER / project
    transcribe_folder = project_base / "transcribe"
    train_text_folder = project_base / "train_text_files"
    train_txt_path = train_text_folder / "train.txt"
    wavs_folder = project_base / "wavs"
    
    # Get all audio files
    audio_files = [
        f.stem for ext in VALID_AUDIO_EXTENSIONS
        for f in wavs_folder.glob(f"*{ext}")
    ] if wavs_folder.exists() else []
    
    # Check if previous run exists
    if not transcribe_folder.exists() or not train_txt_path.exists():
        return False, [], 1, "No previous transcription run found."
    
    # Find processed files by checking transcribe subfolders
    processed_files = [
        folder.name for folder in transcribe_folder.iterdir()
        if folder.is_dir()
    ]
    
    # Get highest segment number from train.txt to determine starting index
    starting_index = 1
    if train_txt_path.exists():
        try:
            with open(train_txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    # Parse the last line to get the highest segment number
                    last_line = lines[-1].strip()
                    if "|" in last_line:
                        filename_part = last_line.split("|")[0].strip()
                        if filename_part.startswith("seg") and filename_part.endswith(".wav"):
                            # Extract number from segXXXX.wav
                            seg_num = filename_part.replace("seg", "").replace(".wav", "")
                            starting_index = int(seg_num) + 1
        except (ValueError, IndexError, FileNotFoundError):
            starting_index = 1
    
    # Find unprocessed files
    unprocessed_files = [f for f in audio_files if f not in processed_files]
    
    if not unprocessed_files:
        return False, processed_files, starting_index, f"All {len(audio_files)} files have been processed."
    
    message = f"Found previous run: {len(processed_files)} files processed, {len(unprocessed_files)} remaining. Next segment: {starting_index}"
    return True, processed_files, starting_index, message

def transcribe_interface(project: str, language, silence_duration, purge_long_segments,
                         max_segment_length, verbose_mode=False, resume_mode=False,
                         slice_method_label="Silence Slicer"):
    if not project:
        return "No project selected."
    
    project_base = DATASETS_FOLDER / project
    wavs_folder = project_base / "wavs"
    if not wavs_folder.exists():
        return "No audio files uploaded. Please upload files into the 'wavs' folder."
    
    transcribe_folder = project_base / "transcribe"
    train_text_folder = project_base / "train_text_files"
    train_txt_path = train_text_folder / "train.txt"
    
    slice_options = globals().get("SLICE_METHOD_OPTIONS", {})
    slice_method = transcriber.SILENCE_SLICE_METHOD
    if isinstance(slice_method_label, str):
        mapped_value = slice_options.get(slice_method_label)
        if mapped_value:
            slice_method = mapped_value
        elif slice_method_label.lower() in transcriber.VALID_SLICE_METHODS:
            slice_method = slice_method_label.lower()
    elif slice_method_label in transcriber.VALID_SLICE_METHODS:
        slice_method = slice_method_label
    
    # Handle resume vs new run
    if resume_mode:
        can_resume, processed_files, starting_index, status_msg = get_resume_status(project)
        if not can_resume:
            return f"Cannot resume: {status_msg}"
        
        logger.info(f"Resuming transcription: {status_msg}")
        
        # Get unprocessed files
        audio_files = [
            f for ext in VALID_AUDIO_EXTENSIONS
            for f in wavs_folder.glob(f"*{ext}")
            if f.stem not in processed_files
        ]
        
    else:
        # New run - check for conflicts
        if transcribe_folder.exists():
            has_transcribe_outputs = any(transcribe_folder.iterdir())
            if has_transcribe_outputs:
                raise gr.Error(f"Transcribe folder already exists. Please remove previous run and try again, or use Resume mode.")
        
        if train_txt_path.exists():
            raise gr.Error(f"Train text file already exists. Please remove previous run and try again, or use Resume mode.")
        
        # Create folders
        transcribe_folder.mkdir(parents=True, exist_ok=True)
        train_text_folder.mkdir(parents=True, exist_ok=True)
        
        starting_index = 1
        audio_files = [
            f for ext in VALID_AUDIO_EXTENSIONS
            for f in wavs_folder.glob(f"*{ext}")
        ]
    
    if not audio_files:
        if resume_mode:
            return "Resume complete: All files have been processed."
        else:
            return "No valid audio files found in the 'wavs' folder."
    
    try:
        model = transcriber.load_whisperx_model("large-v3")
        
        logger.info(f"Starting transcription using slice method '{slice_method}'.")
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
            
            starting_index = transcriber.process_audio_file(
                audio_file=audio_file,
                model=model,
                output_base=transcribe_folder,
                train_txt_path=train_txt_path,
                    silence_duration_sec=silence_duration,
                    purge_long_segments=purge_long_segments,
                    max_segment_length=max_segment_length,
                    verbose_mode=verbose_mode,
                    starting_index=starting_index,
                    language=language,
                    slice_method=slice_method
                )
    
        if train_txt_path.exists():
            with open(train_txt_path, "r", encoding="utf-8") as f:
                content = f.read()
                final_count = len([line for line in content.split('\n') if '|' in line])
                return f"Transcription complete! Generated {final_count} entries.\n\n{content}"
        else:
            return "No train.txt file was generated."
            
    except Exception as e:
        return f"Error during transcription: {str(e)}"

def move_previous_run(project: str):
    project_base = DATASETS_FOLDER / project
    transcribe_folder = project_base / "transcribe"
    train_text_folder = project_base / "train_text_files"
    train_txt_path = train_text_folder / "train.txt"
    old_runs_folder = project_base / "old_runs"
    
    # Get current date and time for folder naming
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if transcribe_folder.exists():
        shutil.move(transcribe_folder, old_runs_folder / f"{timestamp}" / "transcribe_audio_folder")
    if train_txt_path.exists():
        shutil.move(train_txt_path, old_runs_folder / f"{timestamp}")
    return f"Previous run moved to '{timestamp}' in the old_runs folder."

def correct_transcription_interface(project: str):
    if not project:
        return "No project selected."
    
    project_base = DATASETS_FOLDER / project
    train_text_folder = project_base / "train_text_files"
    train_text_folder.mkdir(parents=True, exist_ok=True)
    
    input_filepath = train_text_folder / "train.txt"
    output_filepath = train_text_folder / "train_correct.txt"
    
    if not input_filepath.exists():
        return "train.txt not found in project."
    
    try:
        import tkinter.messagebox
        tkinter.messagebox.showinfo = tkinter.messagebox.showerror = lambda *args, **kwargs: None
        llm_reformatter_script.confirm_overwrite = lambda filepath: True
        
        llm_reformatter_script.process_file(str(input_filepath), str(output_filepath))
        
        if output_filepath.exists():
            with open(output_filepath, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return "No corrected transcript generated."
    except Exception as e:
        return f"Error during correction: {str(e)}"

# =============================================================================
# NEW FUNCTIONS: Preview & Save Conversion for Adjusting train.txt
# =============================================================================
def preview_train_conversion(prefix: str, base_format: str, target_format: str, project: str, speaker_input: str, language_input: str):
    """
    Preview the conversion of train.txt from the selected base format to the target format.
    Conversion rules:
      - Base formats:
          Tortoise: file_id | transcript
          StyleTTS: file_id | transcript | speaker_id
          GPTSoVITS: file_id | slicer_opt | language | transcript
      - Target formats:
          Tortoise: file_id | transcript
          StyleTTS: file_id | transcript | speaker_id   (speaker comes from speaker_input)
          GPTSoVITS: file_id | slicer_opt | language | transcript   (slicer_opt is predetermined,
                      language comes from language_input)
    If fields are missing in the source, they are filled with blanks.
    If extra fields exist, they are dropped.
    The file_id is updated with the prefix if provided.
    """
    if not project:
        return "No project selected."
    project_base = DATASETS_FOLDER / project
    train_text_folder = project_base / "train_text_files"
    train_txt_path = train_text_folder / "train.txt"
    if not train_txt_path.exists():
        return "train.txt not found."
    
    with open(train_txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    base_fields = BASE_FORMATS.get(base_format)
    target_fields = TARGET_FORMATS.get(target_format)
    if base_fields is None or target_fields is None:
        return "Invalid format selection. Valid formats are: Tortoise, StyleTTS, GPTSoVITS."
    
    preview_lines = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        # Only attempt to split lines containing a pipe delimiter
        if "|" not in line:
            preview_lines.append(line)
            continue

        # Split on first pipe to separate file_id and the rest
        parts = [p.strip() for p in line.split("|", 1)]
        if len(parts) != len(base_fields):
            error_msg = (f"Error on line {lineno}: Expected {len(base_fields)} fields for base format '{base_format}', "
                         f"but got {len(parts)}. Valid formats: Tortoise (2 fields), StyleTTS (3 fields), GPTSoVITS (4 fields).")
            return error_msg
        
        # Create a dictionary from the base format.
        source_dict = dict(zip(base_fields, parts))
        
        # Update file_id with prefix if provided.
        orig_file_id = source_dict.get("file_id", "")
        if prefix.strip():
            file_name = os.path.basename(orig_file_id)
            source_dict["file_id"] = f"{prefix.strip()}/{file_name}"
        else:
            source_dict["file_id"] = orig_file_id
        
        # Depending on the target format, update extra fields.
        if target_format == "StyleTTS":
            # Target fields: file_id, transcript, speaker_id.
            # Override speaker_id with speaker_input.
            source_dict["speaker_id"] = speaker_input.strip()
        elif target_format == "GPTSoVITS":
            # Target fields: file_id, slicer_opt, language, transcript.
            # Set slicer_opt to a predetermined value and override language with language_input.
            source_dict["slicer_opt"] = "slicer_opt"
            source_dict["language"] = language_input.strip()
        # For Tortoise, no extra fields.
        
        # For any target field not in the source, fill with blank.
        target_dict = {}
        for field in target_fields:
            target_dict[field] = source_dict.get(field, "")
        
        new_line = "|".join(target_dict[field] for field in target_fields)
        preview_lines.append(new_line)
    
    return "\n".join(preview_lines)

def save_adjusted_train_content(adjusted_content: str, project: str):
    """
    Save the provided adjusted content to the project's train.txt file.
    """
    if not project:
        return "No project selected."
    project_base = DATASETS_FOLDER / project
    train_text_folder = project_base / "train_text_files"
    train_txt_path = train_text_folder / "train_updated.txt"
    try:
        with open(train_txt_path, "w", encoding="utf-8") as f:
            f.write(adjusted_content)
        return "train.txt successfully updated."
    except Exception as e:
        return f"Error saving file: {str(e)}"

# =============================================================================
# GRADIO INTERFACE SETUP
# =============================================================================
def setup_gradio():
    with gr.Blocks(title="Transcription & Correction Interface") as demo:
        
        gr.Markdown("## Project Setup")
        with gr.Row():
            new_project_input = gr.Textbox(label="New Project Name", placeholder="Enter new project name")
            create_project_button = gr.Button("Create Project")
            project_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            projects_root = gr.Textbox(value=str(DATASETS_FOLDER), visible=False)
            projects_valid_ext = gr.Textbox(value="[]", visible=False)
            projects_dir_only = gr.Textbox(value="directory", visible=False)
            
            projects_dropdown = gr.Dropdown(
                label="Select Project",
                choices=get_project_names(),
                value=get_project_names()[0] if get_project_names() else None
            )
            refresh_projects_button = gr.Button("Refresh Projects")
        
        with gr.Row():
            upload_audio = gr.File(label="Upload Audio File(s)", file_count="multiple")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            refresh_audio_button = gr.Button("Refresh Audio Files")
            audio_files_dropdown = gr.Dropdown(label="Project Audio Files (in wavs)", choices=[])
        
        create_project_button.click(
            fn=create_project,
            inputs=new_project_input,
            outputs=[project_status, projects_dropdown],
        )
        refresh_projects_button.click(
            fn=gu.refresh_dropdown_proxy,
            inputs=[projects_root, projects_valid_ext, projects_dir_only],
            outputs=projects_dropdown,
        )
        
        upload_audio.upload(
            fn=upload_audio_files,
            inputs=[projects_dropdown, upload_audio],
            outputs=[upload_status, audio_files_dropdown],
        )
        refresh_audio_button.click(
            fn=list_audio_files,
            inputs=projects_dropdown,
            outputs=audio_files_dropdown,
        )
        
        gr.Markdown("## Project Tasks")

        with gr.Tabs():
            with gr.Tab("Combine Small Samples"):
                gr.Markdown("### Combine All Supported Audio Files into One File")
                gr.Markdown("This will merge all supported audio files in the project's 'wavs' folder into one or more files (each up to 2 hours long) with 10 seconds of silence between each sample. This is NEEDED if your dataset consists short audio samples.  If you don't do this, **Transcribe** will process VERY slowly.  The short samples will be moved into uncombined_wavs folder after combining.")
                gr.Markdown("**Note:** Do NOT include long samples in the 'wavs' folder (1+ hours in length) as this may cause issues with the combining process.")
                combine_button = gr.Button("Combine All Samples")
                combine_status = gr.Textbox(label="Status", lines=2)

                combine_button.click(
                    fn=combine_all_samples,
                    inputs=projects_dropdown,
                    outputs=combine_status,
                )
                
            with gr.Tab("Transcribe"):
                gr.Markdown("### Transcribe All Audio Files in the Project (wavs)")
                gr.Markdown("For optimal speeds, ensure that all files in the 'wavs' folder are 10 minutes or more in length. If not, use **Combine Small Samples** to combine them.")
                gr.Markdown("**NOTE:** It is HIGHLY suggested that audio has no background noise or music.  If it does, running through a background remover like UVR or something similar is necessary before transcribing.")
                
                # Resume status section
                with gr.Row():
                    check_resume_button = gr.Button("Check Resume Status")
                    resume_status = gr.Textbox(label="Resume Status", interactive=False)
                
                check_resume_button.click(
                    fn=lambda project: get_resume_status(project)[3],
                    inputs=projects_dropdown,
                    outputs=resume_status,
                )
                
                with gr.Row():
                    language = gr.Dropdown(label="Language", choices=WHISPER_LANGUAGES,
                                            value="en", interactive=True)
                    slice_method_dropdown = gr.Dropdown(
                        label="Slicing Method",
                        choices=list(SLICE_METHOD_OPTIONS.keys()),
                        value=DEFAULT_SLICE_METHOD_LABEL,
                        interactive=True,
                    )
                with gr.Row():
                    silence_duration = gr.Slider(label="Silence Duration (seconds)", minimum=1, maximum=10, value=6, step=1)
                slice_method_dropdown.change(
                    fn=lambda label: gr.update(
                        interactive=(SLICE_METHOD_OPTIONS.get(label, str(label).lower()) == transcriber.SILENCE_SLICE_METHOD)
                    ),
                    inputs=slice_method_dropdown,
                    outputs=silence_duration,
                )
                with gr.Row():
                    purge_checkbox = gr.Checkbox(label="Purge segments longer than threshold", value=False)
                    max_segment_length_slider = gr.Slider(label="Max Segment Length (seconds)", minimum=1, maximum=60, value=12, step=1)
                verbose_checkbox = gr.Checkbox(label="Verbose (keep short/blank segments)", value=True)
                
                with gr.Row():
                    transcribe_button = gr.Button("Start New Transcription", variant="primary")
                    resume_button = gr.Button("Resume Previous Run", variant="secondary")
                    move_previous_run_button = gr.Button("Archive Previous Run")
                
                transcribe_output = gr.Textbox(label="train.txt Content", lines=10)
                
                transcribe_button.click(
                    fn=lambda proj, lang, silence, purge, max_len, verbose, slice_choice: transcribe_interface(proj, lang, silence, purge, max_len, verbose, resume_mode=False, slice_method_label=slice_choice),
                    inputs=[projects_dropdown, language, silence_duration, purge_checkbox, max_segment_length_slider, verbose_checkbox, slice_method_dropdown],
                    outputs=transcribe_output,
                )
                resume_button.click(
                    fn=lambda proj, lang, silence, purge, max_len, verbose, slice_choice: transcribe_interface(proj, lang, silence, purge, max_len, verbose, resume_mode=True, slice_method_label=slice_choice),
                    inputs=[projects_dropdown, language, silence_duration, purge_checkbox, max_segment_length_slider, verbose_checkbox, slice_method_dropdown],
                    outputs=transcribe_output,
                )
                move_previous_run_button.click(
                    fn=move_previous_run,
                    inputs=projects_dropdown,
                    outputs=transcribe_output,
                )
            

            with gr.Tab("Correct Transcription"):
                gr.Markdown("### Correct the Transcript from train.txt")
                load_transcript_button = gr.Button("Load train.txt")
                transcript_content = gr.Textbox(label="train.txt Content", lines=10)
                correct_button = gr.Button("Correct Transcription")
                corrected_output = gr.Textbox(label="Corrected Transcript", lines=10)
                
                load_transcript_button.click(
                    fn=load_train_txt,
                    inputs=projects_dropdown,
                    outputs=transcript_content,
                )
                correct_button.click(
                    fn=correct_transcription_interface,
                    inputs=projects_dropdown,
                    outputs=corrected_output,
                )
            
            with gr.Tab("Adjust train.txt File"):
                gr.Markdown("### Adjust the File IDs in train.txt")
                load_train_button = gr.Button("Load train.txt")
                adjust_input = gr.Textbox(label="Current train.txt Content", lines=10)
                prefix_text = gr.Textbox(label="Prefix Text (e.g. bob)", value="")
                
                # Dropdown for selecting base format (current file format)
                base_format_dropdown = gr.Dropdown(
                    label="Base Format",
                    choices=["Tortoise", "StyleTTS", "GPTSoVITS"],
                    value="Tortoise"
                )
                # Dropdown for selecting target format (desired output format)
                target_format_dropdown = gr.Dropdown(
                    label="Target Format",
                    choices=["Tortoise", "StyleTTS", "GPTSoVITS"],
                    value="Tortoise"
                )
                
                # Additional fields that only appear if needed:
                speaker_input = gr.Textbox(label="Speaker (for StyleTTS)", visible=False, value="")
                language_input = gr.Textbox(label="Language (for GPTSoVITS)", visible=False, value="")
                
                # Function to update visibility based on target format.
                def update_target_fields(target_format):
                    if target_format == "StyleTTS":
                        return gr.update(visible=True), gr.update(visible=False)
                    elif target_format == "GPTSoVITS":
                        return gr.update(visible=False), gr.update(visible=True)
                    else:
                        return gr.update(visible=False), gr.update(visible=False)
                
                target_format_dropdown.change(
                    fn=update_target_fields,
                    inputs=[target_format_dropdown],
                    outputs=[speaker_input, language_input]
                )
                
                adjust_preview_button = gr.Button("Preview Adjust")
                adjust_preview_output = gr.Textbox(label="Preview Adjusted train.txt Content", lines=10)
                
                save_adjust_button = gr.Button("Save Adjusted train.txt")
                save_adjust_output = gr.Textbox(label="Save Status", lines=2)
                
                load_train_button.click(
                    fn=load_train_with_prefix,
                    inputs=projects_dropdown,
                    outputs=[adjust_input, prefix_text],
                )
                # Auto-load train.txt and prefix when project selection changes
                projects_dropdown.change(
                    fn=load_train_with_prefix,
                    inputs=[projects_dropdown],
                    outputs=[adjust_input, prefix_text],
                )
                adjust_preview_button.click(
                    fn=preview_train_conversion,
                    inputs=[prefix_text, base_format_dropdown, target_format_dropdown, projects_dropdown, speaker_input, language_input],
                    outputs=adjust_preview_output,
                )
                save_adjust_button.click(
                    fn=save_adjusted_train_content,
                    inputs=[adjust_preview_output, projects_dropdown],
                    outputs=save_adjust_output,
                )
            
            with gr.Tab("Export Dataset"):
                gr.Markdown("### Export All Transcribe Folders into a Single Dataset Folder")
                gr.Markdown("This will copy all files from subfolders (and files directly in the folder) of the project's 'transcribe' folder into a single folder named '<project>_dataset'.")
                
                export_format_dropdown = gr.Dropdown(label="Export Format", choices=["Base", "Higgs", "Vibevoice"], value="Base")

                with gr.Row(visible=False) as higgs_options:
                    speaker_id_input = gr.Textbox(label="Speaker ID", placeholder="e.g., alma_speaker, melina_speaker", info="Used for Higgs format file naming", scale=2)
                    gender_dropdown = gr.Dropdown(choices=["male", "female", "unknown"], value="unknown", label="Gender", scale=1)

                with gr.Row(visible=False) as vibevoice_options:
                    vibevoice_jsonl_input = gr.Textbox(label="JSONL Name", placeholder="e.g., vibevoice_train", info="Filename (without extension) for the JSONL manifest", scale=2)

                export_dataset_button = gr.Button("Export Dataset")
                export_dataset_status = gr.Textbox(label="Status", lines=2)
                
                def update_export_options(format_choice):
                    is_higgs = format_choice == "Higgs"
                    is_vibevoice = format_choice == "Vibevoice"
                    return (
                        gr.update(visible=is_higgs),
                        gr.update(visible=is_vibevoice),
                    )
                
                export_format_dropdown.change(
                    fn=update_export_options,
                    inputs=export_format_dropdown,
                    outputs=[higgs_options, vibevoice_options],
                )
                
                export_dataset_button.click(
                    fn=export_dataset,
                    inputs=[projects_dropdown, export_format_dropdown, speaker_id_input, gender_dropdown, vibevoice_jsonl_input],
                    outputs=export_dataset_status,
                )

    return demo

def main():
    demo = setup_gradio()
    demo.launch()

if __name__ == "__main__":
    import os
    import shutil
    from pathlib import Path
    import multiprocessing
    import datetime

    # Import your existing modules.
    import transcriber
    import llm_reformatter_script

    # Import your custom utilities.
    from gradio_utils import utils as gu
    # =============================================================================
    # Global Project Folder
    # =============================================================================
    DATASETS_FOLDER = Path.cwd() / "datasets_folder"

    # Predefined configuration profiles
    # The formats are defined as lists of field names.
    # Tortoise: file_id | transcript
    # StyleTTS: file_id | transcript | speaker_id
    # GPTSoVITS: file_id | slicer_opt | language | transcript
    BASE_FORMATS = {
        "Tortoise": ["file_id", "transcript"],
        "StyleTTS": ["file_id", "transcript", "speaker_id"],
        "GPTSoVITS": ["file_id", "slicer_opt", "language", "transcript"],
    }
    TARGET_FORMATS = {
        "Tortoise": ["file_id", "transcript"],
        "StyleTTS": ["file_id", "transcript", "speaker_id"],
        "GPTSoVITS": ["file_id", "slicer_opt", "language", "transcript"],
    }
    VALID_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".opus", ".webm", ".mp4", ".ogg"]
    WHISPER_LANGUAGES = ["af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","yue","zh"]
    SLICE_METHOD_OPTIONS = {
        "WhisperX Timestamps": transcriber.WHISPERX_SLICE_METHOD,
        "Silence Slicer": transcriber.SILENCE_SLICE_METHOD,
        
    }
    DEFAULT_SLICE_METHOD_LABEL = "WhisperX Timestamps"
    main()
