#!/usr/bin/env python3
"""
Entry point for the Emilia speech preprocessing pipeline.

This script mirrors the original Amphion Emilia-Pipe workflow while integrating
with the local repository layout (modules under `Emilia/`).  It prepares audio
for speech-generation datasets by running:

1. Audio standardisation (24 kHz, mono, loudness normalisation).
2. Source separation to isolate vocals.
3. Speaker diarisation via pyannote.
4. Fine-grained segmentation with Silero VAD and post-processing rules.
5. WhisperX transcription (optionally multilingual).
6. DNSMOS scoring and quality filtering.
7. Export of per-segment MP3 files plus a JSON manifest.

Usage (with uv-managed environment):
    uv run python emilia_pipeline.py --config Emilia/config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import pyannote
import torch
import torch.serialization
import tqdm
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Resolution
import soundfile as sf

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Specifications])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Problem])
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion, Resolution])

import safe_globals  # Ensure torch safe globals registered before loading checkpoints.

from Emilia.models import dnsmos, silero_vad, whisper_asr
from infer_uvr import UVRSeparator
from Emilia.utils.logger import Logger, time_logger
from Emilia.utils.tool import (
    calculate_audio_stats,
    check_env,
    export_to_mp3,
    get_audio_files,
    load_cfg,
)


warnings.filterwarnings("ignore")

AudioDict = Dict[str, Any]
Segment = Dict[str, Any]


def write_debug_audio(path: Path, data: np.ndarray, sample_rate: int) -> None:
    """Write a waveform to disk for debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim > 1:
        data = librosa.to_mono(data.T)
    sf.write(str(path), data.astype(np.float32), sample_rate)


def dump_segments_audio(
    name_prefix: str,
    segments: List[Segment],
    waveform: np.ndarray,
    sample_rate: int,
    folder: Path,
) -> None:
    """Dump each segment to an individual WAV file for inspection."""
    for idx, seg in enumerate(segments):
        start_idx = max(int(seg["start"] * sample_rate), 0)
        end_idx = min(int(seg["end"] * sample_rate), waveform.shape[0])
        if end_idx <= start_idx:
            continue
        clip = waveform[start_idx:end_idx]
        write_debug_audio(folder / f"{name_prefix}_{idx:03}.wav", clip, sample_rate)


def normalize_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    *,
    target_dbfs: float = -20.0,
    max_gain_db: float = 3.0,
) -> Tuple[np.ndarray, int]:
    """Resample and loudness-normalize a waveform using AudioSegment."""
    waveform = np.nan_to_num(waveform.astype(np.float32))
    max_abs = np.max(np.abs(waveform)) or 1.0
    waveform = waveform / max_abs

    audio_segment = AudioSegment(
        (waveform * 32767).astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    audio_segment = audio_segment.set_frame_rate(target_sample_rate)

    gain = target_dbfs - audio_segment.dBFS
    clamped_gain = max(-max_gain_db, min(max_gain_db, gain))
    normalized = audio_segment.apply_gain(clamped_gain)

    data = np.array(normalized.get_array_of_samples(), dtype=np.float32)
    max_amp = np.max(np.abs(data)) or 1.0
    data /= max_amp
    return data, normalized.frame_rate


def _resolve_path(base_cfg: Path, candidate: Union[str, Path]) -> Path:
    """Resolve a path relative to the configuration file location."""
    path = Path(candidate)
    if path.is_absolute():
        return path

    relative_to_config = (base_cfg.parent / path)
    if relative_to_config.exists():
        return relative_to_config.resolve()

    relative_to_cwd = (Path.cwd() / path)
    if relative_to_cwd.exists():
        return relative_to_cwd.resolve()

    # Fall back to the config-relative location even if it does not yet exist,
    # so callers can create it later.
    return relative_to_config.resolve(strict=False)


def _derive_output_name(audio_path: Path, use_hash: bool, length: int = 16) -> str:
    """
    Produce the base name used for exported assets.

    When use_hash is True, return a deterministic hash (first `length`
    chars of SHA-256) derived from the absolute file path.
    """
    if not use_hash:
        return audio_path.stem

    digest = hashlib.sha256(str(audio_path.resolve()).encode("utf-8")).hexdigest()
    return digest[: max(8, length)]


@time_logger
def standardise_audio(
    audio: Union[str, AudioSegment],
    target_sample_rate: int,
    *,
    target_dbfs: float = -20.0,
    max_gain_db: float = 3.0,
) -> AudioDict:
    """
    Convert audio to mono 24 kHz float32 waveform with bounded loudness gain.
    """
    if isinstance(audio, str):
        name = Path(audio).name
        segment = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = "audio_segment"
        segment = audio
    else:
        raise ValueError("Unsupported audio input type.")

    segment = (
        segment.set_frame_rate(target_sample_rate)
        .set_sample_width(2)
        .set_channels(1)
    )

    gain = target_dbfs - segment.dBFS
    clamped_gain = max(-max_gain_db, min(max_gain_db, gain))
    normalised = segment.apply_gain(clamped_gain)

    waveform = np.array(normalised.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform)) or 1.0
    waveform /= max_amplitude

    return {"waveform": waveform, "name": name, "sample_rate": target_sample_rate}


def separate_sources(
    predictor: UVRSeparator,
    audio: Union[str, AudioDict],
    target_sample_rate: int,
) -> AudioDict:
    """Apply UVR-MDX separation on raw audio and return normalized vocals."""
    if isinstance(audio, str):
        waveform, sample_rate = librosa.load(audio, sr=None, mono=False)
        name = Path(audio).name
    elif isinstance(audio, dict):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        name = audio.get("name", "audio")
    else:
        raise ValueError("Unsupported audio source for separation.")

    if waveform.ndim == 1:
        waveform = np.stack([waveform, waveform])

    if waveform.shape[0] > 2:
        waveform = waveform[:2]

    background, vocals = predictor.predict(waveform, sample_rate)

    def mixdown(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr.astype(np.float32)
        if arr.ndim == 2:
            channel_axis = 0 if arr.shape[0] < arr.shape[1] else 1
            return np.mean(arr, axis=channel_axis, keepdims=False).astype(np.float32)
        # Collapse all but time dimension.
        return np.mean(arr, axis=tuple(range(arr.ndim - 1))).astype(np.float32)

    vocal_mono = mixdown(vocals)
    background_mono = mixdown(background)

    vocal_resampled = librosa.resample(
        vocal_mono, orig_sr=sample_rate, target_sr=target_sample_rate
    )
    background_resampled = librosa.resample(
        background_mono, orig_sr=sample_rate, target_sr=target_sample_rate
    )

    vocal_normalized, norm_sr = normalize_waveform(
        vocal_resampled, target_sample_rate, target_sample_rate
    )
    background_normalized, _ = normalize_waveform(
        background_resampled, target_sample_rate, target_sample_rate
    )

    return {
        "waveform": vocal_normalized,
        "instrumental_waveform": background_normalized,
        "sample_rate": norm_sr,
        "name": name,
    }


@time_logger
def diarise_speakers(
    pipeline: Pipeline, audio: AudioDict, device: torch.device
) -> pd.DataFrame:
    """Run speaker diarisation returning a dataframe with segment metadata."""
    waveform = torch.tensor(audio["waveform"], device=device).unsqueeze(0)
    diarisation = pipeline(
        {"waveform": waveform, "sample_rate": audio["sample_rate"], "channel": 0}
    )

    # pyannote >=3 returns DiarizeOutput, older versions return Annotation directly
    if hasattr(diarisation, "itertracks"):
        annotation = diarisation
    elif hasattr(diarisation, "speaker_diarization"):
        annotation = diarisation.speaker_diarization
    else:
        raise AttributeError(
            "Unsupported diarisation return type from pyannote Pipeline. "
            "Expected Annotation or DiarizeOutput with speaker_diarization."
        )

    diarise_df = pd.DataFrame(
        annotation.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarise_df["start"] = diarise_df["segment"].apply(lambda seg: seg.start)
    diarise_df["end"] = diarise_df["segment"].apply(lambda seg: seg.end)
    return diarise_df


@time_logger
def merge_vad_segments(
    vad_list: Iterable[Segment],
    *,
    merge_target: float = 3.0,
    keep_min_len: float = 0.25,
    max_len: float = 30.0,
    merge_gap: float = 2.0,
) -> List[Segment]:
    """Merge Silero VAD segments enforcing speaker consistency and time limits."""
    vad_items = [segment.copy() for segment in vad_list]
    logger = Logger.get_logger()
    merge_target = max(0.0, float(merge_target))
    keep_min_len = max(0.0, float(keep_min_len))
    max_len = max(merge_target if merge_target > 0 else 0.1, float(max_len))
    merge_gap = max(0.0, float(merge_gap))

    merged: List[Segment] = []

    for segment in vad_items:
        last_start = merged[-1]["start"] if merged else None
        last_end = merged[-1]["end"] if merged else None
        last_speaker = merged[-1]["speaker"] if merged else None

        duration = segment["end"] - segment["start"]
        if duration >= max_len:
            current_start = segment["start"]
            segment_end = segment["end"]
            logger.warning(
                "merge_vad_segments > segment longer than 30 seconds, splitting."
            )
            while segment_end - current_start >= max_len:
                segment["end"] = current_start + max_len
                merged.append(segment.copy())
                segment = segment.copy()
                current_start += max_len
                segment["start"] = current_start
                segment["end"] = segment_end
            merged.append(segment)
            continue

        if (
            last_speaker is None
            or last_speaker != segment["speaker"]
            or duration >= merge_target
        ):
            merged.append(segment.copy())
            continue

        if (
            segment["start"] - last_end >= merge_gap
            or segment["end"] - last_start >= max_len
        ):
            merged.append(segment.copy())
        else:
            merged[-1]["end"] = segment["end"]

    filtered = [
        seg for seg in merged if (seg["end"] - seg["start"]) >= keep_min_len
    ]

    if not filtered and vad_items:
        logger.info(
            "merge_vad_segments > fallback to raw VAD segments (count=%d).",
            len(vad_items),
        )
        return vad_items

    logger.debug(
        "merge_vad_segments > merged %d segments",
        len(vad_items) - len(filtered),
    )
    return filtered


@time_logger
def run_asr(
    model: whisper_asr.VadFreeFasterWhisperPipeline,
    segments: List[Segment],
    audio: AudioDict,
    *,
    multilingual: bool,
    supported_languages: List[str],
    batch_size: int,
    forced_language: Optional[str] = None,
) -> List[Segment]:
    """Transcribe each segment using WhisperX with optional language filtering."""
    logger = Logger.get_logger()

    if not segments:
        logger.info("run_asr: received 0 segments, skipping ASR.")
        return []

    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    start_time = segments[0]["start"]
    end_time = segments[-1]["end"]
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)

    trimmed = waveform[start_frame:end_frame]
    resampled = librosa.resample(trimmed, orig_sr=sample_rate, target_sr=16000)

    shifted_segments: List[Segment] = []
    for seg in segments:
        shifted_segments.append(
            {
                **seg,
                "start": seg["start"] - start_time,
                "end": seg["end"] - start_time,
            }
        )

    forced_language = (forced_language or "").strip()

    if forced_language:
        transcribed = model.transcribe(
            resampled,
            shifted_segments,
            batch_size=batch_size,
            language=forced_language,
            print_progress=True,
        )["segments"]
        for seg in transcribed:
            seg["start"] += start_time
            seg["end"] += start_time
            seg["language"] = forced_language
        return sorted(transcribed, key=lambda item: item["start"])

    if multilingual:
        valid_segments: List[Segment] = []
        languages: List[str] = []
        min_prob = 0.3
        for seg in shifted_segments:
            s = int(seg["start"] * 16000)
            e = int(seg["end"] * 16000)
            audio_slice = resampled[s:e]
            language, prob = model.detect_language(audio_slice)
            logger.debug(
                "run_asr: detected language=%s prob=%.3f for segment %.2f-%.2f",
                language,
                prob,
                seg["start"] + start_time,
                seg["end"] + start_time,
            )
            if language in supported_languages and prob >= min_prob:
                valid_segments.append(seg)
                languages.append(language)

        if not valid_segments:
            logger.info("run_asr: no segments passed language/probability filter.")
            return []

        results: List[Segment] = []
        for language in sorted(set(languages)):
            lang_segments = [
                seg for seg, lang in zip(valid_segments, languages) if lang == language
            ]
            transcribed = model.transcribe(
                resampled,
                lang_segments,
                batch_size=batch_size,
                language=language,
                print_progress=True,
            )["segments"]
            for seg in transcribed:
                seg["start"] += start_time
                seg["end"] += start_time
                seg["language"] = language
            results.extend(transcribed)
        return sorted(results, key=lambda item: item["start"])

    language, prob = model.detect_language(resampled)
    if language not in supported_languages or prob < 0.3:
        logger.info(
            "run_asr: detected language=%s prob=%.3f not in supported list %s.",
            language,
            prob,
            supported_languages,
        )
        return []

    transcribed = model.transcribe(
        resampled,
        shifted_segments,
        batch_size=batch_size,
        language=language,
        print_progress=True,
    )["segments"]
    for seg in transcribed:
        seg["start"] += start_time
        seg["end"] += start_time
        seg["language"] = language
    return transcribed


@time_logger
def score_segments(
    scorer: dnsmos.ComputeScore, audio: AudioDict, segments: List[Segment], base_sr: int
) -> Tuple[float, List[Segment]]:
    """Attach DNSMOS scores to each segment and return the average."""
    if not segments:
        Logger.get_logger().info("score_segments: received 0 segments, skipping MOS.")
        return 0.0, []

    waveform = librosa.resample(
        audio["waveform"], orig_sr=base_sr, target_sr=16000
    )

    for segment in tqdm.tqdm(segments, desc="DNSMOS"):
        start = int(segment["start"] * 16000)
        end = int(segment["end"] * 16000)
        clip = waveform[start:end]
        segment["dnsmos"] = scorer(clip, 16000, False)["OVRL"]

    mean_score = float(np.mean([seg["dnsmos"] for seg in segments]))
    return mean_score, segments


def filter_segments(segments: List[Segment], settings: Dict[str, Any]) -> List[Segment]:
    """Filter by duration, MOS, and per-character timing heuristics."""
    if not segments:
        Logger.get_logger().info("filter_segments: no segments to filter.")
        return []

    min_duration = settings.get("min_duration", 0.25)
    max_duration = settings.get("max_duration", 30)
    min_dnsmos = settings.get("min_dnsmos", 3)
    min_char_count = settings.get("min_char_count", 2)

    filtered_stats, _ = calculate_audio_stats(
        segments,
        min_duration=min_duration,
        max_duration=max_duration,
        min_dnsmos=min_dnsmos,
        min_char_count=min_char_count,
    )
    filtered = [segments[idx] for idx, _ in filtered_stats]
    Logger.get_logger().info(
        "filter_segments: kept %d/%d segments after heuristics.",
        len(filtered),
        len(segments),
    )
    return filtered


def export_results(audio: AudioDict, segments: List[Segment], save_dir: Path) -> Path:
    """Write MP3 clips and JSON manifest to the specified directory."""
    save_dir.mkdir(parents=True, exist_ok=True)
    export_to_mp3(audio, segments, str(save_dir), save_dir.name)

    manifest = save_dir / f"{save_dir.name}.json"
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(segments, handle, ensure_ascii=False)
    return manifest


def _cleanup_processed_artifacts(manifest_path: Path, *, logger: Optional[Any] = None) -> None:
    """
    Remove the per-file processed directory (and its parent *_processed folder if empty).
    Mirrors the Gradio-side cleanup to keep disk usage manageable when intermediate
    artifacts are no longer needed.
    """
    manifest_dir = manifest_path.parent
    if not manifest_dir.exists():
        return

    parent_dir = manifest_dir.parent
    if not parent_dir.name.endswith("_processed"):
        return

    log = logger or Logger.get_logger()
    try:
        shutil.rmtree(manifest_dir, ignore_errors=True)
        if parent_dir.exists() and not any(parent_dir.iterdir()):
            shutil.rmtree(parent_dir, ignore_errors=True)
        log.debug("Removed processed artifacts under %s", manifest_dir)
    except Exception as exc:
        log.warning("Failed to clean processed folder %s: %s", manifest_dir, exc)


def process_audio(
    audio_path: Path,
    *,
    save_root: Path,
    sample_rate: int,
    models: Dict[str, Any],
    multilingual: bool,
    supported_languages: List[str],
    batch_size: int,
    filter_settings: Dict[str, Any],
    forced_language: Optional[str] = None,
    hash_names: bool = False,
) -> Tuple[Path, List[Segment]]:
    """Run the full Emilia pipeline for a single audio file."""
    logger = Logger.get_logger()

    output_name = _derive_output_name(audio_path, hash_names)
    if hash_names:
        logger.info("Hashing basename %s -> %s", audio_path.stem, output_name)

    output_dir = save_root / output_name
    debug_dir = output_dir / "debug"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    raw_waveform, raw_sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    write_debug_audio(debug_dir / "step0_original.wav", raw_waveform, raw_sample_rate)

    if models["separator"]:
        audio = separate_sources(models["separator"], str(audio_path), sample_rate)
        audio["name"] = f"{output_name}{audio_path.suffix}"
        write_debug_audio(debug_dir / "step1_vocals.wav", audio["waveform"], sample_rate)
        if "instrumental_waveform" in audio:
            write_debug_audio(
                debug_dir / "step1_instrumental.wav",
                audio["instrumental_waveform"],
                sample_rate,
            )
    else:
        logger.info("UVR separator not configured; using original audio for downstream steps.")
        if raw_sample_rate != sample_rate:
            vocal_waveform = librosa.resample(raw_waveform, orig_sr=raw_sample_rate, target_sr=sample_rate)
        else:
            vocal_waveform = raw_waveform
        vocal_normalized, norm_sr = normalize_waveform(vocal_waveform, sample_rate, sample_rate)
        audio = {
            "waveform": vocal_normalized,
            "sample_rate": norm_sr,
            "name": f"{output_name}{audio_path.suffix}",
        }
        write_debug_audio(debug_dir / "step1_vocals.wav", audio["waveform"], norm_sr)
    
    logger.info("Running diarization...")
    diarisation = diarise_speakers(models["diarisation"], audio, models["device"])
    logger.info("process_audio: diarization produced %d segments.", len(diarisation))
    vad_segments = models["vad"].vad(diarisation, audio)
    logger.info("process_audio: VAD produced %d segments.", len(vad_segments))
    dump_segments_audio(
        "step2_vad_raw", vad_segments, audio["waveform"], audio["sample_rate"], debug_dir
    )
    refined_segments = merge_vad_segments(
        vad_segments,
        merge_target=3.0,
        keep_min_len=filter_settings.get("min_duration", 0.25),
        max_len=filter_settings.get("max_duration", 30),
    )
    logger.info(
        "process_audio: segments after merge/post-process: %d.", len(refined_segments)
    )
    dump_segments_audio(
        "step3_vad_merged",
        refined_segments,
        audio["waveform"],
        audio["sample_rate"],
        debug_dir,
    )
    transcripts = run_asr(
        models["asr"],
        refined_segments,
        audio,
        multilingual=multilingual,
        supported_languages=supported_languages,
        batch_size=batch_size,
        forced_language=forced_language,
    )
    logger.info(
        "process_audio: ASR sample transcripts: %s",
        [
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg.get("speaker"),
                "language": seg.get("language"),
                "text": seg.get("text"),
            }
            for seg in transcripts[:5]
        ],
    )
    logger.info("process_audio: ASR produced %d segments.", len(transcripts))
    dump_segments_audio(
        "step4_asr_passed",
        transcripts,
        audio["waveform"],
        audio["sample_rate"],
        debug_dir,
    )
    _, scored_segments = score_segments(models["dnsmos"], audio, transcripts, sample_rate)
    logger.info(
        "process_audio: scored segments: %s",
        [
            {
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["end"] - seg["start"],
                "language": seg.get("language"),
                "dnsmos": seg.get("dnsmos"),
                "text": seg.get("text"),
            }
            for seg in scored_segments
        ],
    )
    filtered = filter_segments(scored_segments, filter_settings)
    dump_segments_audio(
        "step5_postfilter_kept",
        filtered,
        audio["waveform"],
        audio["sample_rate"],
        debug_dir,
    )
    kept_keys = {
        (round(seg["start"], 3), round(seg["end"], 3)) for seg in filtered
    }
    pruned_segments = [
        seg
        for seg in scored_segments
        if (round(seg["start"], 3), round(seg["end"], 3)) not in kept_keys
    ]
    dump_segments_audio(
        "step5_postfilter_pruned",
        pruned_segments,
        audio["waveform"],
        audio["sample_rate"],
        debug_dir,
    )
    logger.info("process_audio: final kept segments: %d.", len(filtered))
    manifest = export_results(audio, filtered, output_dir)
    return manifest, filtered


def prepare_models(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Load all required models based on the configuration."""
    logger = Logger.get_logger()

    cache_root = Path(
        cfg.get("download_cache")
        or Path(cfg["separate"]["step1"]["model_path"]).resolve().parent
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_map = {
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface",
        "TRANSFORMERS_CACHE": cache_root / "huggingface",
        "PYTORCH_PRETRAINED_BERT_CACHE": cache_root / "huggingface",
        "PYANNOTE_AUDIO_CACHE": cache_root / "pyannote",
        "TORCH_HOME": cache_root / "torch_hub",
        "CT2_CACHE_DIR": cache_root / "ctranslate2",
        "WHISPER_CACHE_DIR": cache_root / "whisper",
    }

    for env_var, path in cache_map.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env_var] = str(path)

    logger.info("Using Emilia cache directory at %s", cache_root)

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = torch.device("cuda")
        device_name = "cuda"
        logger.info("GPU detected, using CUDA execution.")
    else:
        device = torch.device("cpu")
        device_name = "cpu"
        logger.info("GPU not available, running on CPU.")
        args.compute_type = "int8"

    check_env(logger)

    hf_token = cfg["huggingface_token"]
    if not hf_token.startswith("hf"):
        raise ValueError(
            "A valid Hugging Face token is required for pyannote diarisation."
        )

    diarisation = Pipeline.from_pretrained(
        # "pyannote/speaker-diarization-3.1",
        "pyannote/speaker-diarization-community-1", # new
        token=hf_token,
        cache_dir=str(cache_map["PYANNOTE_AUDIO_CACHE"]),

    )
    diarisation.to(device)

    asr_model = whisper_asr.load_asr_model(
        args.whisper_arch,
        device_name,
        compute_type=args.compute_type,
        threads=args.threads,
        download_root=str(cache_map["WHISPER_CACHE_DIR"]),
        asr_options={
            "initial_prompt": (
                "Um, Uh, Ah, Like, you know, I mean, right, actually, basically."
            )
        },
    )

    vad_model = silero_vad.SileroVAD(device=device)
    
    do_uvr = args.do_uvr

    if do_uvr:
        logger.info("UVR enabled, running vocal separation")
        separator = UVRSeparator(
            cfg["separate"]["step1"]["model_path"],
            metadata_json=cfg["separate"]["step1"].get("metadata_json"),
        )
    else:
        logger.info("UVR disabled, skipping vocal separation.")
        separator = None
        

    dnsmos_model = dnsmos.ComputeScore(
        cfg["mos_model"]["primary_model_path"], device_name
    )

    return {
        "device": device,
        "diarisation": diarisation,
        "asr": asr_model,
        "vad": vad_model,
        "separator": separator,
        "dnsmos": dnsmos_model,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Emilia preprocessing pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="Emilia/config.json",
        help="Path to the Emilia configuration JSON.",
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default=None,
        help="Optional override for the input folder containing raw audio.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for WhisperX transcription.",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16",
        help="Compute type for WhisperX (auto-overridden on CPU).",
    )
    parser.add_argument(
        "--whisper-arch",
        type=str,
        default="medium",
        help="Whisper model size to load (e.g., small, medium, large-v3).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU threads to allocate per Whisper worker.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force Whisper transcription language (e.g., en, ja).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Override minimum segment duration (seconds) to keep after VAD filtering.",
    )
    parser.add_argument(
        "--hash-names",
        action="store_true",
        help="Use deterministic file-hash IDs instead of original filenames.",
    )
    parser.add_argument(
        "--keep-processed",
        dest="keep_processed",
        action="store_true",
        help="Retain the Emilia *_processed folders after export (default).",
    )
    parser.add_argument(
        "--cleanup-processed",
        dest="keep_processed",
        action="store_false",
        help="Delete Emilia *_processed folders after results are consumed.",
    )
    parser.set_defaults(keep_processed=True)
    return parser.parse_args()


def run_emilia_pipeline(
    config: Union[str, Path],
    *,
    input_folder: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    compute_type: str = "float16",
    whisper_arch: str = "medium",
    threads: int = 4,
    do_uvr: bool = True,
    min_duration: Optional[float] = None,
    forced_language: Optional[str] = None,
    hash_names: bool = False,
    emilia_keep_processed: bool = True,
    selected_files: Optional[List[Union[str, Path]]] = None,
    on_result: Optional[Callable[[Path, Path, List[Segment]], None]] = None,
) -> List[Tuple[Path, List[Segment]]]:
    """
    Programmatic entry point for running the Emilia preprocessing pipeline.

    Args:
        config: Path to the Emilia configuration JSON file.
        input_folder: Optional override path for raw audio input.
        batch_size: WhisperX transcription batch size.
        compute_type: WhisperX compute precision (auto-overridden on CPU).
        whisper_arch: WhisperX model identifier (e.g., small, medium, large-v3).
        threads: CPU threads allocated per Whisper worker.
        do_uvr: run UVR (background removal) on each clip
        min_duration: override minimum allowed segment duration (seconds)
        forced_language: override Whisper language (disables multilingual detection)
        hash_names: derive output IDs from file hashes instead of original names
        emilia_keep_processed: keep the *_processed folders (set False to delete after use)
        selected_files: optional subset of files (by stem/path) to process
        on_result: optional callback invoked after each file is processed

    Returns:
        A list with one entry per input audio file containing a tuple of
        (manifest_path, filtered_segments).
    """
    config_path = Path(config).resolve()
    cfg = load_cfg(str(config_path))

    resolved_input = input_folder or cfg["entrypoint"]["input_folder_path"]
    cfg["entrypoint"]["input_folder_path"] = str(
        _resolve_path(config_path, resolved_input)
    )
    cfg["separate"]["step1"]["model_path"] = str(
        _resolve_path(config_path, cfg["separate"]["step1"]["model_path"])
    )
    cfg["mos_model"]["primary_model_path"] = str(
        _resolve_path(config_path, cfg["mos_model"]["primary_model_path"])
    )
    if forced_language:
        cfg["language"]["multilingual"] = False
        cfg["language"]["supported"] = [forced_language]

    Logger.init_logger("emilia_pipeline")
    logger = Logger.get_logger()

    runtime_args = SimpleNamespace(
        batch_size=batch_size,
        compute_type=compute_type,
        whisper_arch=whisper_arch,
        threads=threads,
        do_uvr=do_uvr
    )
    models = prepare_models(cfg, runtime_args)

    input_root = Path(cfg["entrypoint"]["input_folder_path"])
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    sample_rate = cfg["entrypoint"]["SAMPLE_RATE"]
    multilingual = cfg["language"]["multilingual"]
    supported_languages = cfg["language"]["supported"]
    filter_settings = cfg.get(
        "filters",
        {
            "min_duration": 0.25,
            "max_duration": 30,
            "min_dnsmos": 3,
            "min_char_count": 2,
        },
    )
    if min_duration is not None:
        filter_settings["min_duration"] = max(0.0, float(min_duration))

    audio_paths = [Path(path) for path in get_audio_files(str(input_root))]
    if not audio_paths:
        logger.warning("No audio files discovered in %s", input_root)
        return []
    if selected_files is not None:
        include_stems = {Path(sf).stem for sf in selected_files}
        audio_paths = [path for path in audio_paths if path.stem in include_stems]
        if not audio_paths:
            logger.warning("Selected Emilia files did not match any inputs in %s", input_root)
            return []

    results: List[Tuple[Path, List[Segment]]] = []
    for audio_path in audio_paths:
        logger.info("Processing %s", audio_path)
        output_root = audio_path.parent.parent / f"{audio_path.parent.name}_processed"
        manifest, segments = process_audio(
            audio_path,
            save_root=output_root,
            sample_rate=sample_rate,
            models=models,
            multilingual=multilingual,
            supported_languages=supported_languages,
            batch_size=batch_size,
            filter_settings=filter_settings,
            forced_language=forced_language,
            hash_names=hash_names,
        )
        logger.info(
            "Saved %d filtered segments to %s",
            len(segments),
            manifest,
        )
        if on_result:
            on_result(audio_path, manifest, segments)
        results.append((manifest, segments))
        if not emilia_keep_processed:
            _cleanup_processed_artifacts(manifest, logger=logger)
    return results


def main() -> None:
    args = parse_args()
    run_emilia_pipeline(
        args.config,
        input_folder=args.input_folder,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        whisper_arch=args.whisper_arch,
        threads=args.threads,
        min_duration=args.min_duration,
        forced_language=args.language,
        hash_names=args.hash_names,
        emilia_keep_processed=args.keep_processed,
    )


if __name__ == "__main__":
    main()
