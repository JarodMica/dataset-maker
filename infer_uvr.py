#!/usr/bin/env python3
"""Run MDX-Net inference via Ultimate Vocal Remover helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import types

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

BASE_DIR = Path(__file__).resolve().parent
UVR_LOCAL_ROOT = BASE_DIR / "ultimatevocalremovergui"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if UVR_LOCAL_ROOT.is_dir() and str(UVR_LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(UVR_LOCAL_ROOT))

try:
    import pyrubberband  # type: ignore
except ModuleNotFoundError:
    try:
        import importlib

        pyrb_local = importlib.import_module("lib_v5.pyrb")
        pyrubberband = types.ModuleType("pyrubberband")  # type: ignore[assignment]
        pyrubberband.pyrb = pyrb_local  # type: ignore[attr-defined]
        sys.modules["pyrubberband"] = pyrubberband  # type: ignore[arg-type]
    except ModuleNotFoundError:
        pass

try:
    import torch_directml  # type: ignore
except ModuleNotFoundError:
    torch_directml = types.ModuleType("torch_directml")  # type: ignore[assignment]
    torch_directml.is_available = lambda: False  # type: ignore[attr-defined]
    torch_directml.device = lambda: "privateuseone:0"  # type: ignore[attr-defined]
    sys.modules["torch_directml"] = torch_directml  # type: ignore[arg-type]

from ultimatevocalremovergui.gui_data.constants import (
    DEFAULT,
    MDX_ARCH_TYPE,
    VOCAL_STEM,
    WAV,
    secondary_stem,
)


def compute_partial_md5(model_path: Path) -> str:
    """Reproduce UVR's partial hash calculation for MDX models."""
    with model_path.open("rb") as fh:
        try:
            fh.seek(-10000 * 1024, 2)
        except OSError:
            fh.seek(0)
        chunk = fh.read()
    return hashlib.md5(chunk).hexdigest()


def load_mdx_metadata(model_path: Path, metadata_path: Path | None = None) -> Dict[str, Any]:
    """Load model metadata (compensate, dims, fft, etc.) using the UVR hash lookup."""
    candidate_paths = []
    if metadata_path is not None:
        candidate_paths.append(Path(metadata_path))

    candidate_paths.append(model_path.parent / "model_data" / "model_data.json")
    candidate_paths.append(Path("MDX_Net_Models") / "model_data" / "model_data.json")
    candidate_paths.append(Path("MDX_Net_Models") / "model_data.json")
    candidate_paths.append(
        Path(__file__).resolve().parent
        / "ultimatevocalremovergui"
        / "models"
        / "MDX_Net_Models"
        / "model_data"
        / "model_data.json"
    )

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidate_paths.append(
            Path(local_appdata)
            / "Programs"
            / "Ultimate Vocal Remover"
            / "models"
            / "MDX_Net_Models"
            / "model_data"
            / "model_data.json"
        )

    metadata_file = next((p for p in candidate_paths if p.is_file()), None)
    if metadata_file is None:
        raise FileNotFoundError(
            "Could not locate MDX metadata file. Checked:\n"
            + "\n".join(str(p) for p in candidate_paths)
        )

    data: Dict[str, Dict[str, Any]] = json.loads(metadata_file.read_text())
    model_hash = compute_partial_md5(model_path)

    entry = data.get(model_hash)
    if entry is None:
        raise KeyError(
            f"Model hash {model_hash} was not found in {metadata_path}. "
            "Update the metadata or supply values manually."
        )

    entry = dict(entry)
    entry["model_hash"] = model_hash
    return entry


class SimpleMDXModel:
    """Lightweight stand-in for UVR's ModelData tailored to single MDX inference."""

    def __init__(self, model_path: Path, metadata: Dict[str, Any], mixer_path: Path):
        primary_stem = metadata.get("primary_stem", VOCAL_STEM)

        # Paths & identifiers
        self.model_path = str(model_path)
        self.model_name = model_path.name
        self.model_basename = model_path.stem
        self.process_method = MDX_ARCH_TYPE
        self.model_hash = metadata.get("model_hash")

        # Core MDX config
        self.compensate = float(metadata["compensate"])
        self.mdx_dim_f_set = int(metadata["mdx_dim_f_set"])
        self.mdx_dim_t_set = int(metadata["mdx_dim_t_set"])
        self.mdx_n_fft_scale_set = int(metadata["mdx_n_fft_scale_set"])
        self.mdx_batch_size = 1
        self.mdx_segment_size = 256
        self.margin = 44100
        self.chunks = 0
        self.mdx_c_configs = None
        self.mdxnet_stem_select = primary_stem
        self.mixer_path = str(mixer_path)
        self.mdx_model_stems: list[str] = []
        self.mdx_stem_count = 1
        self.is_mdx_c = False
        self.is_mdx_c_seg_def = False
        self.is_mdx_ckpt = False
        self.is_mdx_combine_stems = False
        self.is_roformer = bool(metadata.get("is_roformer", False))
        self.roformer_config = None

        # Audio / export options
        self.model_samplerate = 44100
        self.model_capacity = None
        self.save_format = WAV
        self.wav_type_set = "PCM_16"
        self.mp3_bit_set = "320k"
        self.is_normalization = True
        self.is_invert_spec = False
        self.is_denoise = False
        self.is_denoise_model = False
        self.is_deverb_vocals = False
        self.deverb_vocal_opt = VOCAL_STEM
        self.is_save_vocal_only = False

        # Stem bookkeeping
        self.primary_stem = primary_stem
        self.primary_stem_native = primary_stem
        self.secondary_stem = secondary_stem(primary_stem)
        self.primary_model_primary_stem = primary_stem
        self.ensemble_primary_stem = primary_stem
        self.ensemble_secondary_stem = self.secondary_stem
        self.secondary_model = None
        self.secondary_model_scale = None
        self.secondary_model_4_stem: list[Any] = []
        self.secondary_model_4_stem_scale: list[Any] = []

        # Flags toggled in GUI but unused here
        self.is_primary_stem_only = False
        self.is_secondary_stem_only = False
        self.is_primary_model_primary_stem_only = False
        self.is_primary_model_secondary_stem_only = False
        self.is_secondary_model_activated = False
        self.is_secondary_model = False
        self.is_pre_proc_model = False
        self.is_ensemble_mode = False
        self.is_multi_stem_ensemble = False
        self.is_target_instrument = False
        self.is_vr_51_model = False
        self.is_mixer_mode = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.vocal_split_model = None
        self.is_vocal_split_model = False
        self.is_save_inst_vocal_splitter = False
        self.is_inst_only_voc_splitter = False
        self.is_source_swap = False
        self.is_karaoke = bool(metadata.get("is_karaoke", False))
        self.is_bv_model = bool(metadata.get("is_bv_model", False))
        self.bv_model_rebalance = bool(metadata.get("is_bv_model_rebalanced", False))
        self.is_sec_bv_rebalance = False
        self.is_pitch_change = False
        self.semitone_shift = 0.0
        self.is_match_frequency_pitch = False
        self.overlap = 0.25
        self.overlap_mdx = 0.25
        self.overlap_mdx23 = 8
        self.device_set = DEFAULT
        self.is_use_directml = False
        self.is_gpu_conversion = 0
        self.model_status = True
        self.mixer_path = str(mixer_path)
        self.DENOISER_MODEL = None
        self.DEVERBER_MODEL = None
        self.mdxnet_stem_select = primary_stem


def build_process_payload(
    input_path: Path,
    export_path: Path,
    model: SimpleMDXModel,
    *,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    console_callback: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Assemble the minimal payload expected by UVR's separation helpers."""

    audio_file_base = input_path.stem

    progress_state: Dict[str, Any] = {"last": 0.0, "bar": None}

    if progress_callback is None:
        bar = tqdm(total=1.0, desc="UVR Inference", unit="step", leave=False)
        progress_state["bar"] = bar

        def set_progress_bar(step: float, inference_iterations: float = 0.0) -> None:  # type: ignore[redefinition]
            progress = max(0.0, min(1.0, float(step) + float(inference_iterations)))
            last = progress_state["last"]
            delta = progress - last
            if delta >= 0:
                bar.update(delta)
            else:
                bar.n = progress
                bar.refresh()
            progress_state["last"] = progress
            if progress >= 1.0:
                bar.n = 1.0
                bar.refresh()
    else:
        def set_progress_bar(step: float, inference_iterations: float = 0.0) -> None:  # type: ignore[redefinition]
            progress_callback(step, inference_iterations)

    if console_callback is None:
        def write_to_console(message: str, base_text: str = "") -> None:  # type: ignore[redefinition]
            text = f"{base_text}{message}".strip()
            if not text:
                return
            bar = progress_state.get("bar")
            if bar is not None:
                bar.write(text)
            else:
                print(text)
    else:
        def write_to_console(message: str, base_text: str = "") -> None:  # type: ignore[redefinition]
            console_callback(message, base_text)

    def cached_source_callback(arch_type: str, model_name: str | None = None) -> Tuple[Any, Any]:
        _ = (arch_type, model_name)
        return None, None

    def cached_model_source_holder(*_args: Any, **_kwargs: Any) -> None:
        return None

    def cleanup() -> None:
        bar = progress_state.get("bar")
        if bar is not None:
            bar.n = max(bar.n, 1.0)
            bar.close()

    payload = {
        "export_path": str(export_path),
        "audio_file": str(input_path),
        "audio_file_base": audio_file_base,
        "set_progress_bar": set_progress_bar,
        "write_to_console": write_to_console,
        "process_iteration": lambda: None,
        "cached_source_callback": cached_source_callback,
        "cached_model_source_holder": cached_model_source_holder,
        "list_all_models": [model.model_basename],
        "is_ensemble_master": False,
        "is_4_stem_ensemble": False,
    }
    payload["_progress_finalizer"] = cleanup
    return payload


class UVRSeparator:
    """Reusable wrapper around UVR's MDX inference pipeline."""

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        metadata_json: Optional[Union[str, Path]] = None,
        mixer_path: Optional[Union[str, Path]] = None,
    ) -> None:
        ensure_demucs_stubs()
        self.model_path = Path(model_path).expanduser().resolve()
        self.metadata_json = Path(metadata_json).expanduser().resolve() if metadata_json else None
        self.mixer_path = (
            Path(mixer_path).expanduser().resolve()
            if mixer_path
            else resolve_mixer_path()
        )

        self._base_metadata = dict(load_mdx_metadata(self.model_path, self.metadata_json))
        template_model = SimpleMDXModel(
            self.model_path, dict(self._base_metadata), self.mixer_path
        )
        self.primary_stem_name = template_model.primary_stem or VOCAL_STEM
        self.secondary_stem_name = template_model.secondary_stem or secondary_stem(
            self.primary_stem_name
        )
        self._target_stem = self._determine_target(template_model, self.model_path)

    @staticmethod
    def _determine_target(model: SimpleMDXModel, model_path: Path) -> str:
        # Prefer explicit metadata
        candidate = (model.primary_stem or "").lower()
        if any(keyword in candidate for keyword in ("inst", "instrument", "accompaniment", "music", "other")):
            return "instrumental"
        if "vocal" in candidate:
            return "vocals"

        # Fallback to filename heuristics
        model_name = model_path.stem.lower()
        if any(keyword in model_name for keyword in ("inst", "instrument", "karaoke", "no_voc", "accom", "music")):
            return "instrumental"
        return "vocals"

    @property
    def target_stem(self) -> str:
        return self._target_stem

    def _create_model_data(self) -> SimpleMDXModel:
        metadata = dict(self._base_metadata)
        return SimpleMDXModel(self.model_path, metadata, self.mixer_path)

    def _run_inference(
        self,
        input_path: Path,
        export_path: Path,
        *,
        progress_callback: Optional[Callable[[float, float], None]] = None,
        console_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Tuple[SimpleMDXModel, Dict[str, Path]]:
        from ultimatevocalremovergui.separate import SeperateMDX, clear_gpu_cache

        model = self._create_model_data()
        payload = build_process_payload(
            input_path,
            export_path,
            model,
            progress_callback=progress_callback,
            console_callback=console_callback,
        )
        cleanup = payload.pop("_progress_finalizer", lambda: None)
        try:
            seperator = SeperateMDX(model, payload)
            seperator.seperate()
        finally:
            cleanup()
            clear_gpu_cache()

        audio_file_base = payload["audio_file_base"]
        stem_paths: Dict[str, Path] = {}
        for stem_file in export_path.glob(f"{audio_file_base}_(*).wav"):
            stem_name = stem_file.stem.rsplit("(", 1)[-1].rstrip(")")
            stem_paths[stem_name] = stem_file

        return model, stem_paths

    def separate_to_directory(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        *,
        progress_callback: Optional[Callable[[float, float], None]] = None,
        console_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Path]:
        input_path = Path(input_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        _, stem_paths = self._run_inference(
            input_path,
            output_dir,
            progress_callback=progress_callback,
            console_callback=console_callback,
        )
        return stem_paths

    def predict(self, mix: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict separated stems for an in-memory mixture.

        Returns:
            background, vocals
        """
        if mix.ndim == 1:
            mix = np.stack([mix, mix], axis=0)

        if mix.shape[0] > 2:
            mix = mix[:2]

        mix = np.asarray(mix, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_input = Path(tmpdir) / "input.wav"
            sf.write(tmp_input, mix.T, sample_rate, subtype="PCM_16")
            export_dir = Path(tmpdir) / "outputs"
            export_dir.mkdir(parents=True, exist_ok=True)
            model, stem_paths = self._run_inference(
                tmp_input,
                export_dir,
                progress_callback=None,
                console_callback=lambda *_args, **_kwargs: None,
            )

            if not stem_paths:
                raise RuntimeError("UVR separation produced no output stems.")

            primary_name = model.primary_stem or self.primary_stem_name
            secondary_name = model.secondary_stem or self.secondary_stem_name

            def load_stem(name: str) -> Tuple[np.ndarray, int]:
                path = stem_paths.get(name)
                if path is None:
                    raise RuntimeError(f"Expected stem '{name}' not produced by UVR.")
                audio, sr = sf.read(path, dtype="float32", always_2d=True)
                return audio, sr

            primary_audio, sr_out = load_stem(primary_name)
            secondary_audio, _ = load_stem(secondary_name)

            if primary_audio.shape != secondary_audio.shape:
                min_len = min(primary_audio.shape[0], secondary_audio.shape[0])
                primary_audio = primary_audio[:min_len]
                secondary_audio = secondary_audio[:min_len]

            if self.target_stem == "instrumental":
                background, vocals = primary_audio, secondary_audio
            else:
                background, vocals = secondary_audio, primary_audio

            if sr_out != sample_rate:
                import librosa

                def _resample_channels(arr: np.ndarray) -> np.ndarray:
                    # arr shape: (samples, channels)
                    channels = [
                        librosa.resample(ch, orig_sr=sr_out, target_sr=sample_rate)
                        for ch in arr.T
                    ]
                    min_len = min(len(ch) for ch in channels) if channels else 0
                    if min_len <= 0:
                        return np.zeros((0, arr.shape[1]), dtype=np.float32)
                    trimmed = [ch[:min_len] for ch in channels]
                    return np.stack(trimmed, axis=1).astype(np.float32)

                background = _resample_channels(background)
                vocals = _resample_channels(vocals)

            return background, vocals


def run_uvr_separation(
    audio: Union[str, Path, np.ndarray],
    *,
    model_path: Union[str, Path],
    sample_rate: Optional[int] = None,
    metadata_json: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper around :class:`UVRSeparator` returning `(background, vocals)`.

    When `audio` is a numpy array, supply the corresponding `sample_rate`.
    """
    separator = UVRSeparator(model_path, metadata_json=metadata_json)

    if isinstance(audio, (str, Path)):
        import librosa  # Lazy import to avoid mandatory dependency for CLI usage

        waveform, sr = librosa.load(str(audio), sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = np.stack([waveform, waveform])
        sample_rate = int(sr)
    else:
        waveform = np.asarray(audio, dtype=np.float32)
        if sample_rate is None:
            raise ValueError("sample_rate must be provided when audio is supplied as an array.")

    return separator.predict(waveform, sample_rate)


def resolve_mixer_path() -> Path:
    """Locate the bundled MDX mixer checkpoint."""
    mixer_path = Path(__file__).resolve().parent / "ultimatevocalremovergui" / "lib_v5" / "mixer.ckpt"
    if not mixer_path.is_file():
        raise FileNotFoundError(f"Missing mixer checkpoint at {mixer_path}")
    return mixer_path


def ensure_demucs_stubs() -> None:
    """
    Guarantee that modules expected by UVR's Demucs bridge are present.

    The MDX-only path used here never calls into Demucs, so light stubs are enough
    to satisfy the imports when the real dependency (or specific attributes) is
    unavailable.
    """

    import sys

    def _missing_demucs(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - defensive
        raise RuntimeError("Demucs functionality is not available in this environment.")

    try:
        import demucs  # type: ignore
    except ModuleNotFoundError:
        demucs = types.ModuleType("demucs")
        sys.modules["demucs"] = demucs

    try:
        from demucs import apply as apply_mod  # type: ignore
    except ModuleNotFoundError:
        apply_mod = types.ModuleType("demucs.apply")
        apply_mod.apply_model = _missing_demucs  # type: ignore[attr-defined]
        apply_mod.demucs_segments = _missing_demucs  # type: ignore[attr-defined]
        sys.modules["demucs.apply"] = apply_mod
        setattr(sys.modules["demucs"], "apply", apply_mod)
    else:
        if not hasattr(apply_mod, "demucs_segments"):
            apply_mod.demucs_segments = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]

    demucs_root = sys.modules.get("demucs")
    for submodule in ("hdemucs", "model_v2", "pretrained", "utils"):
        module_name = f"demucs.{submodule}"
        if module_name not in sys.modules:
            stub = types.ModuleType(module_name)
            sys.modules[module_name] = stub
            if demucs_root is not None and not hasattr(demucs_root, submodule):
                setattr(demucs_root, submodule, stub)
        else:
            stub = sys.modules[module_name]

        if submodule == "model_v2" and not hasattr(stub, "auto_load_demucs_model_v2"):
            stub.auto_load_demucs_model_v2 = _missing_demucs  # type: ignore[attr-defined]
        elif submodule == "pretrained" and not hasattr(stub, "get_model"):
            stub.get_model = _missing_demucs  # type: ignore[attr-defined]
        elif submodule == "utils":
            if not hasattr(stub, "apply_model_v1"):
                stub.apply_model_v1 = _missing_demucs  # type: ignore[attr-defined]
            if not hasattr(stub, "apply_model_v2"):
                stub.apply_model_v2 = _missing_demucs  # type: ignore[attr-defined]
        elif submodule == "hdemucs" and not hasattr(stub, "HDemucs"):
            class _HDemucs:  # pragma: no cover - defensive
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    raise RuntimeError("Demucs HDemucs model is not available.")

            stub.HDemucs = _HDemucs  # type: ignore[attr-defined]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    default_model = Path("MDX_Net_Models") / "Kim_Vocal_2.onnx"
    parser = argparse.ArgumentParser(description="Run MDX-Net inference using Ultimate Vocal Remover internals.")
    parser.add_argument("input", type=Path, help="Path to the audio file to separate.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model,
        help=f"Path to the MDX model (default: {default_model.as_posix()}).",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional override for model_data.json if it lives elsewhere.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("uvr_outputs"),
        help="Directory for separated stems.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found at {input_path}")

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    separator = UVRSeparator(
        model_path,
        metadata_json=args.metadata_json,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running MDX inference with model '{separator.model_path.stem}' on {input_path.name}")
    stem_paths = separator.separate_to_directory(
        input_path,
        output_dir,
        console_callback=lambda message, base_text="": print(f"{base_text}{message}"),
    )

    print("Inference complete. Generated stems:")
    for stem_name, stem_path in stem_paths.items():
        status = "created" if stem_path.is_file() else "missing"
        print(f"  {stem_path}: {status}")


if __name__ == "__main__":
    main()
