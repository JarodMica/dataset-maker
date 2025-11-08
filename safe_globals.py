"""
Central registration for objects required by torch.load during Whisper/Emilia deserialization.

Import this module and call `register_torch_safe_globals()` before loading
any pickled checkpoints. The helper is safe to call multiple times.
"""

from __future__ import annotations

import builtins
import importlib
import logging
from typing import Iterable, Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_SAFE_GLOBALS = [
    "omegaconf.listconfig.ListConfig",
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.base.ContainerMetadata",
    "omegaconf.base.Metadata",
    "omegaconf.nodes.AnyNode",
    "omegaconf.omegaconf.OmegaConf",
    "torch.torch_version.TorchVersion",
    "pyannote.audio.core.task.Specifications",
    "pyannote.audio.core.task.Problem",
    "pyannote.audio.core.task.Resolution",
    "pyannote.audio.core.model.Introspection",
    "typing.Any",
    "collections.defaultdict",
    "builtins.list",
    "builtins.dict",
    "builtins.int",
]


def _resolve_symbol(qualname: str):
    module_name, attr_name = qualname.rsplit(".", 1)
    if module_name == "builtins":
        return getattr(builtins, attr_name)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def register_torch_safe_globals(extra_symbols: Optional[Iterable[str]] = None) -> None:
    symbols = list(DEFAULT_SAFE_GLOBALS)
    if extra_symbols:
        symbols.extend(extra_symbols)

    for qualname in symbols:
        try:
            obj = _resolve_symbol(qualname)
        except Exception as exc:
            logger.warning("Unable to resolve %s for torch safe globals: %s", qualname, exc)
            continue
        torch.serialization.add_safe_globals([obj])


# Automatically register on import so any module that imports safe_globals is protected.
register_torch_safe_globals()
