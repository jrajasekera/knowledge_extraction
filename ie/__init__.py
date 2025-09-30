"""Information extraction utilities for Discord knowledge extraction."""

from .client import LlamaServerClient, LlamaServerConfig
from .config import FACT_DEFINITIONS, DEFAULT_FACT_TYPES, IEConfig
from .models import ExtractionFact, ExtractionResult
from .runner import IERunStats, reset_ie_progress, run_ie_job
from .types import FactAttribute, FactDefinition, FactType
from .windowing import MessageRecord, MessageWindow, WindowBuilder, iter_message_windows

__all__ = [
    "LlamaServerClient",
    "LlamaServerConfig",
    "FactAttribute",
    "FactDefinition",
    "FactType",
    "FACT_DEFINITIONS",
    "DEFAULT_FACT_TYPES",
    "IEConfig",
    "ExtractionFact",
    "ExtractionResult",
    "IERunStats",
    "run_ie_job",
    "reset_ie_progress",
    "MessageRecord",
    "MessageWindow",
    "WindowBuilder",
    "iter_message_windows",
]
