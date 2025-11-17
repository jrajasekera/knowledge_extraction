"""Tool registry for the memory agent."""

from __future__ import annotations

from typing import Dict

from .base import ToolBase, ToolContext
from .semantic_search import SemanticSearchFactsTool
from .semantic_search_messages import SemanticSearchMessagesTool


def build_toolkit(context: ToolContext) -> Dict[str, ToolBase]:
    """Instantiate all available tools."""
    tools: Dict[str, ToolBase] = {
        "semantic_search_facts": SemanticSearchFactsTool(context),
        "semantic_search_messages": SemanticSearchMessagesTool(context),
    }
    return tools


__all__ = ["ToolContext", "ToolBase", "build_toolkit"]
