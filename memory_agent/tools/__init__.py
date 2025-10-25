"""Tool registry for the memory agent."""

from __future__ import annotations

from typing import Dict

from .base import ToolBase, ToolContext
from .people_by_location import FindPeopleByLocationTool
from .people_by_organization import FindPeopleByOrganizationTool
from .people_by_topic import FindPeopleByTopicTool
from .person_profile import GetPersonProfileTool
from .person_timeline import GetPersonTimelineTool
from .relationships_between import GetRelationshipsBetweenTool
from .semantic_search import SemanticSearchFactsTool


def build_toolkit(context: ToolContext) -> Dict[str, ToolBase]:
    """Instantiate all available tools."""
    tools: Dict[str, ToolBase] = {
        "get_person_profile": GetPersonProfileTool(context),
        "find_people_by_organization": FindPeopleByOrganizationTool(context),
        "get_relationships_between": GetRelationshipsBetweenTool(context),
        "find_people_by_topic": FindPeopleByTopicTool(context),
        "get_person_timeline": GetPersonTimelineTool(context),
        "find_people_by_location": FindPeopleByLocationTool(context),
        "semantic_search_facts": SemanticSearchFactsTool(context),
    }
    return tools


__all__ = ["ToolContext", "ToolBase", "build_toolkit"]
