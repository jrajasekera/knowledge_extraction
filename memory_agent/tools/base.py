"""Base classes and helpers for Neo4j-backed tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from neo4j import Driver, Session
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolContext:
    """Shared resources needed by tool implementations."""

    driver: Driver
    embeddings_model: Any | None = None

    def session(self) -> Session:
        """Create a Neo4j session."""
        return self.driver.session()


class ToolError(RuntimeError):
    """Raised when a tool encounters an unrecoverable error."""


class ToolBase[InputT: BaseModel, OutputT: BaseModel]:
    """Abstract base class enforcing a common execution contract."""

    name: str

    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self.name = self.__class__.__name__

    input_model: type[InputT]
    output_model: type[OutputT]

    def __call__(self, payload: dict[str, Any]) -> OutputT:
        """Validate and execute the tool."""
        input_data = self.input_model.model_validate(payload)
        logger.debug("Executing tool %s with input %s", self.name, input_data.model_dump())
        try:
            result = self.run(input_data)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s failed: %s", self.name, exc)
            raise ToolError(str(exc)) from exc
        return self.output_model.model_validate(result)

    def run(self, input_data: InputT) -> OutputT:
        """Execute tool logic. Must be implemented by subclasses."""
        raise NotImplementedError
