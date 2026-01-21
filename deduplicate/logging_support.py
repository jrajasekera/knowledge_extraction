from __future__ import annotations

import logging
from collections.abc import Iterable

from .display import ProgressDisplay


class ProgressAwareHandler(logging.Handler):
    """Wrap another handler to pause/resume the progress display around logs."""

    def __init__(self, inner: logging.Handler, progress: ProgressDisplay) -> None:
        super().__init__(inner.level)
        self._inner = inner
        self._progress = progress

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - delegate doc
        if self._progress:
            self._progress.pause()
        try:
            self._inner.handle(record)
        finally:
            if self._progress:
                self._progress.resume()

    def setFormatter(self, formatter: logging.Formatter | None) -> None:
        self._inner.setFormatter(formatter)

    def setLevel(self, level: int) -> None:  # type: ignore[override]
        super().setLevel(level)
        self._inner.setLevel(level)

    def addFilter(self, filter: logging.Filter) -> None:  # noqa: A003 - match base signature
        self._inner.addFilter(filter)

    def removeFilter(self, filter: logging.Filter) -> None:
        self._inner.removeFilter(filter)

    def flush(self) -> None:
        self._inner.flush()

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            super().close()


class ProgressLoggingManager:
    """Context manager that wraps logger handlers to respect the progress display."""

    def __init__(self, progress: ProgressDisplay, logger: logging.Logger | None = None) -> None:
        self._progress = progress
        self._logger = logger or logging.getLogger()
        self._original_handlers: list[logging.Handler] | None = None

    def __enter__(self) -> ProgressLoggingManager:
        if not self._progress.requires_log_cooperation:
            return self
        original = list(self._logger.handlers)
        wrapped: list[logging.Handler] = []
        for handler in original:
            wrapped_handler = ProgressAwareHandler(handler, self._progress)
            wrapped_handler.setLevel(handler.level)
            formatter = handler.formatter
            if formatter is not None:
                wrapped_handler.setFormatter(formatter)
            self._copy_filters(handler.filters, wrapped_handler)
            wrapped.append(wrapped_handler)
        self._original_handlers = original
        self._logger.handlers = wrapped
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._original_handlers is not None:
            self._logger.handlers = self._original_handlers
            self._original_handlers = None

    @staticmethod
    def _copy_filters(filters: Iterable[logging.Filter], handler: logging.Handler) -> None:
        for filter_obj in filters:
            handler.addFilter(filter_obj)
