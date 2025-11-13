from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from typing import TextIO

from .models import DeduplicationStats, Partition


class ProgressDisplay:
    """Render a lightweight CLI progress bar with ETA and averages."""

    def __init__(
        self,
        *,
        enabled: bool,
        total_partitions: int,
        base_processed: int = 0,
        base_facts: int = 0,
        base_groups: int = 0,
        base_merged: int = 0,
        started_at: datetime | None = None,
        refresh_interval: float = 1.0,
        stream: TextIO | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._enabled = enabled
        self._total_partitions = total_partitions
        self._base_processed = base_processed
        self._base_facts = base_facts
        self._base_groups = base_groups
        self._base_merged = base_merged
        self._refresh_interval = max(0.2, refresh_interval)
        self._stream = stream or sys.stderr
        self._logger = logger or logging.getLogger(__name__)
        self._use_tty = bool(self._enabled and self._stream and self._stream.isatty())
        self._log_only = bool(self._enabled and not self._use_tty)
        self._line_active = False
        self._last_render = 0.0
        self._last_log = 0.0
        self._current_partition: str | None = None
        self._current_fact_count: int | None = None
        self._started_at = self._normalize_start(started_at)
        self._latest_stats: DeduplicationStats | None = None
        self._last_line_text: str = ""
        self._needs_redraw = False

    def set_active_partition(self, partition: Partition | None, fact_count: int | None = None) -> None:
        if not self._enabled:
            return
        if partition is None:
            self._current_partition = None
            self._current_fact_count = None
            return
        self._current_partition = f"{partition.fact_type.value}/{partition.subject_id}"
        self._current_fact_count = fact_count

    def update(self, stats: DeduplicationStats, *, force: bool = False) -> None:
        if not self._enabled:
            return
        self._latest_stats = stats
        now = time.monotonic()
        if not force and (now - self._last_render) < self._refresh_interval:
            return
        self._last_render = now
        line = self._format_line(stats)
        if self._use_tty:
            self._stream.write("\r" + line)
            self._stream.flush()
            self._line_active = True
            self._last_line_text = line
            self._needs_redraw = False
            return

        if self._log_only and (force or (now - self._last_log) >= max(self._refresh_interval, 5.0)):
            self._logger.info(line)
            self._last_log = now

    def finish(self, stats: DeduplicationStats) -> None:
        if not self._enabled:
            return
        self.update(stats, force=True)
        if self._use_tty and self._line_active:
            self._stream.write("\n")
            self._stream.flush()
            self._line_active = False
            self._last_line_text = ""

    def _format_line(self, stats: DeduplicationStats) -> str:
        total_processed = self._base_processed + stats.processed_partitions
        total_facts = self._base_facts + stats.facts_processed
        total_groups = self._base_groups + stats.candidate_groups_processed
        total_merged = self._base_merged + stats.facts_merged
        total_partitions = self._total_partitions or stats.total_partitions

        percent, bar = self._build_bar(total_processed, total_partitions)
        elapsed = self._elapsed_seconds()
        eta = self._estimate_eta(elapsed, total_processed, total_partitions)
        avg_facts = self._safe_div(total_facts, total_processed)
        avg_groups = self._safe_div(total_groups, total_processed)

        segments = [
            bar,
            f"{percent:5s}",
            f"partitions {total_processed}/{total_partitions or '?'}",
            f"elapsed {self._format_duration(elapsed)}",
            f"eta {self._format_duration(eta) if eta is not None else '--:--'}",
            f"avg facts/part {avg_facts:.1f}" if total_processed else "avg facts/part --",
            f"avg groups/part {avg_groups:.1f}" if total_processed else "avg groups/part --",
            f"merged {total_merged}",
        ]

        if self._current_partition:
            if self._current_fact_count is not None:
                current_segment = (
                    f"running {self._current_partition} ({self._current_fact_count} facts)"
                )
            else:
                current_segment = f"running {self._current_partition}"
            segments.append(current_segment)

        return " | ".join(segments)

    def _build_bar(self, processed: int, total: int) -> tuple[str, str]:
        if total and total > 0:
            ratio = max(0.0, min(1.0, processed / total))
            percent = f"{ratio * 100:4.1f}%"
        else:
            ratio = 0.0
            percent = " ---%"
        width = 20
        filled = int(ratio * width)
        filled_segment = "#" * filled
        bar = f"[{filled_segment.ljust(width, '-')}]"
        return percent, bar

    def _elapsed_seconds(self) -> float:
        now = datetime.now(timezone.utc)
        return max(0.0, (now - self._started_at).total_seconds())

    def _estimate_eta(self, elapsed: float, processed: int, total: int) -> float | None:
        if not total or total <= 0 or processed <= 0:
            return None
        remaining = max(total - processed, 0)
        if remaining == 0:
            return 0.0
        rate = elapsed / processed
        if rate <= 0:
            return None
        return remaining * rate

    @staticmethod
    def _safe_div(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None or seconds < 0:
            return "--:--"
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 99:
            return f"{hours:03d}:{minutes:02d}:{secs:02d}"
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _normalize_start(started_at: datetime | None) -> datetime:
        if started_at is None:
            return datetime.now(timezone.utc)
        if started_at.tzinfo is None:
            return started_at.replace(tzinfo=timezone.utc)
        return started_at.astimezone(timezone.utc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def requires_log_cooperation(self) -> bool:
        return self._enabled and self._use_tty

    def pause(self) -> None:
        if not self.requires_log_cooperation or not self._line_active:
            return
        blank = " " * len(self._last_line_text)
        self._stream.write("\r" + blank + "\r")
        self._stream.flush()
        self._line_active = False
        self._needs_redraw = True

    def resume(self) -> None:
        if not self.requires_log_cooperation:
            return
        if self._needs_redraw and self._latest_stats is not None:
            self.update(self._latest_stats, force=True)
        self._needs_redraw = False
