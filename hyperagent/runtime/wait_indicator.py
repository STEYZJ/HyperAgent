"""Small wait indicators for blocking model calls."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar


T = TypeVar("T")


class WaitIndicator:
    """UI-neutral progress hook for a blocking operation."""

    def start(self, elapsed_sec: float = 0.0) -> None:
        pass

    def update(self, elapsed_sec: float) -> None:
        pass

    def finish(self, elapsed_sec: float, *, failed: bool = False) -> None:
        pass


class NullWaitIndicator(WaitIndicator):
    """No-op indicator for tests and non-interactive callers."""


class ConsoleWaitIndicator(WaitIndicator):
    """Line-based indicator for the stdlib REPL."""

    def __init__(self, output: Callable[[str], None]) -> None:
        self.output = output

    def start(self, elapsed_sec: float = 0.0) -> None:
        self.output(f"思考中...... {elapsed_sec:.0f}s")

    def update(self, elapsed_sec: float) -> None:
        self.output(f"思考中...... {elapsed_sec:.0f}s")

    def finish(self, elapsed_sec: float, *, failed: bool = False) -> None:
        label = "思考失败" if failed else "思考完成"
        self.output(f"{label}，用时 {elapsed_sec:.1f}s")


@dataclass
class TimedOperationResult(Generic[T]):
    value: T
    elapsed_sec: float


def run_with_wait_indicator(
    operation: Callable[[], T],
    indicator: Optional[WaitIndicator] = None,
    *,
    tick_sec: float = 1.0,
) -> TimedOperationResult[T]:
    """Run a blocking operation while the caller can refresh UI progress."""

    active_indicator = indicator or NullWaitIndicator()
    start = time.monotonic()
    active_indicator.start(0.0)
    result: dict[str, T] = {}
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result["value"] = operation()
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller thread.
            error.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(max(float(tick_sec), 0.05))
        elapsed = time.monotonic() - start
        if thread.is_alive():
            active_indicator.update(elapsed)
    elapsed = time.monotonic() - start
    if error:
        active_indicator.finish(elapsed, failed=True)
        raise error[0]
    active_indicator.finish(elapsed, failed=False)
    return TimedOperationResult(value=result["value"], elapsed_sec=elapsed)
