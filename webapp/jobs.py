from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from threading import Lock

logger = logging.getLogger(__name__)


class RunQueue:
    def __init__(self, workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="consolidation")
        self._pending: set[Future[None]] = set()
        self._lock = Lock()

    def submit(self, job: Callable[[], None]) -> Future[None]:
        future = self._executor.submit(job)
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(self._finished)
        return future

    def join(self, timeout: float | None = None) -> bool:
        with self._lock:
            pending = list(self._pending)
        _, unfinished = wait(pending, timeout)
        return not unfinished

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _finished(self, future: Future[None]) -> None:
        with self._lock:
            self._pending.discard(future)
        if not future.cancelled() and future.exception() is not None:
            logger.error("Фонове завдання завершилося з помилкою", exc_info=future.exception())
