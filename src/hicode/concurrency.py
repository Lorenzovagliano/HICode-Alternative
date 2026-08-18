"""Small concurrency and atomic-file helpers shared by HiCode stages."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path


class ConcurrentTaskError(RuntimeError):
    """Raised after active work drains when a bounded task run has failures."""

    def __init__(self, stage, failures, completed, unscheduled):
        self.stage = stage
        self.failures = failures
        self.completed = completed
        self.unscheduled = unscheduled
        super().__init__(
            f"{stage} failed for {len(failures)} task(s); "
            f"completed={len(completed)} unscheduled={len(unscheduled)}"
        )

    def as_dict(self):
        return {
            "stage": self.stage,
            "failures": self.failures,
            "completed": self.completed,
            "unscheduled": self.unscheduled,
        }


def run_bounded_tasks(tasks, worker, max_workers, stage):
    """Run ordered ``(task_id, payload)`` items with bounded concurrency.

    The worker is responsible for durable checkpointing. Results are collected
    by task ID, independent of completion order. Once a task fails, no new
    tasks are scheduled, but active tasks are allowed to finish.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")

    tasks = list(tasks)
    results = {}
    failures = []
    futures = {}
    next_position = 0
    stopping = False

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        while next_position < len(tasks) and len(futures) < max_workers:
            task_id, payload = tasks[next_position]
            next_position += 1
            futures[executor.submit(worker, task_id, payload)] = task_id

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = futures.pop(future)
                try:
                    results[task_id] = future.result()
                except Exception as exc:
                    stopping = True
                    failures.append(
                        {
                            "task_id": task_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:1000],
                        }
                    )

            if stopping:
                continue

            for _ in done:
                if next_position >= len(tasks):
                    break
                task_id, payload = tasks[next_position]
                next_position += 1
                futures[executor.submit(worker, task_id, payload)] = task_id
    finally:
        executor.shutdown(wait=True)

    if failures:
        unscheduled = [task_id for task_id, _ in tasks[next_position:]]
        raise ConcurrentTaskError(
            stage,
            failures,
            sorted(results),
            unscheduled,
        )
    return results


def write_json_atomic(path, value):
    """Write JSON through a flushed sibling temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as target:
        json.dump(value, target, indent=2, ensure_ascii=False)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary_path, path)
