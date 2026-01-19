"""
Progress tracking for parallel experiment execution.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any
from threading import Lock
import time
import sys


@dataclass
class ProgressUpdate:
    """A single progress update from a worker."""
    worker_id: int
    task_id: str
    current: int
    total: int
    message: Optional[str] = None


class ProgressCallback:
    """
    Callback interface for progress reporting from experiments.

    Usage in experiments:
        def run_experiment(..., progress_callback=None):
            for i in range(total_iterations):
                # ... do work ...
                if progress_callback and i % 1000 == 0:
                    progress_callback(current=i, total=total_iterations)
    """

    def __init__(self, worker_id: int, task_id: str, tracker: 'ProgressTracker'):
        self.worker_id = worker_id
        self.task_id = task_id
        self.tracker = tracker

    def __call__(self, current: int, total: int, message: Optional[str] = None):
        """Report progress."""
        update = ProgressUpdate(
            worker_id=self.worker_id,
            task_id=self.task_id,
            current=current,
            total=total,
            message=message
        )
        self.tracker.update(update)


class ProgressTracker:
    """
    Aggregates progress from multiple parallel workers.

    Displays nested tqdm-style progress bars:
    - Overall progress (completed tasks / total tasks)
    - Per-worker progress bars
    """

    def __init__(self, total_tasks: int, desc: str = "Overall"):
        self.total_tasks = total_tasks
        self.desc = desc
        self.completed_tasks = 0
        self.worker_progress: Dict[int, ProgressUpdate] = {}
        self._lock = Lock()
        self._start_time = time.time()
        self._last_render = 0
        self._render_interval = 0.5  # seconds
        self._use_tqdm = True
        self._bars = {}

        # Try to import tqdm
        try:
            from tqdm import tqdm
            self._tqdm = tqdm
            self._overall_bar = tqdm(
                total=total_tasks,
                desc=desc,
                position=0,
                leave=True,
                unit="task"
            )
        except ImportError:
            self._use_tqdm = False
            self._tqdm = None
            self._overall_bar = None

    def update(self, progress: ProgressUpdate):
        """Update progress for a worker."""
        with self._lock:
            self.worker_progress[progress.worker_id] = progress
            self._maybe_render()

    def task_completed(self, worker_id: int, task_id: str):
        """Mark a task as completed."""
        with self._lock:
            self.completed_tasks += 1
            if worker_id in self.worker_progress:
                del self.worker_progress[worker_id]

            if self._use_tqdm and self._overall_bar:
                self._overall_bar.update(1)
                self._overall_bar.set_postfix({"completed": self.completed_tasks})

            # Close worker bar if exists
            if worker_id in self._bars:
                self._bars[worker_id].close()
                del self._bars[worker_id]

    def _maybe_render(self):
        """Render progress if enough time has passed."""
        now = time.time()
        if now - self._last_render < self._render_interval:
            return
        self._last_render = now
        self._render()

    def _render(self):
        """Render current progress state."""
        if not self._use_tqdm:
            self._render_simple()
            return

        # Update worker progress bars
        for worker_id, progress in self.worker_progress.items():
            if worker_id not in self._bars:
                self._bars[worker_id] = self._tqdm(
                    total=progress.total,
                    desc=f"Worker {worker_id}: {progress.task_id}",
                    position=worker_id + 1,
                    leave=False,
                    unit="iter"
                )

            bar = self._bars[worker_id]
            bar.n = progress.current
            bar.total = progress.total
            bar.set_description(f"Worker {worker_id}: {progress.task_id}")
            if progress.message:
                bar.set_postfix_str(progress.message)
            bar.refresh()

    def _render_simple(self):
        """Simple text-based progress rendering (fallback when tqdm unavailable)."""
        elapsed = time.time() - self._start_time

        lines = [f"\r{self.desc}: {self.completed_tasks}/{self.total_tasks} tasks"]

        for worker_id, progress in sorted(self.worker_progress.items()):
            pct = 100 * progress.current / progress.total if progress.total > 0 else 0
            lines.append(f"  Worker {worker_id} ({progress.task_id}): {progress.current}/{progress.total} ({pct:.1f}%)")

        lines.append(f"  Elapsed: {elapsed:.1f}s")

        sys.stdout.write("\033[K" + "\n".join(lines) + "\033[F" * (len(lines) - 1))
        sys.stdout.flush()

    def close(self):
        """Clean up progress display."""
        if self._use_tqdm:
            for bar in self._bars.values():
                bar.close()
            if self._overall_bar:
                self._overall_bar.close()
        else:
            elapsed = time.time() - self._start_time
            print(f"\n{self.desc}: Completed {self.completed_tasks}/{self.total_tasks} tasks in {elapsed:.1f}s")


class NoOpProgressCallback:
    """A no-op callback for when progress tracking is disabled."""

    def __call__(self, current: int, total: int, message: Optional[str] = None):
        pass
