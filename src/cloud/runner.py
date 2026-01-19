"""
Distributed runner for parallel experiment execution.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple
import multiprocessing as mp
import os
import sys
import traceback

from .config import LocalConfig, CloudConfig
from .progress import ProgressTracker, ProgressCallback, NoOpProgressCallback


@dataclass
class TaskResult:
    """Result from a single task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None


def _worker_wrapper(args: Tuple) -> TaskResult:
    """
    Wrapper function for worker processes.

    This runs in a subprocess and executes the actual task function.
    """
    task_id, func, kwargs, worker_id = args

    try:
        # Create a simple callback that doesn't use shared state
        # (multiprocessing doesn't support sharing the ProgressTracker)
        result = func(**kwargs)
        return TaskResult(
            task_id=task_id,
            success=True,
            result=result
        )
    except Exception as e:
        return TaskResult(
            task_id=task_id,
            success=False,
            error=str(e),
            traceback=traceback.format_exc()
        )


class DistributedRunner:
    """
    Unified runner for local parallel and cloud execution.

    Local mode: Uses concurrent.futures.ProcessPoolExecutor
    Cloud mode: SSH to GCE VM, run with multiprocessing (Phase 2)

    Example usage:
        runner = DistributedRunner(LocalConfig(max_workers=4))

        tasks = [
            {'task_id': 'exp1_fpa_none', 'func': run_case_study, 'kwargs': {...}},
            {'task_id': 'exp1_spa_none', 'func': run_case_study, 'kwargs': {...}},
        ]

        results = runner.run(tasks, desc="Experiment 1")
    """

    def __init__(self, config: LocalConfig):
        """
        Initialize the runner.

        Args:
            config: LocalConfig or CloudConfig instance
        """
        self.config = config
        self.is_cloud = isinstance(config, CloudConfig)

    def run(
        self,
        tasks: List[Dict[str, Any]],
        desc: str = "Running tasks"
    ) -> List[TaskResult]:
        """
        Run tasks in parallel.

        Args:
            tasks: List of task dicts with keys:
                - task_id: Unique identifier for the task
                - func: Callable to execute
                - kwargs: Dict of keyword arguments for func
            desc: Description for progress bar

        Returns:
            List of TaskResult objects
        """
        if self.is_cloud:
            return self._run_cloud(tasks, desc)
        return self._run_local(tasks, desc)

    def _run_local(
        self,
        tasks: List[Dict[str, Any]],
        desc: str
    ) -> List[TaskResult]:
        """Run tasks locally using ProcessPoolExecutor."""

        results = []
        n_tasks = len(tasks)
        max_workers = min(self.config.max_workers, n_tasks)

        print(f"\n{desc}: Running {n_tasks} tasks with {max_workers} workers")

        # Use spawn method for cleaner subprocess handling
        ctx = mp.get_context('spawn')

        # Prepare work items
        work_items = [
            (task['task_id'], task['func'], task['kwargs'], i % max_workers)
            for i, task in enumerate(tasks)
        ]

        # Create progress tracker
        tracker = ProgressTracker(total_tasks=n_tasks, desc=desc)

        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(_worker_wrapper, item): item[0]
                    for item in work_items
                }

                # Collect results as they complete
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if result.success:
                            tracker.task_completed(0, task_id)
                        else:
                            print(f"\nTask {task_id} failed: {result.error}")
                            if result.traceback:
                                print(result.traceback)

                    except Exception as e:
                        results.append(TaskResult(
                            task_id=task_id,
                            success=False,
                            error=str(e),
                            traceback=traceback.format_exc()
                        ))
                        print(f"\nTask {task_id} raised exception: {e}")
        finally:
            tracker.close()

        # Report summary
        successful = sum(1 for r in results if r.success)
        print(f"\n{desc}: Completed {successful}/{n_tasks} tasks successfully")

        return results

    def _run_cloud(
        self,
        tasks: List[Dict[str, Any]],
        desc: str
    ) -> List[TaskResult]:
        """Run tasks on Google Cloud (Phase 2 - stub)."""
        raise NotImplementedError(
            "Cloud execution is not yet implemented. "
            "Use LocalConfig for local parallel execution."
        )


def run_sequential(
    tasks: List[Dict[str, Any]],
    desc: str = "Running tasks"
) -> List[TaskResult]:
    """
    Run tasks sequentially (for comparison/debugging).

    Args:
        tasks: List of task dicts with keys:
            - task_id: Unique identifier for the task
            - func: Callable to execute
            - kwargs: Dict of keyword arguments for func
        desc: Description for progress display

    Returns:
        List of TaskResult objects
    """
    from tqdm import tqdm

    results = []

    for task in tqdm(tasks, desc=desc):
        task_id = task['task_id']
        func = task['func']
        kwargs = task['kwargs']

        try:
            result = func(**kwargs)
            results.append(TaskResult(
                task_id=task_id,
                success=True,
                result=result
            ))
        except Exception as e:
            results.append(TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                traceback=traceback.format_exc()
            ))
            print(f"\nTask {task_id} failed: {e}")

    return results
