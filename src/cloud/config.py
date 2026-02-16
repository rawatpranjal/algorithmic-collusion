"""
Configuration dataclasses for local and cloud execution.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class LocalConfig:
    """Configuration for local execution."""

    max_workers: Optional[int] = None  # Auto: cpu_count - 1
    output_dir: str = "results"

    def __post_init__(self):
        if self.max_workers is None:
            # Leave one CPU free for the main process
            self.max_workers = max(1, os.cpu_count() // 2)


@dataclass
class CloudConfig:
    """Configuration for Google Cloud execution."""

    project_id: str
    region: str = "us-central1"
    bucket_name: Optional[str] = None  # Auto: algorithmic-collusion-{project}
    machine_type: str = "n2-highmem-32"  # 32 vCPUs
    max_workers: int = 30

    def __post_init__(self):
        if self.bucket_name is None:
            # Sanitize project_id for bucket name
            safe_project = self.project_id.replace(':', '-').replace('.', '-').lower()
            self.bucket_name = f"algorithmic-collusion-{safe_project}"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    experiment_id: int  # 1, 2, or 3
    quick: bool = False
    output_dir: Optional[str] = None
    seed: int = 42

    # Experiment-specific parameters (filled based on experiment_id)
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.output_dir is None:
            suffix = "/quick_test" if self.quick else ""
            self.output_dir = f"results/exp{self.experiment_id}{suffix}"
