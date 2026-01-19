"""
Google Compute Engine VM management for running experiments.
"""

import subprocess
import time
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class VMConfig:
    """Configuration for a GCE VM."""
    name: str = "collusion-runner"
    machine_type: str = "n2-standard-4"  # 4 vCPUs, 16GB RAM
    zone: str = "us-central1-a"
    image_family: str = "ubuntu-2204-lts"
    image_project: str = "ubuntu-os-cloud"
    disk_size_gb: int = 50


class CloudVM:
    """
    Manages a GCE VM for running experiments.
    """

    def __init__(self, project_id: str, config: Optional[VMConfig] = None):
        self.project_id = project_id
        self.config = config or VMConfig()
        self._exists = None

    def _run_gcloud(self, args: list, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        """Run a gcloud command."""
        cmd = ["gcloud"] + args + [f"--project={self.project_id}"]
        if capture:
            return subprocess.run(cmd, check=check, capture_output=True, text=True)
        return subprocess.run(cmd, check=check)

    def exists(self) -> bool:
        """Check if VM exists."""
        result = self._run_gcloud([
            "compute", "instances", "describe", self.config.name,
            f"--zone={self.config.zone}"
        ], check=False, capture=True)
        return result.returncode == 0

    def create(self):
        """Create the VM."""
        if self.exists():
            print(f"VM {self.config.name} already exists")
            return

        print(f"Creating VM {self.config.name}...")
        self._run_gcloud([
            "compute", "instances", "create", self.config.name,
            f"--zone={self.config.zone}",
            f"--machine-type={self.config.machine_type}",
            f"--image-family={self.config.image_family}",
            f"--image-project={self.config.image_project}",
            f"--boot-disk-size={self.config.disk_size_gb}GB",
            "--scopes=cloud-platform",
        ])
        print(f"VM {self.config.name} created. Waiting for SSH...")
        time.sleep(30)  # Wait for VM to be ready

    def delete(self):
        """Delete the VM."""
        if not self.exists():
            print(f"VM {self.config.name} doesn't exist")
            return

        print(f"Deleting VM {self.config.name}...")
        self._run_gcloud([
            "compute", "instances", "delete", self.config.name,
            f"--zone={self.config.zone}",
            "--quiet"
        ])
        print(f"VM {self.config.name} deleted")

    def ssh(self, command: str, stream: bool = True) -> subprocess.CompletedProcess:
        """Run a command on the VM via SSH."""
        cmd = [
            "gcloud", "compute", "ssh", self.config.name,
            f"--zone={self.config.zone}",
            f"--project={self.project_id}",
            "--command", command
        ]
        if stream:
            return subprocess.run(cmd)
        return subprocess.run(cmd, capture_output=True, text=True)

    def scp_to(self, local_path: str, remote_path: str):
        """Copy file to VM."""
        subprocess.run([
            "gcloud", "compute", "scp",
            local_path,
            f"{self.config.name}:{remote_path}",
            f"--zone={self.config.zone}",
            f"--project={self.project_id}",
            "--recurse"
        ], check=True)

    def scp_from(self, remote_path: str, local_path: str):
        """Copy file from VM."""
        subprocess.run([
            "gcloud", "compute", "scp",
            f"{self.config.name}:{remote_path}",
            local_path,
            f"--zone={self.config.zone}",
            f"--project={self.project_id}",
            "--recurse"
        ], check=True)

    def setup_environment(self):
        """Install Python and dependencies on the VM."""
        print("Setting up Python environment on VM...")

        # Install Python and pip
        self.ssh("sudo apt-get update && sudo apt-get install -y python3-pip python3-venv")

        # Create venv
        self.ssh("python3 -m venv ~/venv")

        print("Environment setup complete")

    def install_requirements(self, requirements_path: str = "requirements.txt"):
        """Install Python requirements on VM."""
        print("Installing Python requirements...")
        self.scp_to(requirements_path, "~/requirements.txt")
        self.ssh("~/venv/bin/pip install -r ~/requirements.txt")
        print("Requirements installed")

    def upload_code(self, local_dir: str = "."):
        """Upload experiment code to VM."""
        print("Uploading code to VM...")
        # Create tarball locally
        subprocess.run([
            "tar", "-czf", "/tmp/code.tar.gz",
            "-C", local_dir,
            "src", "scripts", "requirements.txt"
        ], check=True)

        self.scp_to("/tmp/code.tar.gz", "~/code.tar.gz")
        self.ssh("cd ~ && rm -rf code && mkdir code && tar -xzf code.tar.gz -C code")
        print("Code uploaded")

    def run_experiment(self, exp: int, quick: bool = True, parallel: bool = True, workers: int = 4) -> str:
        """Run an experiment on the VM."""
        cmd_parts = [
            "cd ~/code &&",
            "~/venv/bin/python scripts/run_experiment.py",
            f"--exp {exp}"
        ]
        if quick:
            cmd_parts.append("--quick")
        if parallel:
            cmd_parts.append(f"--parallel --workers {workers}")

        cmd = " ".join(cmd_parts)
        print(f"Running: {cmd}")
        self.ssh(cmd)

        return f"~/code/results/exp{exp}/quick_test" if quick else f"~/code/results/exp{exp}"

    def download_results(self, remote_dir: str, local_dir: str):
        """Download results from VM."""
        print(f"Downloading results from {remote_dir}...")
        os.makedirs(local_dir, exist_ok=True)
        self.scp_from(remote_dir, local_dir)
        print(f"Results downloaded to {local_dir}")
