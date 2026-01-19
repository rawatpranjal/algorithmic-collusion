"""Google Cloud Storage helpers for experiment code and results."""

import subprocess
import os
from datetime import datetime


class GCSBucket:
    """Manages GCS bucket for experiment artifacts."""

    def __init__(self, bucket_name: str, project_id: str):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.bucket_url = f"gs://{bucket_name}"

    def exists(self) -> bool:
        """Check if bucket exists."""
        result = subprocess.run(
            ["gsutil", "ls", "-b", self.bucket_url],
            capture_output=True, text=True
        )
        return result.returncode == 0

    def create(self):
        """Create the bucket if it doesn't exist."""
        if self.exists():
            print(f"Bucket {self.bucket_name} already exists")
            return

        print(f"Creating bucket {self.bucket_name}...")
        subprocess.run([
            "gsutil", "mb", "-p", self.project_id,
            "-l", "us-central1", self.bucket_url
        ], check=True)

    def upload_code(self, local_dir: str = ".") -> str:
        """Upload code tarball to bucket. Returns GCS path."""
        tarball = "/tmp/code.tar.gz"

        # Create tarball
        subprocess.run([
            "tar", "-czf", tarball,
            "-C", local_dir,
            "src", "scripts", "requirements.txt"
        ], check=True)

        # Upload
        gcs_path = f"{self.bucket_url}/code/code.tar.gz"
        subprocess.run(["gsutil", "cp", tarball, gcs_path], check=True)
        print(f"Code uploaded to {gcs_path}")

        return gcs_path

    def list_results(self, exp: int = None) -> list:
        """List available results."""
        path = f"{self.bucket_url}/results/"
        if exp:
            path += f"exp{exp}/"

        result = subprocess.run(
            ["gsutil", "ls", path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return []
        return result.stdout.strip().split('\n')

    def download_results(self, exp: int, timestamp: str, local_dir: str):
        """Download results for a specific run."""
        gcs_path = f"{self.bucket_url}/results/exp{exp}/{timestamp}/"
        os.makedirs(local_dir, exist_ok=True)
        subprocess.run([
            "gsutil", "-m", "cp", "-r", gcs_path, local_dir
        ], check=True)
        print(f"Results downloaded to {local_dir}")

    def get_log(self, exp: int, timestamp: str) -> str:
        """Get log contents for a run."""
        gcs_path = f"{self.bucket_url}/logs/exp{exp}-{timestamp}.log"
        result = subprocess.run(
            ["gsutil", "cat", gcs_path],
            capture_output=True, text=True
        )
        return result.stdout if result.returncode == 0 else f"Log not found: {gcs_path}"
