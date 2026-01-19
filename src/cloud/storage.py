"""
Google Cloud Storage utilities for experiment results.
"""

import os
import json
from typing import Optional
from pathlib import Path


class GCSStorage:
    """
    Google Cloud Storage wrapper for uploading/downloading experiment results.
    """

    def __init__(self, bucket_name: str, project_id: str):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self._client = None
        self._bucket = None

    @property
    def client(self):
        if self._client is None:
            from google.cloud import storage
            self._client = storage.Client(project=self.project_id)
        return self._client

    @property
    def bucket(self):
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    def ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        from google.cloud.exceptions import NotFound
        try:
            self.client.get_bucket(self.bucket_name)
            print(f"Bucket {self.bucket_name} exists")
        except NotFound:
            print(f"Creating bucket {self.bucket_name}...")
            self.client.create_bucket(self.bucket_name, location="us-central1")
            print(f"Created bucket {self.bucket_name}")

    def upload_file(self, local_path: str, remote_path: str):
        """Upload a file to GCS."""
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} -> gs://{self.bucket_name}/{remote_path}")

    def upload_directory(self, local_dir: str, remote_prefix: str):
        """Upload a directory to GCS."""
        local_path = Path(local_dir)
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(local_path)
                remote_path = f"{remote_prefix}/{relative}"
                self.upload_file(str(file_path), remote_path)

    def download_file(self, remote_path: str, local_path: str):
        """Download a file from GCS."""
        blob = self.bucket.blob(remote_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded gs://{self.bucket_name}/{remote_path} -> {local_path}")

    def download_directory(self, remote_prefix: str, local_dir: str):
        """Download all files with prefix to local directory."""
        blobs = self.client.list_blobs(self.bucket_name, prefix=remote_prefix)
        for blob in blobs:
            relative = blob.name[len(remote_prefix):].lstrip("/")
            if relative:
                local_path = os.path.join(local_dir, relative)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob.download_to_filename(local_path)
                print(f"Downloaded {blob.name} -> {local_path}")

    def list_files(self, prefix: str = "") -> list:
        """List files in bucket with prefix."""
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        return [blob.name for blob in blobs]

    def write_json(self, data: dict, remote_path: str):
        """Write JSON data directly to GCS."""
        blob = self.bucket.blob(remote_path)
        blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")

    def read_json(self, remote_path: str) -> dict:
        """Read JSON data from GCS."""
        blob = self.bucket.blob(remote_path)
        return json.loads(blob.download_as_string())
