# Fire-and-Forget Cloud Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--detached` flag that launches experiments on GCE VMs that auto-run, upload results to GCS, and self-delete.

**Architecture:** Local CLI uploads code to GCS, creates VM with startup script metadata. VM downloads code, runs experiment, uploads results/logs to GCS, then self-deletes.

**Tech Stack:** Python 3, gcloud CLI, gsutil, GCE startup scripts

---

## Task 1: Add GCS Helper Module

**Files:**
- Create: `src/cloud/gcs.py`

**Step 1: Create GCS helper class**

```python
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
```

**Step 2: Commit**

```bash
git add src/cloud/gcs.py
git commit -m "feat(cloud): add GCS helper module for code/results storage"
```

---

## Task 2: Create Startup Script Generator

**Files:**
- Create: `src/cloud/startup.py`

**Step 1: Create startup script generator**

```python
"""Generate GCE startup scripts for experiments."""


def generate_startup_script(bucket_name: str, exp: int, quick: bool = False, parallel: bool = True) -> str:
    """Generate bash startup script for VM."""

    quick_flag = "--quick" if quick else ""
    parallel_flag = "--parallel" if parallel else ""

    script = f'''#!/bin/bash
set -e

# Redirect all output to log file
exec > /var/log/experiment.log 2>&1
echo "Starting experiment at $(date)"

# Install system dependencies
apt-get update
apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# Download code from GCS
echo "Downloading code from GCS..."
mkdir -p /opt/code
gsutil cp gs://{bucket_name}/code/code.tar.gz /tmp/code.tar.gz
tar -xzf /tmp/code.tar.gz -C /opt/code

# Install Python dependencies
echo "Installing Python dependencies..."
/opt/venv/bin/pip install -r /opt/code/requirements.txt

# Run experiment
echo "Running experiment {exp}..."
cd /opt/code
/opt/venv/bin/python scripts/run_experiment.py --exp {exp} {quick_flag} {parallel_flag}

# Upload results to GCS
echo "Uploading results..."
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
gsutil -m cp -r /opt/code/results/exp{exp} gs://{bucket_name}/results/exp{exp}/$TIMESTAMP/

# Upload log
echo "Uploading log..."
gsutil cp /var/log/experiment.log gs://{bucket_name}/logs/exp{exp}-$TIMESTAMP.log

# Self-delete
echo "Experiment complete. Self-deleting VM..."
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | cut -d/ -f4)
VM_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet
'''
    return script
```

**Step 2: Commit**

```bash
git add src/cloud/startup.py
git commit -m "feat(cloud): add startup script generator for fire-and-forget VMs"
```

---

## Task 3: Update VMConfig for Startup Scripts

**Files:**
- Modify: `src/cloud/vm.py`

**Step 1: Update VMConfig dataclass**

In `src/cloud/vm.py`, add `startup_script` field to VMConfig:

```python
@dataclass
class VMConfig:
    """Configuration for a GCE VM."""
    name: str = "collusion-runner"
    machine_type: str = "n2-standard-4"  # 4 vCPUs, 16GB RAM
    zone: str = "us-central1-a"
    image_family: str = "ubuntu-2204-lts"
    image_project: str = "ubuntu-os-cloud"
    disk_size_gb: int = 50
    startup_script: str = None  # Add this line
```

**Step 2: Update create() method to use startup script**

In the `create()` method, add startup script support:

```python
def create(self):
    """Create the VM."""
    if self.exists():
        print(f"VM {self.config.name} already exists")
        return

    print(f"Creating VM {self.config.name}...")

    cmd = [
        "compute", "instances", "create", self.config.name,
        f"--zone={self.config.zone}",
        f"--machine-type={self.config.machine_type}",
        f"--image-family={self.config.image_family}",
        f"--image-project={self.config.image_project}",
        f"--boot-disk-size={self.config.disk_size_gb}GB",
        "--scopes=cloud-platform",
    ]

    # Add startup script if provided
    if self.config.startup_script:
        cmd.append(f"--metadata=startup-script={self.config.startup_script}")

    self._run_gcloud(cmd)
    print(f"VM {self.config.name} created.")

    if not self.config.startup_script:
        print("Waiting for SSH...")
        time.sleep(30)  # Only wait if not using startup script
```

**Step 3: Commit**

```bash
git add src/cloud/vm.py
git commit -m "feat(cloud): add startup script support to VMConfig"
```

---

## Task 4: Add run_detached() Function

**Files:**
- Modify: `scripts/run_experiment.py`

**Step 1: Add imports at top of file**

After existing imports, add:

```python
from cloud.gcs import GCSBucket
from cloud.startup import generate_startup_script
```

**Step 2: Add run_detached function**

Add this function before `run_cloud()`:

```python
def run_detached(args):
    """Launch experiment on cloud VM in fire-and-forget mode."""
    from cloud.vm import CloudVM, VMConfig
    from cloud.gcs import GCSBucket
    from cloud.startup import generate_startup_script

    # Get project ID
    if args.project:
        project_id = args.project
    else:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True
        )
        project_id = result.stdout.strip()
        if not project_id:
            print("No GCP project specified. Use --project or run 'gcloud config set project PROJECT_ID'")
            sys.exit(1)

    print(f"Using GCP project: {project_id}")

    # Setup GCS bucket
    bucket_name = f"{project_id}-collusion-experiments"
    bucket = GCSBucket(bucket_name, project_id)
    bucket.create()

    # Upload code
    print("Uploading code to GCS...")
    bucket.upload_code()

    # Generate startup script
    startup_script = generate_startup_script(
        bucket_name=bucket_name,
        exp=args.exp,
        quick=args.quick,
        parallel=True
    )

    # Create VM with startup script
    vm_config = VMConfig(
        name=f"collusion-exp{args.exp}",
        machine_type="n2-standard-4",
        startup_script=startup_script
    )
    vm = CloudVM(project_id, vm_config)

    if vm.exists():
        print(f"VM {vm_config.name} already exists. Delete it first or use a different name.")
        sys.exit(1)

    vm.create()

    print(f"\n{'='*60}")
    print(f"Experiment {args.exp} launched in detached mode!")
    print(f"{'='*60}")
    print(f"\nVM: {vm_config.name}")
    print(f"Bucket: gs://{bucket_name}/")
    print(f"\nThe VM will:")
    print(f"  1. Install dependencies")
    print(f"  2. Run experiment {args.exp}")
    print(f"  3. Upload results to gs://{bucket_name}/results/exp{args.exp}/")
    print(f"  4. Upload logs to gs://{bucket_name}/logs/")
    print(f"  5. Self-delete")
    print(f"\nYou can close this terminal now.")
    print(f"\nTo check progress:")
    print(f"  gcloud compute instances list --filter='name~collusion-exp'")
    print(f"  gsutil ls gs://{bucket_name}/results/")
    print(f"  gsutil ls gs://{bucket_name}/logs/")
```

**Step 3: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat(cloud): add run_detached() for fire-and-forget experiments"
```

---

## Task 5: Add --detached CLI Flag

**Files:**
- Modify: `scripts/run_experiment.py`

**Step 1: Add --detached argument**

In `main()`, after the `--cloud` argument, add:

```python
parser.add_argument(
    "--detached", action="store_true",
    help="Fire-and-forget cloud mode: VM auto-runs, uploads to GCS, self-deletes"
)
```

**Step 2: Update main() to handle --detached**

In the cloud mode section of `main()`, update to:

```python
# Cloud mode
if args.cloud:
    if args.detached:
        run_detached(args)
    else:
        run_cloud(args)
    return
```

**Step 3: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat(cli): add --detached flag for fire-and-forget cloud mode"
```

---

## Task 6: Kill Current VMs and Test

**Step 1: Kill existing experiment VMs**

```bash
gcloud compute instances delete collusion-exp1 collusion-exp2 collusion-exp3 --zone=us-central1-a --quiet
```

**Step 2: Test with quick mode**

```bash
python3 scripts/run_experiment.py --exp 1 --cloud --detached --quick
```

**Step 3: Verify VM is running**

```bash
gcloud compute instances list --filter="name~collusion-exp"
```

**Step 4: Check VM serial output (startup script progress)**

```bash
gcloud compute instances get-serial-port-output collusion-exp1 --zone=us-central1-a | tail -50
```

---

## Task 7: Launch All Experiments

**Step 1: Launch all 3 experiments in detached mode**

```bash
python3 scripts/run_experiment.py --exp 1 --cloud --detached
python3 scripts/run_experiment.py --exp 2 --cloud --detached
python3 scripts/run_experiment.py --exp 3 --cloud --detached
```

**Step 2: Verify all VMs are running**

```bash
gcloud compute instances list --filter="name~collusion-exp"
```

**Step 3: Commit all changes**

```bash
git add -A
git commit -m "feat(cloud): complete fire-and-forget cloud implementation"
```

---

## Checking Results Later

```bash
# List completed results
gsutil ls gs://PROJECT-collusion-experiments/results/

# Download results
gsutil -m cp -r gs://PROJECT-collusion-experiments/results/exp1/TIMESTAMP/ ./results/exp1/

# View logs
gsutil cat gs://PROJECT-collusion-experiments/logs/exp1-TIMESTAMP.log
```
