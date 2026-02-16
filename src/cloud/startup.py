"""Generate GCE startup scripts for experiments."""


def generate_startup_script(bucket_name: str, exp: int, quick: bool = False, parallel: bool = True, runs: int = None) -> str:
    """Generate bash startup script for VM."""

    quick_flag = "--quick" if quick else ""
    parallel_flag = "--parallel" if parallel else ""
    runs_flag = f"--runs {runs}" if runs else ""

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
/opt/venv/bin/python scripts/run_experiment.py --exp {exp} {quick_flag} {parallel_flag} {runs_flag}

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
