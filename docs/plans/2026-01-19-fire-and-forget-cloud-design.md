# Fire-and-Forget Cloud Experiments

## Overview

Run experiments on GCE VMs that don't require an active local session. VMs auto-run experiments, upload results to GCS, and self-delete.

## Architecture

```
Local Machine                         Google Cloud
─────────────────────────────────────────────────────────────────

  run_experiment.py --cloud           GCS Bucket
         │                            ├── code/code.tar.gz
         │ 1. Upload code             ├── results/exp1/...
         ├─────────────────────────►  ├── results/exp2/...
         │                            └── logs/exp1-20260119.log
         │ 2. Create VM with
         │    startup script          GCE VM (ephemeral)
         ├─────────────────────────►  ┌─────────────────────┐
         │                            │ startup.sh:         │
         ▼                            │  1. Download code   │
      Done!                           │  2. Install deps    │
      (session can end)               │  3. Run experiment  │
                                      │  4. Upload results  │
                                      │  5. Upload logs     │
                                      │  6. Self-delete     │
                                      └─────────────────────┘
```

## GCS Bucket Structure

```
gs://collusion-experiments/
├── code/
│   └── code.tar.gz              # Uploaded by local machine
├── results/
│   ├── exp1/
│   │   └── 20260119-143022/     # Timestamped run
│   │       ├── data.csv
│   │       ├── param_mappings.json
│   │       └── trials/...
│   ├── exp2/...
│   └── exp3/...
└── logs/
    ├── exp1-20260119-143022.log
    ├── exp2-20260119-143025.log
    └── exp3-20260119-143028.log
```

## Startup Script

Embedded in VM metadata, runs on boot:

```bash
#!/bin/bash
exec > /var/log/experiment.log 2>&1  # Capture all output

# Install deps
apt-get update && apt-get install -y python3-pip python3-venv
python3 -m venv /opt/venv
gsutil cp gs://BUCKET/code/code.tar.gz /tmp/
mkdir -p /opt/code
tar -xzf /tmp/code.tar.gz -C /opt/code
/opt/venv/bin/pip install -r /opt/code/requirements.txt

# Run experiment (EXP_NUM passed via VM metadata)
EXP=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/attributes/exp -H "Metadata-Flavor: Google")
BUCKET=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket -H "Metadata-Flavor: Google")
ZONE=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H "Metadata-Flavor: Google" | cut -d/ -f4)

/opt/venv/bin/python /opt/code/scripts/run_experiment.py --exp $EXP --parallel

# Upload results & logs
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
gsutil -m cp -r /opt/code/results/exp$EXP gs://$BUCKET/results/exp$EXP/$TIMESTAMP/
gsutil cp /var/log/experiment.log gs://$BUCKET/logs/exp$EXP-$TIMESTAMP.log

# Self-delete
gcloud compute instances delete $(hostname) --zone=$ZONE --quiet
```

## CLI Usage

### Launch detached experiments
```bash
# Single experiment
python scripts/run_experiment.py --exp 1 --cloud --detached

# Multiple experiments (creates 3 VMs)
python scripts/run_experiment.py --exp 1 --cloud --detached
python scripts/run_experiment.py --exp 2 --cloud --detached
python scripts/run_experiment.py --exp 3 --cloud --detached
```

### Check results
```bash
# List completed runs
gsutil ls gs://collusion-experiments/results/

# Download specific run
gsutil -m cp -r gs://collusion-experiments/results/exp1/20260119-143022/ ./results/exp1/

# Check logs
gsutil cat gs://collusion-experiments/logs/exp1-20260119-143022.log

# Check running VMs
gcloud compute instances list --filter="name~collusion-exp"
```

## Implementation Tasks

1. Add `--detached` flag to CLI argument parser
2. Create `run_detached()` function in run_experiment.py
3. Add GCS bucket creation/upload helpers to src/cloud/
4. Generate startup script with embedded metadata
5. Update VMConfig to support startup scripts
6. Test with --quick flag first
