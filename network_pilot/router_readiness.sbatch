#!/bin/bash
#SBATCH --job-name=iota-readiness
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=0-4
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/readiness-%A-%a.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
# Beacon allows few concurrent jobs: array task k runs random-bank shards 3k..3k+2, then replication k+1.
for t in 0 1 2; do
  ../steering_lab/.venv/bin/python router_readiness.py random --task $((SLURM_ARRAY_TASK_ID * 3 + t))
done
../steering_lab/.venv/bin/python router_readiness.py replicate --task "$SLURM_ARRAY_TASK_ID"
