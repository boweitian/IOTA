#!/bin/bash
#SBATCH --job-name=iota-geometry-router
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/steering-lab-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
IOTA_LAB_RUN="${1:-runs/job-${SLURM_JOB_ID}}"
.venv/bin/python experiment.py --output "$IOTA_LAB_RUN"
.venv/bin/python report.py --run "$IOTA_LAB_RUN"
