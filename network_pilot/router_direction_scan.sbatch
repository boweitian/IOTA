#!/bin/bash
#SBATCH --job-name=iota-direction-scan
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/direction-scan-%A-%a.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
../steering_lab/.venv/bin/python -u router_direction_scan.py run --shard "$SLURM_ARRAY_TASK_ID"
