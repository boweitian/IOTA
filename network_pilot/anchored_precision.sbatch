#!/bin/bash
#SBATCH --job-name=iota-anchor
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/anchor-%A_%a.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
if [[ "${1:-}" == "fit" ]]; then
 /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python anchored_precision.py --fit
else
 /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python anchored_precision.py --worker "$SLURM_ARRAY_TASK_ID" --workers 5
fi
