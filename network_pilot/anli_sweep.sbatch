#!/bin/bash
#SBATCH --job-name=iota-anli-r3-9b
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/anli-r3-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
../steering_lab/.venv/bin/python classification_sweep.py --out runs/classification9b-anli-r3-100-v1
