#!/bin/bash
#SBATCH --job-name=iota-character-studio
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/character-studio-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
.venv/bin/python studio_experiment.py --run runs/job-2117172 --output studio_runs/character-v2
