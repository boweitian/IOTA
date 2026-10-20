#!/bin/bash
#SBATCH --job-name=iota-studio-v2
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/studio-v2-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python -m unittest test_studio.py test_protocol.py
.venv/bin/python studio_experiment.py --run runs/qwen35-9b-studio-v2 --output studio_runs/character-9b-v2
