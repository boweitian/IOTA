#!/bin/bash
#SBATCH --job-name=iota-top-three
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/top-three-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python -c 'from top_three_probe import run; run()'
/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python summarize_top_three.py
