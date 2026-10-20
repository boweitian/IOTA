#!/bin/bash
#SBATCH --job-name=iota-identity-pilot
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/identity-pilot-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
.venv/bin/python identity_steering_pilot.py --out "$1"
