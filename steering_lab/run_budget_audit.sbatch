#!/bin/bash
#SBATCH --job-name=iota-budget-audit
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/steering-budget-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
.venv/bin/python budget_audit.py --run "$1"
.venv/bin/python report.py --run "$1"
