#!/bin/bash
#SBATCH --job-name=iota-readiness
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00

#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/readiness-rep1-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
../steering_lab/.venv/bin/python router_readiness.py replicate --task 0
