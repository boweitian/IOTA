#!/bin/bash
#SBATCH --job-name=iota-judge-recovery
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/judge-recovery-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
../steering_lab/.venv/bin/python recover_writing_judge.py --suite writingbench --shard 1 --shards 2
