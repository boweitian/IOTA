#!/bin/bash
#SBATCH --job-name=iota-assets-v2
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-24%25
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/assets-v2-%A-%a.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
IOTA_EXPANDED_SUITES=(gsm8k bbh ifeval truthfulqa writingbench)
IOTA_EXPANDED_SUITE=${IOTA_EXPANDED_SUITES[$((SLURM_ARRAY_TASK_ID / 5))]}
IOTA_EXPANDED_SHARD=$((SLURM_ARRAY_TASK_ID % 5))
../steering_lab/.venv/bin/python expanded_assets.py --suite "$IOTA_EXPANDED_SUITE" --shard "$IOTA_EXPANDED_SHARD" --shards 5
