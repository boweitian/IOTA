#!/bin/bash
#SBATCH --job-name=iota-sample-assets
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%10
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/sample-assets-%A-%a.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
IOTA_SAMPLE_SUITES=(gsm8k bbh ifeval truthfulqa writingbench)
IOTA_SAMPLE_SUITE=${IOTA_SAMPLE_SUITES[$((SLURM_ARRAY_TASK_ID / 2))]}
IOTA_SAMPLE_SHARD=$((SLURM_ARRAY_TASK_ID % 2))
../steering_lab/.venv/bin/python sample_assets.py --suite "$IOTA_SAMPLE_SUITE" --shard "$IOTA_SAMPLE_SHARD" --shards 2
