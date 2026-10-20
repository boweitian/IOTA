#!/bin/bash
#SBATCH --job-name=iota-network-pilot
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/pilot-%A-%a.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
IOTA_PILOT_SUITES=(gsm8k bbh ifeval truthfulqa writingbench)
IOTA_PILOT_RUN="${1:-runs/pilot-${SLURM_ARRAY_JOB_ID}}"
../steering_lab/.venv/bin/python experiment.py --suite "${IOTA_PILOT_SUITES[$SLURM_ARRAY_TASK_ID]}" --output "$IOTA_PILOT_RUN/${IOTA_PILOT_SUITES[$SLURM_ARRAY_TASK_ID]}"
