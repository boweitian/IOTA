#!/bin/bash
#SBATCH --job-name=iota-validation-repair
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --array=0-4%5
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/validation-guard-%A-%a.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
IOTA_PILOT_SUITES=(gsm8k bbh ifeval truthfulqa writingbench)
../steering_lab/.venv/bin/python validation_guard.py --run "$1" --suite "${IOTA_PILOT_SUITES[$SLURM_ARRAY_TASK_ID]}"
