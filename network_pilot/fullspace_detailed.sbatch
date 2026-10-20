#!/bin/bash
#SBATCH --job-name=iota-full-1200
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/fullspace-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
task_python=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 bash -c 'exec /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python fullspace_detailed.py --worker "$((SLURM_ARRAY_TASK_ID * 4 + SLURM_PROCID))"'
