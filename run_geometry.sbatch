#!/bin/bash
#SBATCH --job-name=iota-geometry
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/geometry-slurm-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
IOTA_GEOMETRY_OUTPUT="geometry_mvp/measurements/job-${SLURM_JOB_ID}"
/beacon-projects/worldmodel/btian/miniforge3/envs/pathorob/bin/python geometry_mvp/measure.py --output "$IOTA_GEOMETRY_OUTPUT"
