#!/bin/bash
#SBATCH --job-name=iota-heldout-audit
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/heldout-finalize-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
task_python=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python
"$task_python" heldout_precision.py --summarize
"$task_python" audit_heldout_precision.py
"$task_python" /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/build_studio.py
