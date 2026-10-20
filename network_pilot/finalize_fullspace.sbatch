#!/bin/bash
#SBATCH --job-name=iota-full-audit
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/fullspace-finalize-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
task_python=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python
"$task_python" fullspace_analysis.py --diagnosis-summary
"$task_python" audit_fullspace_detailed.py
"$task_python" build_fullspace_view.py
