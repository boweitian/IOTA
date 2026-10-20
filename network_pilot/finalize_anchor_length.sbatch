#!/bin/bash
#SBATCH --job-name=iota-anchor512-audit
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --account=angliece
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/anchor512-finalize-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python extend_anchor_generation.py --summarize
/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python audit_anchor_length.py
/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/.venv/bin/python ../steering_lab/build_studio.py
