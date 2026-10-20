#!/bin/bash
#SBATCH --job-name=iota-instruct
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/instruct-discovery-slurm-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
IOTA_PY=/beacon-projects/worldmodel/btian/miniforge3/envs/pathorob/bin/python
"$IOTA_PY" scripts/discover.py run --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --cache-dir /beacon-projects/worldmodel/btian/hf_cache/hub --budget 24 --seeds 0 1 2 --output output/discovery_qwen05_instruct_v2
