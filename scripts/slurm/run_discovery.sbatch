#!/bin/bash
#SBATCH --job-name=iota-discovery
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/discovery-slurm-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
IOTA_PY=/beacon-projects/worldmodel/btian/miniforge3/envs/pathorob/bin/python
IOTA_CACHE=/beacon-projects/worldmodel/btian/hf_cache/hub
"$IOTA_PY" scripts/discover.py run --backend hf --model Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --cache-dir "$IOTA_CACHE" --budget 24 --seeds 0 1 2 --output output/discovery_qwen05_v2
"$IOTA_PY" scripts/discover.py run --backend hf --model Qwen/Qwen2.5-1.5B --revision 8faed761d45a263340a0528343f099c05c9a4323 --cache-dir "$IOTA_CACHE" --budget 24 --seeds 0 1 2 --output output/discovery_qwen15_v2
"$IOTA_PY" scripts/discover.py compare output/discovery_qwen05_v2/active_seed0_confirmation.json output/discovery_qwen15_v2/active_seed0_confirmation.json --output output/discovery_qwen_comparison_v2.json
