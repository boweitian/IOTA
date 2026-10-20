#!/bin/bash
#SBATCH --job-name=iota-asset-replay
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/replay-assets-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
for IOTA_REPLAY_SUITE in gsm8k bbh ifeval truthfulqa; do
../steering_lab/.venv/bin/python replay_asset.py --config "runs/sample-assets-v1/replay/$IOTA_REPLAY_SUITE-config.json" --output "runs/sample-assets-v1/replay/$IOTA_REPLAY_SUITE-result.json"
done
