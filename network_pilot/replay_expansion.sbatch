#!/bin/bash
#SBATCH --job-name=iota-replay-v2
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot/replay-v2-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
../steering_lab/.venv/bin/python replay_asset.py --config runs/expanded-assets-v2/replay/gsm3489-config.json --output runs/expanded-assets-v2/replay/gsm3489-replay.json --original-batch
../steering_lab/.venv/bin/python replay_asset.py --config runs/expanded-assets-v2/replay/gsm433-config.json --output runs/expanded-assets-v2/replay/gsm433-replay.json --original-batch
